# Copyright 2024 Alphaqraft
#
# Hierarchical PPO Training for Two-Agent Market Making
#
# Architecture:
# - Inventory Agent (slow, strategic): learns WHAT position to hold
#   - Updates every 100 steps (50 seconds)
#   - Reward: unrealized P&L delta (market direction)
#   - Observations: AMM flow, volatility, position state
#
# - MM Agent (fast, tactical): learns HOW to execute toward target
#   - Updates every step (500ms)
#   - Reward: realized P&L + spread capture + fees (execution quality)
#   - Observations: market microstructure + target from Inventory Agent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time

import litepool
from hierarchical_policy import HierarchicalPolicy, create_hierarchical_policy
from inventory_agent import INVENTORY_OBS_INDICES
from mm_agent import MARKET_OBS_DIM
from metric_logger import MetricLogger


# === Device setup ===
device = torch.device("cpu")
print(f"Using device: {device}")


# === Configuration ===
@dataclass
class HierarchicalConfig:
    # Environment
    num_envs: int = 6
    num_threads: int = 6
    n_steps: int = 4096
    max_episode_steps: int = 4096
    
    # Hierarchical
    inventory_update_freq: int = 100  # Every 100 steps = 50 seconds (5 ticks/step × 100ms/tick)
    
    # PPO hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.995
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 1
    minibatch_size: int = 128
    
    # Trading parameters
    base_spread_bps: float = 1.0
    min_size_pct: float = 1.0
    max_size_pct: float = 5.0
    balance: float = 20000.0
    
    # Training
    total_epochs: int = 10000
    save_interval: int = 100
    log_interval: int = 1


config = HierarchicalConfig()


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: np.ndarray
    actions: np.ndarray
    inv_actions: np.ndarray  # Inventory agent actions (targets)
    mm_rewards: np.ndarray   # MM agent rewards
    inv_rewards: np.ndarray  # Inventory agent rewards
    dones: np.ndarray
    log_probs_mm: np.ndarray
    log_probs_inv: np.ndarray
    values_mm: np.ndarray
    values_inv: np.ndarray
    # Info tracking for logging
    infos: List = None
    advantages_mm: np.ndarray = None
    advantages_inv: np.ndarray = None
    returns_mm: np.ndarray = None
    returns_inv: np.ndarray = None
    
    @classmethod
    def create(cls, n_steps: int, num_envs: int, obs_dim: int, action_dim: int):
        return cls(
            observations=np.zeros((n_steps, num_envs, obs_dim), dtype=np.float32),
            actions=np.zeros((n_steps, num_envs, action_dim), dtype=np.float32),
            inv_actions=np.zeros((n_steps, num_envs, 1), dtype=np.float32),
            mm_rewards=np.zeros((n_steps, num_envs), dtype=np.float32),
            inv_rewards=np.zeros((n_steps, num_envs), dtype=np.float32),
            dones=np.zeros((n_steps, num_envs), dtype=np.float32),
            log_probs_mm=np.zeros((n_steps, num_envs), dtype=np.float32),
            log_probs_inv=np.zeros((n_steps, num_envs), dtype=np.float32),
            values_mm=np.zeros((n_steps, num_envs), dtype=np.float32),
            values_inv=np.zeros((n_steps, num_envs), dtype=np.float32),
            infos=[],
        )
    
    def compute_gae(self, last_values_mm: np.ndarray, last_values_inv: np.ndarray,
                    gamma: float, gae_lambda: float):
        """Compute GAE for both agents."""
        n_steps = self.mm_rewards.shape[0]
        
        # GAE for MM agent
        self.advantages_mm = np.zeros_like(self.mm_rewards)
        last_gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values_mm
            else:
                next_values = self.values_mm[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.mm_rewards[t] + gamma * next_values * next_non_terminal - self.values_mm[t]
            self.advantages_mm[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns_mm = self.advantages_mm + self.values_mm
        
        # GAE for Inventory agent
        self.advantages_inv = np.zeros_like(self.inv_rewards)
        last_gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values_inv
            else:
                next_values = self.values_inv[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.inv_rewards[t] + gamma * next_values * next_non_terminal - self.values_inv[t]
            self.advantages_inv[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns_inv = self.advantages_inv + self.values_inv


@dataclass
class EpisodeInfo:
    """Track info for a completed episode."""
    env_id: int
    steps: int
    mm_reward: float
    inv_reward: float
    total_reward: float
    realized_pnl: float
    unrealized_pnl: float
    spread_capture: float  # LIFO round-trip profit
    fees: float
    trade_count: int
    net_amount_btc: float


class HierarchicalPPOTrainer:
    """Trainer for hierarchical two-agent PPO."""
    
    def __init__(self, config: HierarchicalConfig):
        self.config = config
        
        # Create environment
        self.env = self._create_env()
        
        # Create hierarchical policy
        self.policy = create_hierarchical_policy(
            inventory_update_freq=config.inventory_update_freq,
            device=str(device),
        )
        
        # Separate optimizers for each agent
        self.optimizer_inv = optim.Adam(
            self.policy.inventory_agent.parameters(),
            lr=config.learning_rate,
        )
        self.optimizer_mm = optim.Adam(
            self.policy.mm_agent.parameters(),
            lr=config.learning_rate,
        )
        
        # Metrics
        self.logger = MetricLogger(print_interval=1024)
        self.episode_rewards = deque(maxlen=100)
        self.episode_mm_rewards = deque(maxlen=100)
        self.episode_inv_rewards = deque(maxlen=100)
        self.completed_episodes: List[EpisodeInfo] = []
        
        # Rollout buffer
        obs_dim = 36  # 13 market + 4 AMM + 8 trade + 11 agent state
        action_dim = 4
        self.buffer = RolloutBuffer.create(
            config.n_steps, config.num_envs, obs_dim, action_dim
        )
        
        # Tracking
        self.global_step = 0
        self.epochs_completed = 0
        self.best_reward = float('-inf')
        
        # Per-env tracking
        self.episode_steps = np.zeros(config.num_envs, dtype=np.int32)
        self.episode_mm_total = np.zeros(config.num_envs)
        self.episode_inv_total = np.zeros(config.num_envs)
        
        # Results directory
        self.results_dir = Path("results/hierarchical")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_env(self):
        """Create the litepool environment."""
        return litepool.make(
            "RlTrader-v0",
            env_type="gymnasium",
            num_envs=self.config.num_envs,
            batch_size=self.config.num_envs,
            num_threads=self.config.num_threads,
            is_prod=False,
            is_inverse_instr=False,
            api_key="",
            api_secret="",
            symbol="BTC_USDC-PERPETUAL",
            hedge_symbol="BTC_USDC-18APR25",
            tick_size=0.5,
            min_amount=0.0001,
            maker_fee=-0.000025,
            taker_fee=0.0005,
            foldername="/home/pravin/dev/alphaqraft_litepool/data/training/",
            balance=self.config.balance,
            start=360000,
            max_episode_steps=self.config.max_episode_steps,
            base_spread_bps=self.config.base_spread_bps,
            min_size_pct=self.config.min_size_pct,
            max_size_pct=self.config.max_size_pct,
        )
    
    def _extract_info_value(self, env_info: dict, key: str, env_id: int, default=0.0) -> float:
        """Safely extract a value from env_info for a specific environment."""
        val = env_info.get(key, default)
        if isinstance(val, np.ndarray):
            return float(val[env_id]) if env_id < len(val) else default
        return float(val) if val is not None else default
    
    def collect_rollout(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect one rollout of experience."""
        obs, _ = self.env.reset()
        self.policy.reset(self.config.num_envs)
        
        # Reset per-env tracking
        self.episode_steps.fill(0)
        self.episode_mm_total.fill(0)
        self.episode_inv_total.fill(0)
        self.completed_episodes.clear()
        self.buffer.infos = []
        
        for step in range(self.config.n_steps):
            # Get action from hierarchical policy
            action, info = self.policy.get_action(obs)
            
            # Store experience
            self.buffer.observations[step] = obs
            self.buffer.actions[step] = action
            self.buffer.inv_actions[step] = info['targets']
            self.buffer.log_probs_mm[step] = info['log_prob_mm'].squeeze()
            self.buffer.log_probs_inv[step] = info['log_prob_inv'].squeeze()
            self.buffer.values_mm[step] = info['value_mm'].squeeze()
            self.buffer.values_inv[step] = info['value_inv'].squeeze()
            
            # Step environment
            next_obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated | truncated
            
            # Store info for logging
            self.buffer.infos.append(env_info)
            
            # Extract separate rewards from env_info
            mm_reward = env_info.get('mm_reward', np.zeros(self.config.num_envs))
            inv_reward = env_info.get('inv_reward', np.zeros(self.config.num_envs))
            
            # Handle numpy array extraction
            if isinstance(mm_reward, np.ndarray):
                mm_reward = mm_reward.flatten()
            else:
                mm_reward = np.array([mm_reward] * self.config.num_envs)
            if isinstance(inv_reward, np.ndarray):
                inv_reward = inv_reward.flatten()
            else:
                inv_reward = np.array([inv_reward] * self.config.num_envs)
            
            self.buffer.mm_rewards[step] = mm_reward
            self.buffer.inv_rewards[step] = inv_reward
            self.buffer.dones[step] = done
            
            # Accumulate per-env rewards
            self.episode_mm_total += mm_reward
            self.episode_inv_total += inv_reward
            self.episode_steps += 1
            
            # Handle episode ends
            for env_id in range(self.config.num_envs):
                if done[env_id]:
                    # Extract terminal info from final_* fields
                    episode_info = EpisodeInfo(
                        env_id=env_id,
                        steps=int(self.episode_steps[env_id]),
                        mm_reward=float(self.episode_mm_total[env_id]),
                        inv_reward=float(self.episode_inv_total[env_id]),
                        total_reward=float(self.episode_mm_total[env_id] + self.episode_inv_total[env_id]),
                        realized_pnl=self._extract_info_value(env_info, 'final_realized_pnl', env_id),
                        unrealized_pnl=self._extract_info_value(env_info, 'final_unrealized_pnl', env_id),
                        spread_capture=self._extract_info_value(env_info, 'final_spread_capture', env_id),
                        fees=self._extract_info_value(env_info, 'final_fees', env_id),
                        trade_count=int(self._extract_info_value(env_info, 'final_trade_count', env_id)),
                        net_amount_btc=self._extract_info_value(env_info, 'final_net_amount_btc', env_id),
                    )
                    self.completed_episodes.append(episode_info)
                    
                    # Add to running averages
                    self.episode_mm_rewards.append(episode_info.mm_reward)
                    self.episode_inv_rewards.append(episode_info.inv_reward)
                    self.episode_rewards.append(episode_info.total_reward)
                    
                    # Reset per-env tracking
                    self.episode_steps[env_id] = 0
                    self.episode_mm_total[env_id] = 0
                    self.episode_inv_total[env_id] = 0
                    self.policy.reset_env(env_id)
            
            obs = next_obs
            self.global_step += self.config.num_envs
        
        # Get last values for GAE
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, log_probs, values, _ = self.policy.forward(obs_tensor)
            last_values_mm = values['mm'].cpu().numpy().squeeze()
            last_values_inv = values['inventory'].cpu().numpy().squeeze()
        
        # Compute GAE
        self.buffer.compute_gae(
            last_values_mm, last_values_inv,
            self.config.gamma, self.config.gae_lambda
        )
        
        return last_values_mm, last_values_inv
    
    def update(self) -> Dict[str, float]:
        """Update both agents using PPO."""
        # Flatten buffer
        n_samples = self.config.n_steps * self.config.num_envs
        
        obs_flat = self.buffer.observations.reshape(n_samples, -1)
        actions_flat = self.buffer.actions.reshape(n_samples, -1)
        inv_actions_flat = self.buffer.inv_actions.reshape(n_samples, -1)
        
        old_log_probs_mm = self.buffer.log_probs_mm.reshape(n_samples)
        old_log_probs_inv = self.buffer.log_probs_inv.reshape(n_samples)
        
        advantages_mm = self.buffer.advantages_mm.reshape(n_samples)
        advantages_inv = self.buffer.advantages_inv.reshape(n_samples)
        returns_mm = self.buffer.returns_mm.reshape(n_samples)
        returns_inv = self.buffer.returns_inv.reshape(n_samples)
        
        # Normalize advantages
        advantages_mm = (advantages_mm - advantages_mm.mean()) / (advantages_mm.std() + 1e-8)
        advantages_inv = (advantages_inv - advantages_inv.mean()) / (advantages_inv.std() + 1e-8)
        
        # Convert to tensors
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
        # MM actions: [bid_spread, ask_spread, requote] = indices [0, 1, 3]
        mm_actions = np.concatenate([actions_flat[:, 0:2], actions_flat[:, 3:4]], axis=1)
        actions_t = torch.as_tensor(mm_actions, dtype=torch.float32, device=device)
        inv_actions_t = torch.as_tensor(inv_actions_flat, dtype=torch.float32, device=device)
        old_log_probs_mm_t = torch.as_tensor(old_log_probs_mm, dtype=torch.float32, device=device)
        old_log_probs_inv_t = torch.as_tensor(old_log_probs_inv, dtype=torch.float32, device=device)
        advantages_mm_t = torch.as_tensor(advantages_mm, dtype=torch.float32, device=device)
        advantages_inv_t = torch.as_tensor(advantages_inv, dtype=torch.float32, device=device)
        returns_mm_t = torch.as_tensor(returns_mm, dtype=torch.float32, device=device)
        returns_inv_t = torch.as_tensor(returns_inv, dtype=torch.float32, device=device)
        
        losses = {'policy_loss_mm': 0, 'value_loss_mm': 0, 'entropy_mm': 0,
                  'policy_loss_inv': 0, 'value_loss_inv': 0, 'entropy_inv': 0}
        n_updates = 0
        
        for _ in range(self.config.update_epochs):
            # Random permutation for minibatches
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                obs_batch = obs_t[batch_indices]
                actions_batch = actions_t[batch_indices]
                inv_actions_batch = inv_actions_t[batch_indices]
                old_log_probs_mm_batch = old_log_probs_mm_t[batch_indices]
                old_log_probs_inv_batch = old_log_probs_inv_t[batch_indices]
                advantages_mm_batch = advantages_mm_t[batch_indices]
                advantages_inv_batch = advantages_inv_t[batch_indices]
                returns_mm_batch = returns_mm_t[batch_indices]
                returns_inv_batch = returns_inv_t[batch_indices]
                
                # Evaluate actions
                eval_results = self.policy.evaluate_actions(
                    obs_batch, actions_batch, inv_actions_batch
                )
                
                # === Update MM Agent ===
                log_prob_mm, entropy_mm, value_mm = eval_results['mm']
                log_prob_mm = log_prob_mm.squeeze()
                entropy_mm = entropy_mm.mean()
                value_mm = value_mm.squeeze()
                
                # PPO clipped objective
                ratio_mm = torch.exp(log_prob_mm - old_log_probs_mm_batch)
                surr1_mm = ratio_mm * advantages_mm_batch
                surr2_mm = torch.clamp(ratio_mm, 1 - self.config.clip_range, 
                                       1 + self.config.clip_range) * advantages_mm_batch
                policy_loss_mm = -torch.min(surr1_mm, surr2_mm).mean()
                
                value_loss_mm = nn.functional.mse_loss(value_mm, returns_mm_batch)
                
                loss_mm = (policy_loss_mm 
                          + self.config.value_coef * value_loss_mm 
                          - self.config.entropy_coef * entropy_mm)
                
                self.optimizer_mm.zero_grad()
                loss_mm.backward()
                nn.utils.clip_grad_norm_(self.policy.mm_agent.parameters(), 
                                         self.config.max_grad_norm)
                self.optimizer_mm.step()
                
                # === Update Inventory Agent ===
                log_prob_inv, entropy_inv, value_inv = eval_results['inventory']
                log_prob_inv = log_prob_inv.squeeze()
                entropy_inv = entropy_inv.mean()
                value_inv = value_inv.squeeze()
                
                ratio_inv = torch.exp(log_prob_inv - old_log_probs_inv_batch)
                surr1_inv = ratio_inv * advantages_inv_batch
                surr2_inv = torch.clamp(ratio_inv, 1 - self.config.clip_range,
                                        1 + self.config.clip_range) * advantages_inv_batch
                policy_loss_inv = -torch.min(surr1_inv, surr2_inv).mean()
                
                value_loss_inv = nn.functional.mse_loss(value_inv, returns_inv_batch)
                
                loss_inv = (policy_loss_inv
                           + self.config.value_coef * value_loss_inv
                           - self.config.entropy_coef * entropy_inv)
                
                self.optimizer_inv.zero_grad()
                loss_inv.backward()
                nn.utils.clip_grad_norm_(self.policy.inventory_agent.parameters(),
                                         self.config.max_grad_norm)
                self.optimizer_inv.step()
                
                # Track losses
                losses['policy_loss_mm'] += policy_loss_mm.item()
                losses['value_loss_mm'] += value_loss_mm.item()
                losses['entropy_mm'] += entropy_mm.item()
                losses['policy_loss_inv'] += policy_loss_inv.item()
                losses['value_loss_inv'] += value_loss_inv.item()
                losses['entropy_inv'] += entropy_inv.item()
                n_updates += 1
        
        # Average losses
        if n_updates > 0:
            for key in losses:
                losses[key] /= n_updates
        
        return losses
    
    def _log_epoch(self, epoch: int, losses: Dict[str, float]):
        """Log epoch statistics."""
        # Compute averages from completed episodes
        if self.completed_episodes:
            avg_mm_reward = np.mean([e.mm_reward for e in self.completed_episodes])
            avg_inv_reward = np.mean([e.inv_reward for e in self.completed_episodes])
            avg_realized_pnl = np.mean([e.realized_pnl for e in self.completed_episodes])
            avg_unrealized_pnl = np.mean([e.unrealized_pnl for e in self.completed_episodes])
            total_trades = sum(e.trade_count for e in self.completed_episodes)
        else:
            avg_mm_reward = np.mean(self.episode_mm_rewards) if self.episode_mm_rewards else 0
            avg_inv_reward = np.mean(self.episode_inv_rewards) if self.episode_inv_rewards else 0
            avg_realized_pnl = 0
            avg_unrealized_pnl = 0
            total_trades = 0
        
        # Compute action statistics
        actions = self.buffer.actions.reshape(-1, 4)
        requote_actions = actions[:, 3]
        requote_rate = (requote_actions > 0).mean()
        
        # Spread actions when requoting
        requote_mask = requote_actions > 0
        if requote_mask.sum() > 0:
            quote_actions = actions[requote_mask]
            avg_bid_spread = quote_actions[:, 0].mean()
            avg_ask_spread = quote_actions[:, 1].mean()
            avg_target = quote_actions[:, 2].mean()
        else:
            avg_bid_spread = 0
            avg_ask_spread = 0
            avg_target = 0
        
        # Buy/sell breakdown from last info
        buy_trades = 0
        sell_trades = 0
        if self.buffer.infos and len(self.buffer.infos) > 0:
            last_info = self.buffer.infos[-1]
            if 'buy_trades' in last_info:
                buy_val = last_info['buy_trades']
                if isinstance(buy_val, np.ndarray):
                    buy_trades = int(buy_val.sum())
                else:
                    buy_trades = int(buy_val)
            if 'sell_trades' in last_info:
                sell_val = last_info['sell_trades']
                if isinstance(sell_val, np.ndarray):
                    sell_trades = int(sell_val.sum())
                else:
                    sell_trades = int(sell_val)
        
        # Get action standard deviations from policies
        mm_spread_std = torch.exp(self.policy.mm_agent.spread_log_std).mean().item()
        inv_std = torch.exp(self.policy.inventory_agent.actor_log_std).mean().item()
        
        # Print epoch summary with full learning diagnostics
        print(f"Epoch {epoch:5d} | "
              f"Step {self.global_step:8d} | "
              f"MM.Rew {avg_mm_reward:8.2f} | "
              f"Inv.Rew {avg_inv_reward:8.2f} | "
              f"R.PnL ${avg_realized_pnl:7.2f} | "
              f"U.PnL ${avg_unrealized_pnl:7.2f} | "
              f"ReqRate {requote_rate:.1%} | "
              f"Trades {total_trades:4d} (B:{buy_trades}/S:{sell_trades})")
        print(f"         Loss: PL_mm {losses['policy_loss_mm']:.4f} VL_mm {losses['value_loss_mm']:.4f} | "
              f"PL_inv {losses['policy_loss_inv']:.4f} VL_inv {losses['value_loss_inv']:.4f} | "
              f"Ent {losses['entropy_mm']:.3f}/{losses['entropy_inv']:.3f} | "
              f"Std {mm_spread_std:.3f}/{inv_std:.3f}")
        
        # Print completed episodes
        if self.completed_episodes:
            print(f"\n  Completed {len(self.completed_episodes)} episode(s):")
            for ep in self.completed_episodes:
                net_pnl = ep.realized_pnl + ep.unrealized_pnl + ep.fees
                print(f"  [Episode] Env {ep.env_id} | "
                      f"Steps {ep.steps:5d} | "
                      f"MM.Rew {ep.mm_reward:7.2f} | "
                      f"Inv.Rew {ep.inv_reward:7.2f} | "
                      f"SprdCap ${ep.spread_capture:6.2f} | "
                      f"U.PnL ${ep.unrealized_pnl:7.2f} | "
                      f"Fees ${ep.fees:5.2f} | "
                      f"Net ${net_pnl:7.2f} | "
                      f"Trades {ep.trade_count:4d} | "
                      f"Pos {ep.net_amount_btc:+.5f} BTC")
            print()
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("Hierarchical PPO Training - Two-Agent Market Making")
        print("="*80)
        print(f"Inventory Agent: updates every {self.config.inventory_update_freq} steps (50 sec)")
        print(f"MM Agent: updates every step (500ms)")
        print(f"Steps per epoch: {self.config.n_steps}")
        print(f"Total epochs: {self.config.total_epochs}")
        print(f"Observations: 36 signals (13 market + 4 AMM + 8 trade + 11 agent state)")
        print(f"Actions: 4 (bid_spread, ask_spread, target_inventory, requote)")
        print("="*80 + "\n")
        
        for epoch in range(self.config.total_epochs):
            epoch_start = time.time()
            
            # Collect rollout
            self.collect_rollout()
            
            # Update policy
            losses = self.update()
            
            self.epochs_completed += 1
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self._log_epoch(epoch, losses)
            
            # Track best model
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save_checkpoint(str(self.results_dir / "best_model.pt"))
            
            # Save checkpoint periodically
            if epoch % self.config.save_interval == 0 and epoch > 0:
                self.save_checkpoint(str(self.results_dir / f"epoch_{epoch}.pt"))
            
            # Clear infos to prevent memory accumulation
            self.buffer.infos = []
            
            # GC periodically
            if epoch % 100 == 0:
                gc.collect()
        
        print("\nTraining complete!")
        print(f"Best reward: {self.best_reward:.4f}")
        self.save_checkpoint(str(self.results_dir / "final_model.pt"))
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'inventory_agent': self.policy.inventory_agent.state_dict(),
            'mm_agent': self.policy.mm_agent.state_dict(),
            'optimizer_inv': self.optimizer_inv.state_dict(),
            'optimizer_mm': self.optimizer_mm.state_dict(),
            'global_step': self.global_step,
            'epochs_completed': self.epochs_completed,
            'best_reward': self.best_reward,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.policy.inventory_agent.load_state_dict(checkpoint['inventory_agent'])
        self.policy.mm_agent.load_state_dict(checkpoint['mm_agent'])
        self.optimizer_inv.load_state_dict(checkpoint['optimizer_inv'])
        self.optimizer_mm.load_state_dict(checkpoint['optimizer_mm'])
        self.global_step = checkpoint['global_step']
        self.epochs_completed = checkpoint['epochs_completed']
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        print(f"Loaded checkpoint from {path}")
        print(f"  Resuming from epoch {self.epochs_completed}, step {self.global_step}")


def main():
    """Main entry point."""
    # Create trainer
    trainer = HierarchicalPPOTrainer(config)
    
    # Check for existing checkpoint
    checkpoint_path = Path("results/hierarchical/latest.pt")
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}")
        trainer.load_checkpoint(str(checkpoint_path))
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
