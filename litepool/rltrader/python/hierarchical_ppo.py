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

# Ensure we use the local litepool, not system-installed version
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels to project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
from datetime import datetime

import litepool
from torch.utils.tensorboard import SummaryWriter
from hierarchical_policy import HierarchicalPolicy, create_hierarchical_policy
from inventory_agent import INVENTORY_OBS_INDICES
from mm_agent import MARKET_OBS_DIM
from metric_logger import MetricLogger
from hierarchical_config import HierarchicalConfig

# === Device setup ===
device = torch.device("cpu")
print(f"Using device: {device}")


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
    inv_decision_mask: np.ndarray  # Mask: 1 if inventory decision was made at this timestep, 0 otherwise
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
            inv_decision_mask=np.zeros((n_steps, num_envs), dtype=np.float32),
            infos=[],
        )
    
    def compute_gae(self, last_values_mm: np.ndarray, last_values_inv: np.ndarray,
                    gamma: float, gae_lambda: float, inventory_update_freq: int):
        """
        Compute GAE for both agents.
        
        For inventory agent: Use effective gamma^update_freq to account for temporal mismatch.
        Decisions are made every update_freq steps, so rewards accumulate over that period.
        """
        n_steps = self.mm_rewards.shape[0]
        num_envs = self.mm_rewards.shape[1]
        
        # GAE for MM agent (standard, updates every step)
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
        
        # GAE for Inventory agent (account for temporal mismatch)
        # Effective discount: gamma^update_freq because decisions happen every update_freq steps
        effective_gamma = gamma ** inventory_update_freq
        effective_gae_lambda = gae_lambda ** inventory_update_freq
        
        self.advantages_inv = np.zeros_like(self.inv_rewards)
        
        # Compute GAE only at decision boundaries, then propagate to intermediate steps
        # For each environment, find decision timesteps and compute GAE there
        for env_id in range(num_envs):
            # Find decision timesteps for this environment
            decision_timesteps = np.where(self.inv_decision_mask[:, env_id] > 0.5)[0]
            
            if len(decision_timesteps) == 0:
                # No decisions in this rollout for this env, set advantages to 0
                self.advantages_inv[:, env_id] = 0.0
                continue
            
            # Compute GAE backwards from last decision to first
            last_gae = 0.0
            for i in reversed(range(len(decision_timesteps))):
                t = decision_timesteps[i]
                
                # Find next decision timestep (or end of rollout)
                if i == len(decision_timesteps) - 1:
                    # Last decision: use last value estimate
                    next_values = last_values_inv[env_id] if t == n_steps - 1 else self.values_inv[t + 1, env_id]
                else:
                    # Next decision timestep
                    next_t = decision_timesteps[i + 1]
                    next_values = self.values_inv[next_t, env_id]
                
                # Accumulate rewards between this decision and next (or end)
                if i == len(decision_timesteps) - 1:
                    # Last decision: accumulate to end of rollout
                    reward_sum = np.sum(self.inv_rewards[t:, env_id])
                    # Check if episode ended
                    episode_ended = np.any(self.dones[t:, env_id])
                    next_non_terminal = 1.0 - (1.0 if episode_ended else 0.0)
                else:
                    # Accumulate rewards to next decision
                    next_t = decision_timesteps[i + 1]
                    reward_sum = np.sum(self.inv_rewards[t:next_t, env_id])
                    next_non_terminal = 1.0
                
                # Compute delta and GAE
                delta = reward_sum + effective_gamma * next_values * next_non_terminal - self.values_inv[t, env_id]
                gae = delta + effective_gamma * effective_gae_lambda * next_non_terminal * last_gae
                self.advantages_inv[t, env_id] = gae
                last_gae = gae
                
                # Propagate advantage to intermediate steps (use same advantage for all steps until next decision)
                if i < len(decision_timesteps) - 1:
                    next_t = decision_timesteps[i + 1]
                    # Use decaying advantage for intermediate steps
                    for intermediate_t in range(t + 1, next_t):
                        steps_away = intermediate_t - t
                        decay_factor = gamma ** steps_away
                        self.advantages_inv[intermediate_t, env_id] = gae * decay_factor
                else:
                    # Last decision: propagate to end
                    for intermediate_t in range(t + 1, n_steps):
                        steps_away = intermediate_t - t
                        decay_factor = gamma ** steps_away
                        # Also account for episode termination
                        if self.dones[intermediate_t, env_id]:
                            break
                        self.advantages_inv[intermediate_t, env_id] = gae * decay_factor
        
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
            target_range=config.target_range,
        )
        
        # Separate optimizers for each agent
        self.optimizer_inv = optim.Adam(
            self.policy.inventory_agent.parameters(),
            lr=config.inv_learning_rate,  # Higher LR for noisier reward signal
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
        obs_dim = 40  # 13 market + 4 AMM + 8 trade + 11 agent state + 1 previous spread + 2 bid/ask distances + 1 mid_change
        action_dim = 3  # bid_spread, ask_spread, target_inventory (requote removed)
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
        self.current_obs = None  # Track current observation across epochs
        
        # Results directory
        self.results_dir = Path("results/hierarchical")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logging
        run_name = f"hierarchical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tb_writer = SummaryWriter(log_dir=f"runs/{run_name}")
        print(f"TensorBoard logging to: runs/{run_name}")
        print("View with: tensorboard --logdir=runs/")
    
    def _get_entropy_coef_mm(self) -> float:
        """Get current entropy coefficient for MM agent with annealing."""
        # Linear annealing: start at initial value, decay to 0.01 over total training
        initial_coef = self.config.entropy_coef_mm
        final_coef = 0.01
        total_steps = self.config.total_epochs * self.config.n_steps * self.config.num_envs
        progress = min(1.0, self.global_step / (total_steps * 0.5))  # Anneal over first 50% of training
        return initial_coef * (1.0 - progress) + final_coef * progress
    
    def _get_entropy_coef_inv(self) -> float:
        """Get current entropy coefficient for Inventory agent with annealing."""
        # Linear annealing: start at initial value, decay to 0.01 over total training
        initial_coef = self.config.entropy_coef_inv
        final_coef = 0.01
        total_steps = self.config.total_epochs * self.config.n_steps * self.config.num_envs
        progress = min(1.0, self.global_step / (total_steps * 0.5))  # Anneal over first 50% of training
        return initial_coef * (1.0 - progress) + final_coef * progress
    
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
        # DON'T call env.reset() every epoch! Episodes run continuously with auto-reset.
        # Only reset on first epoch when self.current_obs is None
        if self.current_obs is None:
            obs, _ = self.env.reset()
            self.policy.reset(self.config.num_envs)
            # Reset accumulators at training start
            self.episode_steps.fill(0)
            self.episode_mm_total.fill(0)
            self.episode_inv_total.fill(0)
        else:
            obs = self.current_obs
        
        # Keep completed episodes across epochs (persist episode metrics)
        # Limit to last 1 episode to keep only the most recent
        if len(self.completed_episodes) > 1:
            self.completed_episodes = self.completed_episodes[-1:]
        
        # Track episode count at start of epoch to show only new episodes in logging
        self._episode_count_at_epoch_start = len(self.completed_episodes)
        
        # Only clear buffer infos for epoch-level metrics
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
            # Track which timesteps had inventory decisions
            self.buffer.inv_decision_mask[step] = info['updated_inventory'].astype(np.float32)
            
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
            
            # Accumulate this step's reward FIRST (before handling episode ends)
            # IMPORTANT: Rewards on done=True are from the ENDING episode, not the new one!
            # The C++ env calculates rewards BEFORE reset, so they belong to the current episode.
            self.episode_mm_total += mm_reward
            self.episode_inv_total += inv_reward
            self.episode_steps += 1
            
            # Handle episode ends AFTER accumulating rewards
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
                    
                    # Reset per-env tracking for the NEW episode
                    self.episode_steps[env_id] = 0
                    self.episode_mm_total[env_id] = 0
                    self.episode_inv_total[env_id] = 0
                    self.policy.reset_env(env_id)
            
            obs = next_obs
            self.global_step += self.config.num_envs
        
        # Store current observation for next epoch
        self.current_obs = obs
        
        # Get last values for GAE
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, log_probs, values, _ = self.policy.forward(obs_tensor)
            last_values_mm = values['mm'].cpu().numpy().squeeze()
            last_values_inv = values['inventory'].cpu().numpy().squeeze()
        
        # Compute GAE
        self.buffer.compute_gae(
            last_values_mm, last_values_inv,
            self.config.gamma, self.config.gae_lambda,
            inventory_update_freq=self.config.inventory_update_freq)
        
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
        
        # Decision mask for inventory agent: only train on timesteps where decisions were made
        inv_decision_mask = self.buffer.inv_decision_mask.reshape(n_samples)
        
        # Normalize advantages (optional - can disable if max >> mean suggests over-normalization)
        if self.config.normalize_advantages:
            advantages_mm = (advantages_mm - advantages_mm.mean()) / (advantages_mm.std() + 1e-8)
            # Only normalize inventory advantages over decision timesteps
            inv_decision_advantages = advantages_inv[inv_decision_mask > 0.5]
            if len(inv_decision_advantages) > 0:
                inv_mean = inv_decision_advantages.mean()
                inv_std = inv_decision_advantages.std() + 1e-8
                advantages_inv = (advantages_inv - inv_mean) / inv_std
        
        # Convert to tensors
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
        # MM actions: [bid_spread, ask_spread] = indices [0, 1]
        mm_actions = actions_flat[:, 0:2]
        actions_t = torch.as_tensor(mm_actions, dtype=torch.float32, device=device)
        inv_actions_t = torch.as_tensor(inv_actions_flat, dtype=torch.float32, device=device)
        old_log_probs_mm_t = torch.as_tensor(old_log_probs_mm, dtype=torch.float32, device=device)
        old_log_probs_inv_t = torch.as_tensor(old_log_probs_inv, dtype=torch.float32, device=device)
        advantages_mm_t = torch.as_tensor(advantages_mm, dtype=torch.float32, device=device)
        advantages_inv_t = torch.as_tensor(advantages_inv, dtype=torch.float32, device=device)
        returns_mm_t = torch.as_tensor(returns_mm, dtype=torch.float32, device=device)
        returns_inv_t = torch.as_tensor(returns_inv, dtype=torch.float32, device=device)
        inv_decision_mask_t = torch.as_tensor(inv_decision_mask, dtype=torch.float32, device=device)
        
        losses = {'policy_loss_mm': 0, 'value_loss_mm': 0, 'entropy_mm': 0,
                  'policy_loss_inv': 0, 'value_loss_inv': 0, 'entropy_inv': 0,
                  'advantage_mm_mean': 0, 'advantage_mm_std': 0, 'advantage_mm_max': 0,
                  'advantage_inv_mean': 0, 'advantage_inv_std': 0, 'advantage_inv_max': 0,
                  'grad_norm_mm': 0, 'grad_norm_inv': 0}
        n_updates = 0
        
        # Track advantage statistics (before normalization)
        advantages_mm_raw = advantages_mm.copy()
        advantages_inv_raw = advantages_inv.copy()
        losses['advantage_mm_mean'] = float(np.mean(advantages_mm_raw))
        losses['advantage_mm_std'] = float(np.std(advantages_mm_raw))
        losses['advantage_mm_max'] = float(np.max(np.abs(advantages_mm_raw)))
        losses['advantage_inv_mean'] = float(np.mean(advantages_inv_raw))
        losses['advantage_inv_std'] = float(np.std(advantages_inv_raw))
        losses['advantage_inv_max'] = float(np.max(np.abs(advantages_inv_raw)))
        
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
                inv_decision_mask_batch = inv_decision_mask_t[batch_indices]
                
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
                
                # Value loss with clipping and L2 regularization
                value_loss_mm = nn.functional.mse_loss(value_mm, returns_mm_batch)
                value_loss_mm = torch.clamp(value_loss_mm, max=self.config.value_loss_clip)
                
                # L2 regularization on value function parameters
                value_l2_reg_mm = 0.0
                for param in self.policy.mm_agent.critic.parameters():
                    value_l2_reg_mm += torch.sum(param ** 2)
                value_l2_reg_mm = self.config.value_l2_reg * value_l2_reg_mm
                
                # Get current entropy coefficient (with annealing)
                current_entropy_coef_mm = self._get_entropy_coef_mm()
                    
                loss_mm = (policy_loss_mm 
                          + self.config.value_coef * value_loss_mm 
                          + value_l2_reg_mm
                          - current_entropy_coef_mm * entropy_mm)
                
                self.optimizer_mm.zero_grad()
                loss_mm.backward()
                grad_norm_mm = nn.utils.clip_grad_norm_(self.policy.mm_agent.parameters(), 
                                                       self.config.max_grad_norm)
                self.optimizer_mm.step()
                
                # Track gradient norm
                losses['grad_norm_mm'] += grad_norm_mm.item()
                
                # === Update Inventory Agent ===
                # Only update on timesteps where decisions were actually made
                log_prob_inv, entropy_inv, value_inv = eval_results['inventory']
                log_prob_inv = log_prob_inv.squeeze()
                entropy_inv = entropy_inv.mean()
                value_inv = value_inv.squeeze()
                
                # Mask: only compute loss for decision timesteps
                decision_mask = inv_decision_mask_batch > 0.5
                n_decisions = decision_mask.sum().item()
                
                if n_decisions > 0:
                    # Compute losses only on decision timesteps
                    ratio_inv = torch.exp(log_prob_inv - old_log_probs_inv_batch)
                    surr1_inv = ratio_inv * advantages_inv_batch
                    surr2_inv = torch.clamp(ratio_inv, 1 - self.config.clip_range,
                                            1 + self.config.clip_range) * advantages_inv_batch
                    policy_loss_inv = -torch.min(surr1_inv[decision_mask], surr2_inv[decision_mask]).mean()
                    
                    # Value loss with clipping and L2 regularization (only on decision timesteps)
                    value_loss_inv = nn.functional.mse_loss(
                        value_inv[decision_mask], 
                        returns_inv_batch[decision_mask]
                    )
                    value_loss_inv = torch.clamp(value_loss_inv, max=self.config.value_loss_clip)
                    
                    # L2 regularization on value function parameters
                    value_l2_reg_inv = 0.0
                    for param in self.policy.inventory_agent.critic.parameters():
                        value_l2_reg_inv += torch.sum(param ** 2)
                    value_l2_reg_inv = self.config.value_l2_reg * value_l2_reg_inv
                    
                    # Get current entropy coefficient (with annealing)
                    current_entropy_coef_inv = self._get_entropy_coef_inv()
                    
                    loss_inv = (policy_loss_inv
                               + self.config.value_coef * value_loss_inv
                               + value_l2_reg_inv
                               - current_entropy_coef_inv * entropy_inv)
                else:
                    # No decisions in this batch, skip update
                    policy_loss_inv = torch.tensor(0.0, device=device)
                    value_loss_inv = torch.tensor(0.0, device=device)
                    loss_inv = torch.tensor(0.0, device=device)
                
                self.optimizer_inv.zero_grad()
                loss_inv.backward()
                grad_norm_inv = nn.utils.clip_grad_norm_(self.policy.inventory_agent.parameters(),
                                                         self.config.max_grad_norm)
                self.optimizer_inv.step()
                
                # Track gradient norm
                losses['grad_norm_inv'] += grad_norm_inv.item()
                
                # Track losses
                losses['policy_loss_mm'] += policy_loss_mm.item()
                losses['value_loss_mm'] += value_loss_mm.item()
                losses['entropy_mm'] += entropy_mm.item()
                losses['policy_loss_inv'] += policy_loss_inv.item()
                losses['value_loss_inv'] += value_loss_inv.item()
                losses['entropy_inv'] += entropy_inv.item()
                n_updates += 1
        
        # Average losses (except advantage stats which are already computed once)
        if n_updates > 0:
            for key in losses:
                if key not in ['advantage_mm_mean', 'advantage_mm_std', 'advantage_mm_max',
                              'advantage_inv_mean', 'advantage_inv_std', 'advantage_inv_max']:
                    losses[key] /= n_updates
        
        return losses
    
    def _log_epoch(self, epoch: int, losses: Dict[str, float]):
        """Log epoch statistics."""
        # Always compute epoch-level stats from the buffer (this epoch's data)
        # Sum rewards across all steps and envs for this epoch
        avg_mm_reward = self.buffer.mm_rewards.sum() / self.config.num_envs
        avg_inv_reward = self.buffer.inv_rewards.sum() / self.config.num_envs
        
        # Get P&L from last info in buffer (end of epoch snapshot - RUNNING TOTALS, not episode totals)
        # NOTE: These are running totals from the current episode, not final episode values
        # For final episode values, see completed_episodes which use final_* keys
        avg_realized_pnl = 0.0
        avg_unrealized_pnl = 0.0
        total_trades = 0
        if self.buffer.infos and len(self.buffer.infos) > 0:
            last_info = self.buffer.infos[-1]
            if 'realized_pnl' in last_info:
                val = last_info['realized_pnl']
                avg_realized_pnl = float(val.mean()) if isinstance(val, np.ndarray) else float(val)
            if 'lifo_unrealized_pnl' in last_info:
                val = last_info['lifo_unrealized_pnl']
                avg_unrealized_pnl = float(val.mean()) if isinstance(val, np.ndarray) else float(val)
            if 'trade_count' in last_info:
                val = last_info['trade_count']
                total_trades = int(val.sum()) if isinstance(val, np.ndarray) else int(val)
        
        # Compute action statistics (3-action space: bid_spread, ask_spread, target)
        actions = self.buffer.actions.reshape(-1, 3)
        avg_bid_spread = actions[:, 0].mean()
        avg_ask_spread = actions[:, 1].mean()
        avg_target = actions[:, 2].mean()
        
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
        # NOTE: R.PnL and U.PnL are RUNNING TOTALS (current episode state), not final episode values
        print(f"Epoch {epoch:5d} | "
              f"Step {self.global_step:8d} | "
              f"MM.Rew {avg_mm_reward:8.2f} | "
              f"Inv.Rew {avg_inv_reward:8.2f} | "
              f"R.PnL ${avg_realized_pnl:7.2f} (running) | "
              f"U.PnL ${avg_unrealized_pnl:7.2f} (running) | "
              f"Trades {total_trades:4d} (B:{buy_trades}/S:{sell_trades})")
        print(f"         Loss: PL_mm {losses['policy_loss_mm']:.4f} VL_mm {losses['value_loss_mm']:.4f} | "
              f"PL_inv {losses['policy_loss_inv']:.4f} VL_inv {losses['value_loss_inv']:.4f} | "
              f"Ent {losses['entropy_mm']:.3f}/{losses['entropy_inv']:.3f} | "
              f"Std {mm_spread_std:.3f}/{inv_std:.3f}")
        print(f"         Adv: MM μ={losses['advantage_mm_mean']:.4f} σ={losses['advantage_mm_std']:.4f} max={losses['advantage_mm_max']:.4f} | "
              f"Inv μ={losses['advantage_inv_mean']:.4f} σ={losses['advantage_inv_std']:.4f} max={losses['advantage_inv_max']:.4f}")
        print(f"         Grad: MM={losses['grad_norm_mm']:.4f} Inv={losses['grad_norm_inv']:.4f}")
        
        # Print completed episodes (only show episodes from this epoch to avoid spam)
        episodes_this_epoch = self.completed_episodes[self._episode_count_at_epoch_start:]
        if episodes_this_epoch:
            print(f"\n  Completed {len(episodes_this_epoch)} episode(s) this epoch (total: {len(self.completed_episodes)}):")
            for ep in episodes_this_epoch:
                net_pnl = ep.realized_pnl + ep.unrealized_pnl + ep.fees
                # Note: R.PnL includes fees (balance change), so net_pnl double-counts fees
                # True Net = R.PnL + U.PnL (fees already in R.PnL)
                # NOTE: These are FINAL episode values (from final_* keys), not running totals
                true_net = ep.realized_pnl + ep.unrealized_pnl
                print(f"  [Episode FINAL] Env {ep.env_id} | "
                      f"Steps {ep.steps:5d} | "
                      f"MM.Rew {ep.mm_reward:7.2f} | "
                      f"Inv.Rew {ep.inv_reward:7.2f} | "
                      f"R.PnL ${ep.realized_pnl:7.2f} (final) | "
                      f"SprdCap ${ep.spread_capture:6.2f} (final) | "
                      f"U.PnL ${ep.unrealized_pnl:7.2f} (final) | "
                      f"Net ${true_net:7.2f} (final) | "
                      f"Trades {ep.trade_count:4d} (final) | "
                      f"Pos {ep.net_amount_btc:+.5f} BTC (final)")
            print()
        
        # === TensorBoard Logging ===
        step = self.global_step
        
        # Rewards
        self.tb_writer.add_scalar('Reward/MM_Agent', avg_mm_reward, step)
        self.tb_writer.add_scalar('Reward/Inventory_Agent', avg_inv_reward, step)
        self.tb_writer.add_scalar('Reward/Total', avg_mm_reward + avg_inv_reward, step)
        
        # Advantage statistics
        self.tb_writer.add_scalar('Advantage/MM_Mean', losses['advantage_mm_mean'], step)
        self.tb_writer.add_scalar('Advantage/MM_Std', losses['advantage_mm_std'], step)
        self.tb_writer.add_scalar('Advantage/MM_Max', losses['advantage_mm_max'], step)
        self.tb_writer.add_scalar('Advantage/Inv_Mean', losses['advantage_inv_mean'], step)
        self.tb_writer.add_scalar('Advantage/Inv_Std', losses['advantage_inv_std'], step)
        self.tb_writer.add_scalar('Advantage/Inv_Max', losses['advantage_inv_max'], step)
        
        # Gradient norms
        self.tb_writer.add_scalar('Gradient/MM_Norm', losses['grad_norm_mm'], step)
        self.tb_writer.add_scalar('Gradient/Inv_Norm', losses['grad_norm_inv'], step)
        
        # P&L Metrics
        self.tb_writer.add_scalar('PnL/Realized', avg_realized_pnl, step)
        self.tb_writer.add_scalar('PnL/Unrealized', avg_unrealized_pnl, step)
        
        # Losses - MM Agent
        self.tb_writer.add_scalar('Loss/MM_Policy', losses['policy_loss_mm'], step)
        self.tb_writer.add_scalar('Loss/MM_Value', losses['value_loss_mm'], step)
        self.tb_writer.add_scalar('Loss/MM_Entropy', losses['entropy_mm'], step)
        
        # Losses - Inventory Agent
        self.tb_writer.add_scalar('Loss/Inv_Policy', losses['policy_loss_inv'], step)
        self.tb_writer.add_scalar('Loss/Inv_Value', losses['value_loss_inv'], step)
        self.tb_writer.add_scalar('Loss/Inv_Entropy', losses['entropy_inv'], step)
        
        # Action Statistics
        self.tb_writer.add_scalar('Actions/MM_Spread_Std', mm_spread_std, step)
        self.tb_writer.add_scalar('Actions/Inv_Target_Std', inv_std, step)
        self.tb_writer.add_scalar('Actions/Avg_Bid_Spread', avg_bid_spread, step)
        self.tb_writer.add_scalar('Actions/Avg_Ask_Spread', avg_ask_spread, step)
        self.tb_writer.add_scalar('Actions/Avg_Target_Inventory', avg_target, step)
        
        # Trading Statistics
        self.tb_writer.add_scalar('Trading/Total_Trades', total_trades, step)
        self.tb_writer.add_scalar('Trading/Buy_Trades', buy_trades, step)
        self.tb_writer.add_scalar('Trading/Sell_Trades', sell_trades, step)
        if total_trades > 0:
            self.tb_writer.add_scalar('Trading/Buy_Sell_Ratio', 
                                      buy_trades / max(sell_trades, 1), step)
        
        # Episode Statistics (if episodes completed)
        if self.completed_episodes:
            avg_episode_mm_rew = np.mean([ep.mm_reward for ep in self.completed_episodes])
            avg_episode_inv_rew = np.mean([ep.inv_reward for ep in self.completed_episodes])
            avg_spread_capture = np.mean([ep.spread_capture for ep in self.completed_episodes])
            avg_unrealized = np.mean([ep.unrealized_pnl for ep in self.completed_episodes])
            avg_fees = np.mean([ep.fees for ep in self.completed_episodes])
            avg_trades = np.mean([ep.trade_count for ep in self.completed_episodes])
            avg_net = np.mean([ep.realized_pnl + ep.unrealized_pnl + ep.fees 
                              for ep in self.completed_episodes])
            
            self.tb_writer.add_scalar('Episode/MM_Reward', avg_episode_mm_rew, step)
            self.tb_writer.add_scalar('Episode/Inv_Reward', avg_episode_inv_rew, step)
            self.tb_writer.add_scalar('Episode/Spread_Capture', avg_spread_capture, step)
            self.tb_writer.add_scalar('Episode/Unrealized_PnL', avg_unrealized, step)
            self.tb_writer.add_scalar('Episode/Fees', avg_fees, step)
            self.tb_writer.add_scalar('Episode/Net_PnL', avg_net, step)
            self.tb_writer.add_scalar('Episode/Trades', avg_trades, step)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("Hierarchical PPO Training - Two-Agent Market Making")
        print("="*80)
        print(f"Inventory Agent: updates every {self.config.inventory_update_freq} steps (50 sec)")
        print(f"MM Agent: updates every step (500ms)")
        print(f"Steps per epoch: {self.config.n_steps}")
        print(f"Total epochs: {self.config.total_epochs}")
        print(f"Observations: 40 signals (13 market + 4 AMM + 8 trade + 11 agent state + 1 previous spread + 2 bid/ask distances + 1 mid_change)")
        print(f"Actions: 3 (bid_spread, ask_spread, target_inventory)")
        print(f"Smart requote: only when prices change by >2 ticks")
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
            
            # Always save latest checkpoint for resuming
            self.save_checkpoint(str(self.results_dir / "latest.pt"))
            
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
        
        # Close TensorBoard writer
        self.tb_writer.close()
        print("TensorBoard logs saved.")
    
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
        # weights_only=False needed for PyTorch 2.6+ (checkpoint contains config objects)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
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
    
    # Check for existing checkpoint (prefer best_model, then latest)
    checkpoint_path = None
    best_model_path = Path("results/hierarchical/best_model.pt")
    latest_path = Path("results/hierarchical/latest.pt")
    
    if best_model_path.exists():
        checkpoint_path = best_model_path
        print(f"Found best model checkpoint at {checkpoint_path}")
    elif latest_path.exists():
        checkpoint_path = latest_path
        print(f"Found latest checkpoint at {checkpoint_path}")
    
    if checkpoint_path:
        trainer.load_checkpoint(str(checkpoint_path))
    else:
        print("No checkpoint found, starting from scratch")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
