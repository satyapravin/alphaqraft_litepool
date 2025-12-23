# Copyright 2024 Alphaqraft
#
# Hierarchical PPO Training for Two-Agent Market Making
#
# Architecture:
# - Inventory Agent (slow, strategic): learns WHAT position to hold
#   - Updates every 100 steps (10 seconds)
#   - Reward: unrealized P&L delta (market direction)
#   - Observations: AMM flow, volatility, position state
#
# - MM Agent (fast, tactical): learns HOW to execute toward target
#   - Updates every step (100ms)
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
    inventory_update_freq: int = 100  # Every 100 steps = 10 seconds
    
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
        self.logger = MetricLogger(log_dir="logs/hierarchical")
        self.episode_rewards = deque(maxlen=100)
        self.episode_mm_rewards = deque(maxlen=100)
        self.episode_inv_rewards = deque(maxlen=100)
        
        # Rollout buffer
        obs_dim = 32
        action_dim = 4
        self.buffer = RolloutBuffer.create(
            config.n_steps, config.num_envs, obs_dim, action_dim
        )
        
        # Tracking
        self.global_step = 0
        self.epochs_completed = 0
    
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
    
    def collect_rollout(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect one rollout of experience."""
        obs, _ = self.env.reset()
        self.policy.reset(self.config.num_envs)
        
        # Tracking for logging
        cum_mm_rewards = np.zeros(self.config.num_envs)
        cum_inv_rewards = np.zeros(self.config.num_envs)
        
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
            
            # Extract separate rewards
            mm_reward = env_info.get('mm_reward', np.zeros(self.config.num_envs))
            inv_reward = env_info.get('inv_reward', np.zeros(self.config.num_envs))
            
            self.buffer.mm_rewards[step] = mm_reward
            self.buffer.inv_rewards[step] = inv_reward
            self.buffer.dones[step] = done
            
            cum_mm_rewards += mm_reward
            cum_inv_rewards += inv_reward
            
            # Handle episode ends
            for env_id in range(self.config.num_envs):
                if done[env_id]:
                    self.episode_mm_rewards.append(cum_mm_rewards[env_id])
                    self.episode_inv_rewards.append(cum_inv_rewards[env_id])
                    self.episode_rewards.append(reward[env_id])
                    cum_mm_rewards[env_id] = 0
                    cum_inv_rewards[env_id] = 0
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
        actions_t = torch.as_tensor(actions_flat[:, :3], dtype=torch.float32, device=device)  # MM actions only
        inv_actions_t = torch.as_tensor(inv_actions_flat, dtype=torch.float32, device=device)
        old_log_probs_mm_t = torch.as_tensor(old_log_probs_mm, dtype=torch.float32, device=device)
        old_log_probs_inv_t = torch.as_tensor(old_log_probs_inv, dtype=torch.float32, device=device)
        advantages_mm_t = torch.as_tensor(advantages_mm, dtype=torch.float32, device=device)
        advantages_inv_t = torch.as_tensor(advantages_inv, dtype=torch.float32, device=device)
        returns_mm_t = torch.as_tensor(returns_mm, dtype=torch.float32, device=device)
        returns_inv_t = torch.as_tensor(returns_inv, dtype=torch.float32, device=device)
        
        losses = {'policy_loss_mm': 0, 'value_loss_mm': 0, 'entropy_mm': 0,
                  'policy_loss_inv': 0, 'value_loss_inv': 0, 'entropy_inv': 0}
        
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
        
        # Average losses
        num_updates = self.config.update_epochs * (n_samples // self.config.minibatch_size)
        for key in losses:
            losses[key] /= num_updates
        
        return losses
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Hierarchical PPO Training")
        print("="*60)
        print(f"Inventory Agent: updates every {self.config.inventory_update_freq} steps")
        print(f"MM Agent: updates every step")
        print(f"Total epochs: {self.config.total_epochs}")
        print("="*60 + "\n")
        
        for epoch in range(self.config.total_epochs):
            epoch_start = time.time()
            
            # Collect rollout
            self.collect_rollout()
            
            # Update policy
            losses = self.update()
            
            self.epochs_completed += 1
            
            # Logging
            if epoch % self.config.log_interval == 0:
                avg_mm_reward = np.mean(self.episode_mm_rewards) if self.episode_mm_rewards else 0
                avg_inv_reward = np.mean(self.episode_inv_rewards) if self.episode_inv_rewards else 0
                
                print(f"Epoch {epoch:5d} | "
                      f"Step {self.global_step:8d} | "
                      f"MM Rew {avg_mm_reward:7.2f} | "
                      f"Inv Rew {avg_inv_reward:7.2f} | "
                      f"PL_mm {losses['policy_loss_mm']:.4f} | "
                      f"PL_inv {losses['policy_loss_inv']:.4f} | "
                      f"Ent_mm {losses['entropy_mm']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 and epoch > 0:
                self.save_checkpoint(f"checkpoints/hierarchical_epoch_{epoch}.pt")
            
            # GC
            if epoch % 100 == 0:
                gc.collect()
        
        print("\nTraining complete!")
        self.save_checkpoint("checkpoints/hierarchical_final.pt")
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'policy': {
                'inventory_agent': self.policy.inventory_agent.state_dict(),
                'mm_agent': self.policy.mm_agent.state_dict(),
            },
            'optimizer_inv': self.optimizer_inv.state_dict(),
            'optimizer_mm': self.optimizer_mm.state_dict(),
            'global_step': self.global_step,
            'epochs_completed': self.epochs_completed,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.policy.inventory_agent.load_state_dict(checkpoint['policy']['inventory_agent'])
        self.policy.mm_agent.load_state_dict(checkpoint['policy']['mm_agent'])
        self.optimizer_inv.load_state_dict(checkpoint['optimizer_inv'])
        self.optimizer_mm.load_state_dict(checkpoint['optimizer_mm'])
        self.global_step = checkpoint['global_step']
        self.epochs_completed = checkpoint['epochs_completed']
        print(f"Loaded checkpoint from {path}")


def main():
    """Main entry point."""
    # Create trainer
    trainer = HierarchicalPPOTrainer(config)
    
    # Check for existing checkpoint
    checkpoint_path = Path("checkpoints/hierarchical_latest.pt")
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}")
        trainer.load_checkpoint(str(checkpoint_path))
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

