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
import torch.nn.functional as F
import torch.optim as optim
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


def robust_normalize(advantages):
    """
    Use median and IQR instead of mean/std.
    Much more robust to outliers.
    """
    median = torch.median(advantages)
    q75 = torch.quantile(advantages, 0.75)
    q25 = torch.quantile(advantages, 0.25)
    iqr = q75 - q25
    
    if iqr < 1e-8:
        return advantages - median
    
    # Normalize by IQR (1.349 converts IQR to std for normal dist)
    normalized = (advantages - median) / (iqr / 1.349)
    
    # Soft clip extreme values
    return torch.tanh(normalized / 5.0) * 5.0
import time
from datetime import datetime

import litepool
from torch.utils.tensorboard import SummaryWriter
from hierarchical_policy import HierarchicalPolicy, create_hierarchical_policy
from metric_logger import MetricLogger
from hierarchical_config import HierarchicalConfig

# === Device setup ===
device = torch.device("cpu")
print(f"Using device: {device}")


config = HierarchicalConfig()


class ValueNormalizer:
    """Normalizes values to help value function learn with high standard deviations."""
    def __init__(self, shape, clip_range=10.0):
        self.mean = torch.zeros(shape)
        self.std = torch.ones(shape)
        self.count = 0
        self.clip_range = clip_range
        
    def normalize(self, values):
        """Normalize values using running statistics."""
        # Normalize
        normalized = (values - self.mean.to(values.device)) / (self.std.to(values.device) + 1e-8)
        # Clip
        return torch.clamp(normalized, -self.clip_range, self.clip_range)
    
    def update(self, batch_values):
        """Update running statistics with new batch."""
        # Update running statistics
        batch_mean = batch_values.mean().item()
        batch_std = batch_values.std().item()
        
        self.mean = 0.99 * self.mean + 0.01 * batch_mean
        self.std = 0.99 * self.std + 0.01 * batch_std
        self.count += 1



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
            inv_actions=np.zeros((n_steps, num_envs, 2), dtype=np.float32),  # target_inventory, risk_aversion
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
                    gamma: float, gae_lambda: float, inventory_update_freq: int,
                    clip_delta: float = 2.0, normalize_gae: bool = True):
        """
        Compute GAE for both agents.
        
        For inventory agent: Use effective gamma^update_freq to account for temporal mismatch.
        Decisions are made every update_freq steps, so rewards accumulate over that period.
        
        Args:
            clip_delta: Clip TD errors (delta) to prevent extreme values
            normalize_gae: Normalize advantages after computation
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
            # Clip TD error to prevent extreme values
            delta = np.clip(delta, -clip_delta, clip_delta)
            self.advantages_mm[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns_mm = self.advantages_mm + self.values_mm
        
        # Normalize MM advantages if enabled
        if normalize_gae:
            mm_advantages_flat = self.advantages_mm.flatten()
            if len(mm_advantages_flat) > 0:
                # Convert to torch tensor for robust_normalize
                mm_advantages_tensor = torch.from_numpy(mm_advantages_flat).float()
                mm_normalized = robust_normalize(mm_advantages_tensor)
                # Reshape back to original shape
                self.advantages_mm = mm_normalized.numpy().reshape(self.advantages_mm.shape)
        
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
                # Clip TD error to prevent extreme values
                delta = np.clip(delta, -clip_delta, clip_delta)
                last_gae = delta + effective_gamma * effective_gae_lambda * next_non_terminal * last_gae
                
                # Assign GAE to this decision timestep
                self.advantages_inv[t, env_id] = last_gae
                
                # Propagate GAE to intermediate steps (if any) with discounting
                if i < len(decision_timesteps) - 1:
                    num_intermediate = next_t - t - 1
                    if num_intermediate > 0:
                        discount_factors = np.power(gamma, np.arange(1, num_intermediate + 1))
                        self.advantages_inv[t+1:next_t, env_id] = last_gae * discount_factors[::-1]
        
        self.returns_inv = self.advantages_inv + self.values_inv

        # Normalize inventory advantages if enabled
        if normalize_gae:
            inv_advantages_flat = self.advantages_inv.flatten()
            if len(inv_advantages_flat) > 0:
                inv_advantages_tensor = torch.from_numpy(inv_advantages_flat).float()
                inv_normalized = robust_normalize(inv_advantages_tensor)
                self.advantages_inv = inv_normalized.numpy().reshape(self.advantages_inv.shape)

class HierarchicalPPOTrainer:
    """Hierarchical PPO Trainer for Two-Agent Market Making."""
    
    def __init__(self, config: HierarchicalConfig):
        self.config = config
        self.device = device
        
        # Results directory
        self.results_dir = Path("results/hierarchical")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.tb_writer = SummaryWriter(str(self.results_dir / "runs"))
        
        # Policy
        self.policy = create_hierarchical_policy(
            obs_dim=40,
            inventory_update_freq=config.inventory_update_freq,
            device=str(device),
            target_range=config.target_range,
        )
        
        # Define separate optimizers for shared and specific parameters
        shared_params = set(self.policy.shared_encoder.parameters())
        inv_specific_params = [p for p in self.policy.inventory_agent.parameters() if p not in shared_params]
        mm_specific_params = [p for p in self.policy.mm_agent.parameters() if p not in shared_params]

        self.optimizer_shared = optim.Adam(self.policy.shared_encoder.parameters(), lr=config.learning_rate)
        self.optimizer_inv = optim.Adam(inv_specific_params, lr=config.inv_learning_rate)
        self.optimizer_mm = optim.Adam(mm_specific_params, lr=config.learning_rate)
        
        # Environment
        self.env = litepool.make(
            "RlTrader-v0",
            env_type="gymnasium",
            num_envs=config.num_envs,
            batch_size=config.num_envs,
            num_threads=config.num_threads,
            seed=42,
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
            balance=config.balance,
            start=36000,
            max_episode_steps=config.max_episode_steps,
            base_spread_bps=config.base_spread_bps,
            min_size_pct=config.min_size_pct,
            max_size_pct=config.max_size_pct,
        )
        self.env.spec.id = "RlTrader-v0"
        
        # Buffer
        self.buffer = RolloutBuffer.create(
            n_steps=config.n_steps,
            num_envs=config.num_envs,
            obs_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
        )
        
        # Tracking
        self.global_step = 0
        self.epochs_completed = 0
        self.best_reward = float('-inf')
        self.episode_rewards = []
        self.completed_episodes = []
        
        # Value normalizers
        self.value_norm_mm = ValueNormalizer((1,))
        self.value_norm_inv = ValueNormalizer((1,))
    
    def collect_rollout(self):
        """Collect rollout data."""
        # Reset policy state
        self.policy.reset(self.config.num_envs)
        
        # Get initial observations (gymnasium returns (obs, info) tuple)
        obs, _ = self.env.reset()
        self.buffer.observations[0] = obs
        
        # Track running statistics
        episode_infos = [{} for _ in range(self.config.num_envs)]
        done_mask = np.zeros(self.config.num_envs, dtype=bool)
        
        for step in range(self.config.n_steps):
            self.global_step += self.config.num_envs
            
            # Get actions
            with torch.no_grad():
                action, info = self.policy.get_action(obs)
            
            # Step environment
            next_obs, _, done, truncated, env_info = self.env.step(action)
            
            # Store data
            self.buffer.actions[step] = action
            # Concatenate target_inventory and risk_aversion into [num_envs, 2]
            targets = info['targets']  # [num_envs, 1]
            risk_aversion = info['risk_aversion']  # [num_envs, 1]
            self.buffer.inv_actions[step] = np.concatenate([targets, risk_aversion], axis=-1)  # [num_envs, 2]
            self.buffer.log_probs_mm[step] = info['log_prob_mm'].squeeze(-1)
            self.buffer.log_probs_inv[step] = info['log_prob_inv'].squeeze(-1)
            self.buffer.values_mm[step] = info['value_mm'].squeeze(-1)
            self.buffer.values_inv[step] = info['value_inv'].squeeze(-1)
            self.buffer.inv_decision_mask[step] = info['updated_inventory'].astype(np.float32)
            self.buffer.dones[step] = (done | truncated).astype(np.float32)
            
            # Extract rewards from env_info
            # env_info is a dict where each key maps to array of shape (num_envs,)
            def _extract_info_value(env_info: dict, key: str, default=0.0):
                """Safely extract a value from env_info, returning array of shape (num_envs,)."""
                val = env_info.get(key, default)
                if isinstance(val, np.ndarray):
                    return val.flatten()
                else:
                    return np.array([val] * self.config.num_envs)
            
            mm_reward = _extract_info_value(env_info, 'mm_reward', 0.0)
            inv_reward = _extract_info_value(env_info, 'inv_reward', 0.0)
            realized_pnl = _extract_info_value(env_info, 'realized_pnl', 0.0)
            unrealized_pnl = _extract_info_value(env_info, 'unrealized_pnl', 0.0)
            spread_capture = _extract_info_value(env_info, 'spread_capture', 0.0)
            fees = _extract_info_value(env_info, 'fees', 0.0)
            trade_count = _extract_info_value(env_info, 'trade_count', 0)
            net_amount_btc = _extract_info_value(env_info, 'net_amount_btc', 0.0)
            
            for env_id in range(self.config.num_envs):
                self.buffer.mm_rewards[step, env_id] = mm_reward[env_id]
                self.buffer.inv_rewards[step, env_id] = inv_reward[env_id]
                
                # Accumulate episode info
                episode_infos[env_id]['mm_reward'] = episode_infos[env_id].get('mm_reward', 0.0) + mm_reward[env_id]
                episode_infos[env_id]['inv_reward'] = episode_infos[env_id].get('inv_reward', 0.0) + inv_reward[env_id]
                episode_infos[env_id]['realized_pnl'] = episode_infos[env_id].get('realized_pnl', 0.0) + realized_pnl[env_id]
                episode_infos[env_id]['unrealized_pnl'] = unrealized_pnl[env_id]
                episode_infos[env_id]['spread_capture'] = episode_infos[env_id].get('spread_capture', 0.0) + spread_capture[env_id]
                episode_infos[env_id]['fees'] = episode_infos[env_id].get('fees', 0.0) + fees[env_id]
                episode_infos[env_id]['trade_count'] = episode_infos[env_id].get('trade_count', 0) + int(trade_count[env_id])
                episode_infos[env_id]['net_amount_btc'] = net_amount_btc[env_id]
            
            # Handle done environments
            new_done = (done | truncated) & ~done_mask
            for env_id in np.where(new_done)[0]:
                # Use final episode metrics from terminal info if available (more accurate)
                # Extract arrays and then index by env_id
                final_realized_pnl_arr = _extract_info_value(env_info, 'final_realized_pnl', 0.0)
                final_unrealized_pnl_arr = _extract_info_value(env_info, 'final_unrealized_pnl', 0.0)
                final_spread_capture_arr = _extract_info_value(env_info, 'final_spread_capture', 0.0)
                final_fees_arr = _extract_info_value(env_info, 'final_fees', 0.0)
                final_trade_count_arr = _extract_info_value(env_info, 'final_trade_count', 0)
                final_net_amount_btc_arr = _extract_info_value(env_info, 'final_net_amount_btc', 0.0)
                
                final_realized_pnl = float(final_realized_pnl_arr[env_id])
                final_unrealized_pnl = float(final_unrealized_pnl_arr[env_id])
                final_spread_capture = float(final_spread_capture_arr[env_id])
                final_fees = float(final_fees_arr[env_id])
                final_trade_count = int(final_trade_count_arr[env_id])
                final_net_amount_btc = float(final_net_amount_btc_arr[env_id])
                
                # Create final episode summary (use final values if available, otherwise use accumulated)
                episode_summary = episode_infos[env_id].copy()
                if final_realized_pnl != 0.0 or final_unrealized_pnl != 0.0:
                    # Use final values from terminal info (more accurate)
                    episode_summary['realized_pnl'] = final_realized_pnl
                    episode_summary['unrealized_pnl'] = final_unrealized_pnl
                    episode_summary['spread_capture'] = final_spread_capture
                    episode_summary['fees'] = final_fees
                    episode_summary['trade_count'] = final_trade_count
                    episode_summary['net_amount_btc'] = final_net_amount_btc
                
                self.completed_episodes.append(episode_summary)
                self.episode_rewards.append(
                    episode_summary['mm_reward'] + episode_summary['inv_reward']
                )
                # Reset episode info for this env
                episode_infos[env_id] = {}
                # Reset policy state for this env
                self.policy.reset_env(env_id)
            
            done_mask = (done | truncated)
            
            # Store next obs
            obs = next_obs
            if step < self.config.n_steps - 1:
                self.buffer.observations[step + 1] = obs
        
            # Store info for logging
            self.buffer.infos.append(env_info)
        
        # Compute advantages
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(obs).float().to(self.device)
            _, _, last_values, _ = self.policy.forward(last_obs_tensor)
            last_values_mm = last_values['mm'].cpu().numpy().squeeze(-1)
            last_values_inv = last_values['inventory'].cpu().numpy().squeeze(-1)
        
        self.buffer.compute_gae(
            last_values_mm=last_values_mm,
            last_values_inv=last_values_inv,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            inventory_update_freq=self.config.inventory_update_freq,
            clip_delta=self.config.gae_clip_delta,
            normalize_gae=self.config.normalize_gae,
        )
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        # Flatten buffer for easier batching
        flat_obs = self.buffer.observations.reshape((-1, self.buffer.observations.shape[-1]))
        flat_actions = self.buffer.actions.reshape((-1, self.buffer.actions.shape[-1]))
        flat_inv_actions = self.buffer.inv_actions.reshape((-1, self.buffer.inv_actions.shape[-1]))
        flat_log_probs_mm = self.buffer.log_probs_mm.flatten()
        flat_log_probs_inv = self.buffer.log_probs_inv.flatten()
        flat_values_mm = self.buffer.values_mm.flatten()
        flat_values_inv = self.buffer.values_inv.flatten()
        flat_advantages_mm = self.buffer.advantages_mm.flatten()
        flat_advantages_inv = self.buffer.advantages_inv.flatten()
        flat_returns_mm = self.buffer.returns_mm.flatten()
        flat_returns_inv = self.buffer.returns_inv.flatten()
        flat_inv_mask = self.buffer.inv_decision_mask.flatten()
        
        total_steps = flat_obs.shape[0]
        
        losses = {
            'policy_loss_mm': 0.0,
            'value_loss_mm': 0.0,
            'entropy_mm': 0.0,
            'policy_loss_inv': 0.0,
            'value_loss_inv': 0.0,
            'entropy_inv': 0.0,
            'grad_norm_mm': 0.0,
            'grad_norm_inv': 0.0,
            'advantage_mm_mean': np.mean(flat_advantages_mm),
            'advantage_mm_std': np.std(flat_advantages_mm),
            'advantage_mm_max': np.max(flat_advantages_mm),
            'advantage_inv_mean': np.mean(flat_advantages_inv[flat_inv_mask > 0.5]),
            'advantage_inv_std': np.std(flat_advantages_inv[flat_inv_mask > 0.5]),
            'advantage_inv_max': np.max(flat_advantages_inv[flat_inv_mask > 0.5]),
        }
        
        num_updates = 0
        
        for epoch in range(self.config.update_epochs):
            indices = torch.randperm(total_steps, device=self.device)
            
            for start in range(0, total_steps, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                # Minibatch tensors
                obs_tensor = torch.from_numpy(flat_obs[mb_indices.cpu().numpy()]).float().to(self.device)
                # MM agent only needs first 2 actions (bid_spread, ask_spread)
                mm_actions_tensor = torch.from_numpy(flat_actions[mb_indices.cpu().numpy()][:, 0:2]).float().to(self.device)
                inv_actions_tensor = torch.from_numpy(flat_inv_actions[mb_indices.cpu().numpy()]).float().to(self.device)
                old_log_probs_mm = torch.from_numpy(flat_log_probs_mm[mb_indices.cpu().numpy()]).float().to(self.device)
                old_log_probs_inv = torch.from_numpy(flat_log_probs_inv[mb_indices.cpu().numpy()]).float().to(self.device)
                advantages_mm = torch.from_numpy(flat_advantages_mm[mb_indices.cpu().numpy()]).float().to(self.device)
                advantages_inv = torch.from_numpy(flat_advantages_inv[mb_indices.cpu().numpy()]).float().to(self.device)
                returns_mm = torch.from_numpy(flat_returns_mm[mb_indices.cpu().numpy()]).float().to(self.device)
                returns_inv = torch.from_numpy(flat_returns_inv[mb_indices.cpu().numpy()]).float().to(self.device)
                inv_update_mask = torch.from_numpy(flat_inv_mask[mb_indices.cpu().numpy()]).float().to(self.device)
                
                # Evaluate actions
                eval_results = self.policy.evaluate_actions(
                    obs_tensor,
                    mm_actions_tensor,
                    inv_actions_tensor,
                )
                
                # Extract
                mm_log_prob, mm_entropy, mm_value = eval_results['mm']
                inv_log_prob, inv_entropy, inv_value = eval_results['inventory']
                
                # Inventory agent update (only if there are decisions)
                loss_inv = None
                if inv_update_mask.sum() > 0:
                    # Policy loss (masked)
                    ratio_inv = torch.exp(inv_log_prob - old_log_probs_inv.unsqueeze(-1))
                    surrogate1_inv = ratio_inv * advantages_inv.unsqueeze(-1)
                    surrogate2_inv = torch.clamp(ratio_inv, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages_inv.unsqueeze(-1)
                    policy_loss_inv = -torch.min(surrogate1_inv, surrogate2_inv).mean()
                    
                    # Value loss (clipped and masked)
                    value_loss_inv = F.mse_loss(inv_value, returns_inv.unsqueeze(-1), reduction='none')
                    value_loss_inv = (value_loss_inv * inv_update_mask.unsqueeze(-1)).mean()
                    
                    # Entropy
                    entropy_inv = inv_entropy.mean()
                    
                    # Regularization
                    value_l2_reg_inv = torch.mean(inv_value ** 2)
                    # Inventory agent uses Beta distribution with concentration parameters, not log_std
                    # Light regularization on concentration parameters to prevent extreme values
                    concentration_l2_reg_inv = torch.sum(
                        (self.policy.inventory_agent.target_inv_concentration.weight ** 2).sum(dim=1) +
                        (self.policy.inventory_agent.risk_aversion_concentration.weight ** 2).sum(dim=1)
                    )
                    concentration_l2_reg_inv = 1e-4 * concentration_l2_reg_inv  # Light regularization
                    
                    loss_inv = (
                        policy_loss_inv +
                        self.config.value_coef * value_loss_inv -
                        self.config.entropy_coef_inv * entropy_inv +
                        self.config.value_l2_reg * value_l2_reg_inv +
                        concentration_l2_reg_inv
                    )
                
                # MM agent
                ratio_mm = torch.exp(mm_log_prob - old_log_probs_mm.unsqueeze(-1))
                surrogate1_mm = ratio_mm * advantages_mm.unsqueeze(-1)
                surrogate2_mm = torch.clamp(ratio_mm, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages_mm.unsqueeze(-1)
                policy_loss_mm = -torch.min(surrogate1_mm, surrogate2_mm).mean()
                
                # Value loss (clipped)
                value_loss_mm = F.mse_loss(mm_value, returns_mm.unsqueeze(-1))
                
                # Entropy
                entropy_mm = mm_entropy.mean()
                
                # Regularization
                value_l2_reg_mm = torch.mean(mm_value ** 2)
                # MM agent uses Beta distribution with spread_concentration, not log_std
                # Light regularization on concentration parameters to prevent extreme values
                concentration_l2_reg_mm = torch.sum(
                    (self.policy.mm_agent.spread_concentration.weight ** 2).sum(dim=1)
                )
                concentration_l2_reg_mm = 1e-4 * concentration_l2_reg_mm  # Light regularization
                
                loss_mm = (
                    policy_loss_mm +
                    self.config.value_coef * value_loss_mm -
                    self.config.entropy_coef_mm * entropy_mm +
                    self.config.value_l2_reg * value_l2_reg_mm +
                    concentration_l2_reg_mm
                )
                
                # Backprop BOTH losses FIRST (grads accumulate on shared params)
                if loss_inv is not None:
                    loss_inv.backward()
                loss_mm.backward()
                
                # Clip grads
                grad_norm_inv = nn.utils.clip_grad_norm_(self.policy.inventory_agent.parameters(), self.config.max_grad_norm)
                grad_norm_mm = nn.utils.clip_grad_norm_(self.policy.mm_agent.parameters(), self.config.max_grad_norm)
                
                # Step ALL optimizers
                if loss_inv is not None:
                    self.optimizer_inv.step()
                self.optimizer_mm.step()
                self.optimizer_shared.step()
                
                # Zero ALL
                self.optimizer_inv.zero_grad()
                self.optimizer_mm.zero_grad()
                self.optimizer_shared.zero_grad()
                
                # Accumulate losses
                losses['policy_loss_mm'] += policy_loss_mm.item()
                losses['value_loss_mm'] += value_loss_mm.item()
                losses['entropy_mm'] += entropy_mm.item()
                losses['grad_norm_mm'] += grad_norm_mm.item()
                
                if loss_inv is not None:
                    losses['policy_loss_inv'] += policy_loss_inv.item()
                    losses['value_loss_inv'] += value_loss_inv.item()
                    losses['entropy_inv'] += entropy_inv.item()
                    losses['grad_norm_inv'] += grad_norm_inv.item()
                
                num_updates += 1
        
        # Average losses
        for key in losses:
            if 'advantage' not in key:
                losses[key] /= max(num_updates, 1)
        
        return losses
    
    def _log_epoch(self, epoch: int, losses: Dict[str, float]):
        """Log epoch statistics."""
        # Rewards
        avg_mm_reward = np.mean(self.buffer.mm_rewards)
        avg_inv_reward = np.mean(self.buffer.inv_rewards)
        
        # P&L - env_info is a dict where each key maps to array of shape (num_envs,)
        realized_pnl_list = []
        unrealized_pnl_list = []
        for env_info in (self.buffer.infos if self.buffer.infos else []):
            rpnl = env_info.get('realized_pnl', np.zeros(self.config.num_envs))
            upnl = env_info.get('unrealized_pnl', np.zeros(self.config.num_envs))
            if isinstance(rpnl, np.ndarray):
                realized_pnl_list.extend(rpnl.flatten().tolist())
            else:
                realized_pnl_list.append(float(rpnl))
            if isinstance(upnl, np.ndarray):
                unrealized_pnl_list.extend(upnl.flatten().tolist())
            else:
                unrealized_pnl_list.append(float(upnl))
        avg_realized_pnl = np.mean(realized_pnl_list) if realized_pnl_list else 0.0
        avg_unrealized_pnl = np.mean(unrealized_pnl_list) if unrealized_pnl_list else 0.0
        
        # Actions
        bid_spreads = self.buffer.actions[:, :, 0].flatten()
        ask_spreads = self.buffer.actions[:, :, 1].flatten()
        targets = self.buffer.inv_actions[:, :, 0].flatten()
        avg_bid_spread = np.mean(bid_spreads)
        avg_ask_spread = np.mean(ask_spreads)
        avg_target = np.mean(targets)
        mm_spread_std = np.std(bid_spreads + ask_spreads)
        inv_std = np.std(targets)
        
        # Trading stats - env_info is a dict where each key maps to array
        total_trades = 0
        buy_trades = 0
        sell_trades = 0
        for env_info in (self.buffer.infos if self.buffer.infos else []):
            trade_count = env_info.get('trade_count', np.zeros(self.config.num_envs))
            buy_count = env_info.get('buy_trades', np.zeros(self.config.num_envs))
            sell_count = env_info.get('sell_trades', np.zeros(self.config.num_envs))
            if isinstance(trade_count, np.ndarray):
                total_trades += int(np.sum(trade_count))
            else:
                total_trades += int(trade_count)
            
            if isinstance(buy_count, np.ndarray):
                buy_trades += int(np.sum(buy_count))
            else:
                buy_trades += int(buy_count)
            
            if isinstance(sell_count, np.ndarray):
                sell_trades += int(np.sum(sell_count))
            else:
                sell_trades += int(sell_count)
        
        print(f"\nEpoch {epoch}/{self.config.total_epochs}")
        print(f"MM Reward: {avg_mm_reward:.4f} | Inv Reward: {avg_inv_reward:.4f}")
        print(f"Advantages MM: mean={losses['advantage_mm_mean']:.4f}, std={losses['advantage_mm_std']:.4f}, max={losses['advantage_mm_max']:.4f}")
        print(f"Advantages Inv: mean={losses['advantage_inv_mean']:.4f}, std={losses['advantage_inv_std']:.4f}, max={losses['advantage_inv_max']:.4f}")
        print(f"Policy Loss MM: {losses['policy_loss_mm']:.4f} | Value Loss MM: {losses['value_loss_mm']:.4f}")
        print(f"Policy Loss Inv: {losses['policy_loss_inv']:.4f} | Value Loss Inv: {losses['value_loss_inv']:.4f}")
        print(f"Entropy MM: {losses['entropy_mm']:.4f} | Entropy Inv: {losses['entropy_inv']:.4f}")
        print(f"Grad Norm MM: {losses['grad_norm_mm']:.4f} | Grad Norm Inv: {losses['grad_norm_inv']:.4f}")
        print(f"Avg Bid Spread: {avg_bid_spread:.4f} | Avg Ask Spread: {avg_ask_spread:.4f}")
        print(f"Avg Target Inventory: {avg_target:.4f}")
        print(f"Realized PnL: {avg_realized_pnl:.2f} | Unrealized PnL: {avg_unrealized_pnl:.2f}")
        
        # Log completed episodes
        if self.completed_episodes:
            print("\nCompleted Episodes:")
            for i, ep in enumerate(self.completed_episodes, 1):
                # Net PnL = Realized + Unrealized - Fees
                # Fees are negative for maker rebates (we earn money), positive for taker fees (we pay)
                true_net = ep['realized_pnl'] + ep['unrealized_pnl'] - ep.get('fees', 0.0)
                print(f"Ep {i}: MM Rew {ep['mm_reward']:.2f} | Inv Rew {ep['inv_reward']:.2f} | "
                      f"R.PnL ${ep['realized_pnl']:7.2f} | "
                      f"SprdCap ${ep['spread_capture']:6.2f} | "
                      f"U.PnL ${ep['unrealized_pnl']:7.2f} | "
                      f"Net ${true_net:7.2f} | "
                      f"Trades {ep['trade_count']:4d} | "
                      f"Pos {ep['net_amount_btc']:+.5f} BTC")
            print()
        
        # === TensorBoard Logging ===
        step = self.global_step
        
        try:
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
                avg_episode_mm_rew = np.mean([ep['mm_reward'] for ep in self.completed_episodes])
                avg_episode_inv_rew = np.mean([ep['inv_reward'] for ep in self.completed_episodes])
                avg_spread_capture = np.mean([ep['spread_capture'] for ep in self.completed_episodes])
                avg_unrealized = np.mean([ep['unrealized_pnl'] for ep in self.completed_episodes])
                avg_fees = np.mean([ep['fees'] for ep in self.completed_episodes])
                avg_trades = np.mean([ep['trade_count'] for ep in self.completed_episodes])
                avg_net = np.mean([ep['realized_pnl'] + ep['unrealized_pnl'] - ep.get('fees', 0.0) 
                              for ep in self.completed_episodes])
                
                self.tb_writer.add_scalar('Episode/MM_Reward', avg_episode_mm_rew, step)
                self.tb_writer.add_scalar('Episode/Inv_Reward', avg_episode_inv_rew, step)
                self.tb_writer.add_scalar('Episode/Spread_Capture', avg_spread_capture, step)
                self.tb_writer.add_scalar('Episode/Unrealized_PnL', avg_unrealized, step)
                self.tb_writer.add_scalar('Episode/Fees', avg_fees, step)
                self.tb_writer.add_scalar('Episode/Net_PnL', avg_net, step)
                self.tb_writer.add_scalar('Episode/Trades', avg_trades, step)
            
            # Flush TensorBoard to ensure data is written immediately
            self.tb_writer.flush()
        except Exception as e:
            print(f"Warning: TensorBoard logging error: {e}")
            import traceback
            traceback.print_exc()
    
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
            self.completed_episodes = []
            self.episode_rewards = []
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
            'shared_encoder': self.policy.shared_encoder.state_dict(),
            'inventory_agent': self.policy.inventory_agent.state_dict(),
            'mm_agent': self.policy.mm_agent.state_dict(),
            'optimizer_shared': self.optimizer_shared.state_dict(),
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
        if 'shared_encoder' in checkpoint:
            self.policy.shared_encoder.load_state_dict(checkpoint['shared_encoder'])
        self.policy.inventory_agent.load_state_dict(checkpoint['inventory_agent'])
        self.policy.mm_agent.load_state_dict(checkpoint['mm_agent'])
        self.optimizer_shared.load_state_dict(checkpoint['optimizer_shared'])
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
