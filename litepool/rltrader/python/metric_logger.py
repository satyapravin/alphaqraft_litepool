# Copyright 2024 Alphaqraft
#
# Metric Logger for RL Trading
# Logs per-environment metrics during training

import numpy as np
import torch


def to_scalar(value):
    """Convert various types to scalar float."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        else:
            return value.mean().item()
    return float(value)


class MetricLogger:
    """Logger for training metrics with per-environment breakdown."""
    
    def __init__(self, print_interval=1024):
        self.print_interval = print_interval
        self.last_print_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def log(self, step_count, infos, rew, policy=None, num_envs=None):
        """
        Log metrics for current step.
        
        Args:
            step_count: Current training step
            infos: Environment info dict (keys map to arrays of shape (num_envs,))
            rew: Rewards array of shape (n_steps, num_envs) or (num_envs,)
            policy: Policy object (optional, for alpha logging if available)
            num_envs: Number of environments (auto-detected from rew if not provided)
        """
        self.last_print_step = step_count

        print(f"\nStep: {step_count}")
        print(f"{'Env':>3} | {'Net_PnL':>10} | {'Realized':>10} | {'Unrealized':>10} | {'Fees':>10} | {'Trades':>7} | {'Drawdown':>10} | {'Leverage':>10} | {'Reward':>10}")
        print("-" * 100)

        # Determine number of environments from reward array
        if num_envs is None:
            if isinstance(rew, (np.ndarray, list)):
                if len(rew.shape) > 1:
                    num_envs = rew.shape[-1]
                else:
                    num_envs = len(rew)
            else:
                num_envs = 8  # Default fallback
        
        env_ids = range(num_envs)
        
        # Handle infos structure (dict of arrays or nested dict)
        if 'infos' in infos:
            infos = infos['infos']
        
        # Sum rewards across steps if needed
        if isinstance(rew, (np.ndarray, torch.Tensor)) and len(rew.shape) > 1:
            rewlast = rew.sum(axis=0)
        else:
            rewlast = rew

        for env_id in env_ids:
            # Extract values safely with fallbacks
            def get_value(key, default=0.0):
                val = infos.get(key, default)
                if isinstance(val, (np.ndarray, list)):
                    return val[env_id] if env_id < len(val) else default
                return val
            
            realized_pnl = get_value('realized_pnl', 0.0)
            unrealized_pnl = get_value('unrealized_pnl', 0.0)
            fees = get_value('fees', 0.0)
            trades = get_value('trade_count', 0)
            drawdown = get_value('drawdown', 0.0)
            leverage = get_value('leverage', 0.0)
            
            # Get reward for this environment
            if isinstance(rewlast, (torch.Tensor, np.ndarray)):
                reward = rewlast[env_id].item() if hasattr(rewlast[env_id], 'item') else rewlast[env_id]
            else:
                reward = rewlast

            # Net P&L = Realized + Unrealized - Fees (fees are positive for costs, negative for rebates)
            net_pnl = realized_pnl + unrealized_pnl - fees

            print(f"{env_id:3d} | "
                  f"{net_pnl:+10.6f} | "
                  f"{realized_pnl:+10.6f} | "
                  f"{unrealized_pnl:+10.6f} | "
                  f"{fees:+10.6f} | "
                  f"{int(trades):7d} | "
                  f"{drawdown:+10.6f} | "
                  f"{leverage:+10.6f} | "
                  f"{reward:+10.6f}")

        # Summary statistics
        if isinstance(rew, torch.Tensor):
            rew = rew.detach().cpu().numpy()
        rew_flat = np.array(rew).flatten()
        avg_reward = np.mean(rew_flat) if len(rew_flat) > 0 else 0.0
        min_reward = np.min(rew_flat) if len(rew_flat) > 0 else 0.0
        max_reward = np.max(rew_flat) if len(rew_flat) > 0 else 0.0

        print(f"\n[Rewards] Avg: {avg_reward:+.6f} | Min: {min_reward:+.6f} | Max: {max_reward:+.6f}")
        print("-" * 100)

