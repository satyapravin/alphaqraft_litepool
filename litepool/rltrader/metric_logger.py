import numpy as np
import torch

import numpy as np
import torch

class MetricLogger:
    def __init__(self, print_interval=1000):
        self.print_interval = print_interval
        self.last_print_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def log(self, step_count, info, rew, policy):
        if step_count % self.print_interval != 0:
            return
        self.last_print_step = step_count

        print(f"\nStep: {step_count}")
        print("Env | Net_PnL   | R_PnL     | UR_PnL    | Fees       | Trades   | Drawdown | Leverage | Reward")
        print("-" * 80)

        # Assuming info is a dict with keys mapping to arrays of length 64 (num_envs)
        num_envs = 64  # Based on your setup with 64 environments
        env_ids = range(num_envs)

        for env_id in env_ids:
            # Extract values directly from info dictionary for this environment
            realized_pnl = info.get('realized_pnl', np.zeros(num_envs))[env_id]
            unrealized_pnl = info.get('unrealized_pnl', np.zeros(num_envs))[env_id]
            fees = info.get('fees', np.zeros(num_envs))[env_id]
            trades = info.get('trade_count', np.zeros(num_envs))[env_id]
            drawdown = info.get('drawdown', np.zeros(num_envs))[env_id]
            leverage = info.get('leverage', np.zeros(num_envs))[env_id]
            reward = rew[env_id] if isinstance(rew, (np.ndarray, list)) and len(rew) > env_id else 0.0

            # Calculate net PnL
            net_pnl = realized_pnl + unrealized_pnl - fees

            # Convert to scalar if needed (in case values are tensors or arrays)
            def to_scalar(value):
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    return value.item() if value.size == 1 else float(value)
                return float(value)

            net_pnl = to_scalar(net_pnl)
            realized_pnl = to_scalar(realized_pnl)
            unrealized_pnl = to_scalar(unrealized_pnl)
            fees = to_scalar(fees)
            trades = to_scalar(trades)
            drawdown = to_scalar(drawdown)
            leverage = to_scalar(leverage)
            reward = to_scalar(reward)

            # Print formatted row for this environment
            print(f"{env_id:3d} | {net_pnl:+7.6f} | "
                  f"{realized_pnl:+6.6f} | "
                  f"{unrealized_pnl:+6.6f} | "
                  f"{fees:+6.6f} | "
                  f"{trades:8.0f} | "
                  f"{drawdown:+8.6f} | "
                  f"{leverage:+8.6f} | "
                  f"{reward:+8.6f}")

        # Print alpha value if available
        if hasattr(policy, 'get_alpha'):
            alpha = policy.get_alpha.item() if isinstance(policy.get_alpha, torch.Tensor) else policy.get_alpha
            print(f"\nAlpha: {alpha:.6f}")
        print("-" * 80)
