import numpy as np
import torch

def to_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        else:
            return value.mean().item()
    return float(value)

class MetricLogger:
    def __init__(self, print_interval=1000):
        self.print_interval = print_interval
        self.last_print_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def log(self, step_count, infos, rew, policy):
        if step_count % self.print_interval != 0:
            return
        self.last_print_step = step_count

        print(f"\nStep: {step_count}")
        print(f"{'Env':>3} | {'Net_PnL':>10} | {'Realized':>10} | {'Unrealized':>10} | {'Fees':>10} | {'Trades':>7} | {'Drawdown':>10} | {'Leverage':>10} | {'Reward':>10}")
        print("-" * 100)

        num_envs = len(rew) if isinstance(rew, (np.ndarray, list)) else 64
        env_ids = range(num_envs)
        infos = infos['infos'][-1]
        print(rew[0])

        for env_id in env_ids:
            # Now correctly handle infos as a list of dicts
            realized_pnl = infos['realized_pnl'][env_id]
            unrealized_pnl = infos['unrealized_pnl'][env_id]
            fees = infos['fees'][env_id]
            trades = infos['trade_count'][env_id] 
            drawdown = infos['drawdown'][env_id] 
            leverage = infos['leverage'][env_id] 
            reward = to_scalar(rew[env_id]) if isinstance(rew, (np.ndarray, list)) and len(rew) > env_id else 0.0

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

        if hasattr(policy, 'get_alpha'):
            alpha = policy.get_alpha.item() if isinstance(policy.get_alpha, torch.Tensor) else policy.get_alpha
            print(f"\nAlpha: {alpha:.6f}")
        print("-" * 100)
