import numpy as np
import torch

class MetricLogger:
    def __init__(self, print_interval=1000):
        self.print_interval = print_interval
        self.last_print_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def log(self, step_count, info, rew, policy):
        if step_count % self.print_interval != 0 and step_count != self.last_print_step:
            return
        self.last_print_step = step_count

        print(f"\nStep: {step_count}")
        print("Env | Net_PnL   | R_PnL     | UR_PnL    | Fees       | Trades   | Drawdown | Leverage | Reward")
        print("-" * 80)

        # Helper function to extract scalar from value
        def to_scalar(value):
            if isinstance(value, np.ndarray):
                return value.item() if value.size == 1 else value[-1]  # Last element if multi-element
            return value

        # Handle info structure
        if isinstance(info, dict):
            # info is {key: [env_list_of_steps, ...]}, extract latest step
            num_envs = len(next(iter(info.values())))
            env_ids = range(num_envs)
            for ii in env_ids:
                # Get the latest step's value for each key, ensure scalar
                latest_info = {
                    k: to_scalar(v[ii][-1]) if isinstance(v[ii], (list, np.ndarray)) else to_scalar(v[ii])
                    for k, v in info.items()
                }
                net_pnl = (latest_info.get('realized_pnl', 0) + 
                          latest_info.get('unrealized_pnl', 0) - 
                          latest_info.get('fees', 0))
                net_pnl = to_scalar(net_pnl)  # Ensure scalar
                r_pnl = latest_info.get('realized_pnl', 0)
                ur_pnl = latest_info.get('unrealized_pnl', 0)
                fees = latest_info.get('fees', 0)
                trades = latest_info.get('trade_count', 0)
                drawdown = latest_info.get('drawdown', 0)
                leverage = latest_info.get('leverage', 0)
                reward = to_scalar(rew[ii])
                
                print(f"{ii:3d} | {net_pnl:+7.6f} | "
                      f"{r_pnl:+6.6f} | "
                      f"{ur_pnl:+6.6f} | "
                      f"{fees:+6.6f} | "
                      f"{trades:8d} | "
                      f"{drawdown:+8.6f} | "
                      f"{leverage:+8.6f} | "
                      f"{reward:+8.6f}")
        else:
            # info is a list/array of [env, steps], assume latest step
            num_envs = len(info)
            env_ids = range(num_envs)
            for ii in env_ids:
                latest_info = info[ii][-1] if isinstance(info[ii], (list, np.ndarray)) else info[ii]
                if isinstance(latest_info, dict):
                    # Ensure scalar values
                    latest_info = {k: to_scalar(v) for k, v in latest_info.items()}
                    net_pnl = (latest_info.get('realized_pnl', 0) + 
                              latest_info.get('unrealized_pnl', 0) - 
                              latest_info.get('fees', 0))
                    net_pnl = to_scalar(net_pnl)
                    r_pnl = latest_info.get('realized_pnl', 0)
                    ur_pnl = latest_info.get('unrealized_pnl', 0)
                    fees = latest_info.get('fees', 0)
                    trades = latest_info.get('trade_count', 0)
                    drawdown = latest_info.get('drawdown', 0)
                    leverage = latest_info.get('leverage', 0)
                    reward = to_scalar(rew[ii])
                    
                    print(f"{ii:3d} | {net_pnl:+7.6f} | "
                          f"{r_pnl:+6.6f} | "
                          f"{ur_pnl:+6.6f} | "
                          f"{fees:+6.6f} | "
                          f"{trades:8f} | "
                          f"{drawdown:+8.6f} | "
                          f"{leverage:+8.6f} | "
                          f"{reward:+8.6f}")
                else:
                    # Fallback if info is not a dict
                    reward = to_scalar(rew[ii])
                    print(f"{ii:3d} | {0:+7.6f} | {0:+6.6f} | {0:+6.6f} | {0:+6.6f} | "
                          f"{0:8d} | {0:+8.6f} | {0:+8.6f} | {reward:+8.6f}")

        if hasattr(policy, 'get_alpha'):
            alpha = policy.get_alpha.item() if isinstance(policy.get_alpha, torch.Tensor) else policy.get_alpha
            print(f"\nAlpha: {alpha:.6f}")
        print("-" * 80)
