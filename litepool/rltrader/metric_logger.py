class MetricLogger:
    def __init__(self, print_interval=1000):
        self.print_interval = print_interval
        self.last_print_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def log(self, step_count, info, rew, policy):
        if step_count % self.print_interval == 0:
            print(f"\nStep: {step_count}")
            print("Env | Net_PnL   | R_PnL     | UR_PnL    | Fees       | Trades   | Drawdown | Leverage | Reward")
            print("-" * 80)

            for ii in info['env_id']:
                net_pnl = info['realized_pnl'][ii] + info['unrealized_pnl'][ii] - info['fees'][ii]
                print(f"{ii:3d} | {net_pnl:+7.6f} | {info['realized_pnl'][ii]:+6.6f} | "
                      f"{info['unrealized_pnl'][ii]:+6.6f} | {info['fees'][ii]:+6.6f} | "
                      f"{info['trade_count'][ii]} | {info['drawdown'][ii]:+8.6f} | "
                      f"{info['leverage'][ii]:+8.6f} | {rew[ii]:+8.6f}")

            if hasattr(policy, 'get_alpha'):
                print(f"\nAlpha: {policy.get_alpha.item():.6f}")
            print("-" * 80)

