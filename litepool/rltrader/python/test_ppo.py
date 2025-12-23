#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test/evaluation script for trained Hierarchical PPO model.
Runs on simulated environment to evaluate performance.
"""

from pathlib import Path
import numpy as np
import torch
import litepool

from hierarchical_policy import create_hierarchical_policy
from metric_logger import MetricLogger

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
device = torch.device("cpu")
NUM_ENVS = 1
MAX_STEPS = 3600 * 100  # 100 hours of testing

# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def extract_pnl(info):
    """Safely pull scalar PnL, fees, leverage from info dict."""
    if isinstance(info, list):
        info = info[0]

    def _get(k):
        v = info.get(k, 0.0)
        return float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)

    return {k: _get(k) for k in
            ("realized_pnl", "unrealized_pnl", "fees", "leverage", "mm_reward", "inv_reward")}


# --------------------------------------------------------------------------- #
# Load env & model                                                            #
# --------------------------------------------------------------------------- #
def load_model_and_env():
    """Load test environment and trained model."""
    env = litepool.make(
        "RlTrader-v0",
        env_type="gymnasium",
        num_envs=NUM_ENVS,
        batch_size=NUM_ENVS,
        num_threads=1,
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
        foldername="/home/pravin/dev/alphaqraft_litepool/data/testing/",
        balance=20000.0,
        start=1,
        max_episode_steps=MAX_STEPS,
    )
    env.spec.id = "RlTrader-v0"

    # Create hierarchical policy
    policy = create_hierarchical_policy(
        inventory_update_freq=100,
        device=str(device),
    )
    policy.eval()

    # Load trained model
    results_dir = Path("results/hierarchical")
    model_path = results_dir / "best_model.pt"
    if not model_path.exists():
        model_path = results_dir / "final_model.pt"
    if not model_path.exists():
        # Try old location
        model_path = Path("results") / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {results_dir}")
    
    checkpoint = torch.load(model_path, map_location=device)
    policy.inventory_agent.load_state_dict(checkpoint['inventory_agent'])
    policy.mm_agent.load_state_dict(checkpoint['mm_agent'])
    print(f"[Model] Loaded hierarchical policy from {model_path}")

    return env, policy


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main():
    env, policy = load_model_and_env()
    logger = MetricLogger(print_interval=512)

    obs, info = env.reset()
    policy.reset(NUM_ENVS)
    
    step = 0
    cum_reward = 0.0
    cum_mm_reward = 0.0
    cum_inv_reward = 0.0
    ep_len = 0
    
    # Tracking metrics
    total_realized_pnl = 0.0
    total_fees = 0.0
    max_leverage = 0.0

    print("\n" + "=" * 90)
    print("Hierarchical PPO Model Evaluation")
    print("=" * 90)
    print(f"{'Step':>8} | {'MM.Rew':>8} | {'Inv.Rew':>8} | {'R.PnL':>10} | {'U.PnL':>10} | {'Fees':>8} | {'Lev':>6}")
    print("-" * 90)

    try:
        while step < MAX_STEPS:
            # Get action from hierarchical policy (deterministic for evaluation)
            action, info_dict = policy.get_action(obs, deterministic=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)

            # Extract rewards
            pnl = extract_pnl(info)
            mm_r = pnl.get('mm_reward', 0.0)
            inv_r = pnl.get('inv_reward', 0.0)
            
            cum_reward += float(reward[0] if isinstance(reward, np.ndarray) else reward)
            cum_mm_reward += mm_r
            cum_inv_reward += inv_r
            ep_len += 1
            step += 1

            # Track metrics
            total_realized_pnl = pnl['realized_pnl']
            total_fees = pnl['fees']
            max_leverage = max(max_leverage, pnl['leverage'])

            # Print progress
            if step % 100 == 0:
                print(f"{step:8d} | {mm_r:+8.2f} | {inv_r:+8.2f} | "
                      f"{pnl['realized_pnl']:+10.4f} | {pnl['unrealized_pnl']:+10.4f} | "
                      f"{pnl['fees']:8.4f} | {pnl['leverage']:6.2f}x")

            if np.any(done):
                print(f"\n[Episode End] len={ep_len}  MM.Rew={cum_mm_reward:.2f}  Inv.Rew={cum_inv_reward:.2f}  "
                      f"R.PnL={total_realized_pnl:.4f}  Fees={total_fees:.4f}")
                print("-" * 90)
                
                # Reset for next episode
                obs, info = env.reset()
                policy.reset(NUM_ENVS)
                cum_reward = 0.0
                cum_mm_reward = 0.0
                cum_inv_reward = 0.0
                ep_len = 0
            else:
                obs = next_obs

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")

    finally:
        print("\n" + "=" * 90)
        print("Evaluation Summary")
        print("=" * 90)
        print(f"Total steps:        {step}")
        print(f"Cumulative reward:  {cum_reward:.4f}")
        print(f"  MM reward:        {cum_mm_reward:.4f}")
        print(f"  Inv reward:       {cum_inv_reward:.4f}")
        print(f"Realized PnL:       {total_realized_pnl:.4f}")
        print(f"Total fees:         {total_fees:.4f}")
        print(f"Max leverage:       {max_leverage:.2f}x")
        print("=" * 90)
        env.close()


if __name__ == "__main__":
    main()
