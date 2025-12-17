#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test/evaluation script for trained PPO model.
Runs on simulated environment to evaluate performance.
"""

from pathlib import Path
import numpy as np
import torch
import litepool

from simple_actor_critic import SimpleActorCritic
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
            ("realized_pnl", "unrealized_pnl", "fees", "leverage")}


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
        is_inverse_instr=True,
        api_key="",
        api_secret="",
        symbol="BTC-PERPETUAL",
        hedge_symbol="BTC-18APR25",
        tick_size=0.5,
        min_amount=10,
        maker_fee=-0.0001,
        taker_fee=0.0005,
        foldername="/home/pravin/dev/alphaqraft_litepool/data/testing/",
        balance=1.0,
        start=1,
        max=MAX_STEPS,
    )
    env.spec.id = "RlTrader-v0"

    # All observation signals are already bounded to [-1, 1]:
    # - Market signals (13): all use tanh or are bounded by construction
    # - AMM signals (3): all clamped or bounded to [-1, 1]
    # No normalization needed!
    results_dir = Path("results")

    # Model
    model = SimpleActorCritic(
        obs_dim=16,
        action_dim=5,  # 4 continuous (spread, size, skew, target_inventory) + 1 binary (requote)
        hidden_dim=64,
    )
    model.eval()

    # Try to load best model first, fall back to final
    model_path = results_dir / "best_model.pth"
    if not model_path.exists():
        model_path = results_dir / "final_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {results_dir}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[Model] Loaded weights from {model_path}")

    return env, model


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main():
    env, model = load_model_and_env()
    logger = MetricLogger(print_interval=512)

    obs, info = env.reset()
    
    step = 0
    cum_reward = 0.0
    ep_rewards = []
    ep_infos = []
    ep_len = 0
    
    # Tracking metrics
    total_realized_pnl = 0.0
    total_fees = 0.0
    max_leverage = 0.0

    print("\n" + "=" * 70)
    print("PPO Model Evaluation")
    print("=" * 70)
    print(f"{'Step':>8} | {'Reward':>8} | {'Cum.R':>8} | {'R.PnL':>10} | {'U.PnL':>10} | {'Fees':>8} | {'Lev':>6}")
    print("-" * 70)

    try:
        while step < MAX_STEPS:
            # Get action from model (deterministic for evaluation)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action, _, _ = model.get_action(obs_tensor, deterministic=True)
            
            # Ensure action is 2D for vectorized env
            if action.ndim == 1:
                action = action.reshape(1, -1)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)

            r = float(reward[0] if isinstance(reward, np.ndarray) else reward)
            cum_reward += r
            ep_rewards.append(r)
            ep_infos.append(info)
            ep_len += 1
            step += 1

            # Extract PnL info
            pnl = extract_pnl(info)
            total_realized_pnl = pnl['realized_pnl']
            total_fees = pnl['fees']
            max_leverage = max(max_leverage, pnl['leverage'])

            # Print progress
            if step % 100 == 0:
                print(f"{step:8d} | {r:+8.4f} | {cum_reward:+8.2f} | "
                      f"{pnl['realized_pnl']:+10.6f} | {pnl['unrealized_pnl']:+10.6f} | "
                      f"{pnl['fees']:8.6f} | {pnl['leverage']:6.2f}x")

            if np.any(done):
                print(f"\n[Episode End] len={ep_len}  ΣR={sum(ep_rewards):.4f}  "
                      f"R.PnL={total_realized_pnl:.6f}  Fees={total_fees:.6f}")
                print("-" * 70)
                
                # Reset for next episode
                obs, info = env.reset()
                ep_rewards = []
                ep_infos = []
                ep_len = 0
            else:
                obs = next_obs

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")

    finally:
        print("\n" + "=" * 70)
        print("Evaluation Summary")
        print("=" * 70)
        print(f"Total steps:        {step}")
        print(f"Cumulative reward:  {cum_reward:.4f}")
        print(f"Realized PnL:       {total_realized_pnl:.6f}")
        print(f"Total fees:         {total_fees:.6f}")
        print(f"Max leverage:       {max_leverage:.2f}x")
        print("=" * 70)
        env.close()


if __name__ == "__main__":
    main()
