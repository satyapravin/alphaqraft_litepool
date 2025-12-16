#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live inference script for trained PPO model.
Runs on production environment with real exchange connection.
"""

import os
from pathlib import Path
import numpy as np
import torch
import litepool

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from vec_normalizer import VecNormalize
from simple_actor_critic import SimpleActorCritic
from metric_logger import MetricLogger

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
device = torch.device("cpu")
NUM_ENVS = 1
MAX_STEPS = 3600 * 24 * 7  # Run for up to a week

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
    """Load production environment and trained model."""
    
    api_key = os.environ.get("RLTRADER_API_KEY", "")
    api_secret = os.environ.get("RLTRADER_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("WARNING: API credentials not found in environment variables!")
        print("Set RLTRADER_API_KEY and RLTRADER_API_SECRET in .env file")
    
    env = litepool.make(
        "RlTrader-v0",
        env_type="gymnasium",
        num_envs=NUM_ENVS,
        batch_size=NUM_ENVS,
        num_threads=1,
        is_prod=True,  # Production mode!
        is_inverse_instr=True,
        api_key=api_key,
        api_secret=api_secret,
        symbol="BTC-PERPETUAL",
        hedge_symbol="BTC-18APR25",
        tick_size=0.5,
        min_amount=10,
        maker_fee=-0.0001,
        taker_fee=0.0005,
        foldername="",  # Not used in prod
        balance=1.0,  # Will be fetched from exchange
        start=1,
        max=MAX_STEPS,
    )
    env.spec.id = "RlTrader-v0"

    env = VecNormalize(
        env,
        device=device,
        num_envs=NUM_ENVS,
        obs_dim=16,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )

    # Note: VecNormalize no longer needs saved stats since all signals 
    # are bounded to [-1, 1] in C++. Just optional clipping for safety.
    results_dir = Path("results")

    # Model
    model = SimpleActorCritic(
        obs_dim=16,
        action_dim=4,
        hidden_dim=64,
    )
    model.eval()

    # Load best model for production
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
    print("\n" + "=" * 70)
    print("PPO Live Inference - PRODUCTION MODE")
    print("=" * 70)
    print("WARNING: This will execute real trades on the exchange!")
    print("=" * 70 + "\n")
    
    env, model = load_model_and_env()
    logger = MetricLogger(print_interval=60)  # Log every minute

    obs, info = env.reset()
    print("[Env] Connected and reset")
    
    step = 0
    cum_reward = 0.0
    ep_rewards = []
    ep_infos = []
    
    # Tracking
    total_realized_pnl = 0.0
    total_fees = 0.0

    print(f"\n{'Step':>8} | {'Reward':>8} | {'R.PnL':>10} | {'U.PnL':>10} | {'Fees':>8} | {'Lev':>6}")
    print("-" * 70)

    try:
        while step < MAX_STEPS:
            # Get action from model (deterministic for production)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action, _, _ = model.get_action(obs_tensor, deterministic=True)
            
            # Ensure action is 2D for vectorized env
            if action.ndim == 1:
                action = action.reshape(1, -1)

            # Step environment (this sends orders to exchange!)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)

            r = float(reward[0] if isinstance(reward, np.ndarray) else reward)
            cum_reward += r
            ep_rewards.append(r)
            ep_infos.append(info)
            step += 1

            # Extract PnL info
            pnl = extract_pnl(info)
            total_realized_pnl = pnl['realized_pnl']
            total_fees = pnl['fees']

            # Print progress
            print(f"{step:8d} | {r:+8.4f} | "
                  f"{pnl['realized_pnl']:+10.6f} | {pnl['unrealized_pnl']:+10.6f} | "
                  f"{pnl['fees']:8.6f} | {pnl['leverage']:6.2f}x", end="\r")

            if np.any(done):
                print(f"\n[Session End] steps={step}  ΣR={sum(ep_rewards):.4f}  "
                      f"R.PnL={total_realized_pnl:.6f}")
                
                # In production, we typically just continue
                obs, info = env.reset()
                ep_rewards = []
                ep_infos = []
            else:
                obs = next_obs

    except KeyboardInterrupt:
        print("\n\n[CTRL+C] Shutting down gracefully...")

    finally:
        print("\n" + "=" * 70)
        print("Session Summary")
        print("=" * 70)
        print(f"Total steps:        {step}")
        print(f"Cumulative reward:  {cum_reward:.4f}")
        print(f"Realized PnL:       {total_realized_pnl:.6f}")
        print(f"Total fees:         {total_fees:.6f}")
        print("=" * 70)
        env.close()
        print("[Env] Closed. Goodbye!")


if __name__ == "__main__":
    main()
