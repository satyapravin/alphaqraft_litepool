#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live inference script for trained Hierarchical PPO model.
Runs on production environment with real exchange connection.
"""

# Ensure we use the local litepool, not system-installed version
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
import numpy as np
import torch
import litepool

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from hierarchical_policy import create_hierarchical_policy
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
            ("realized_pnl", "unrealized_pnl", "fees", "leverage", "mm_reward", "inv_reward")}


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
        is_inverse_instr=False,
        api_key=api_key,
        api_secret=api_secret,
        symbol="BTC_USDC-PERPETUAL",
        hedge_symbol="BTC_USDC-18APR25",
        tick_size=0.5,
        min_amount=0.0001,
        maker_fee=-0.000025,
        taker_fee=0.0005,
        foldername="",  # Not used in prod
        balance=2000.0,  # Initial balance (will be fetched from exchange in prod)
        start=1,
        max_episode_steps=MAX_STEPS,
        base_spread_bps=1.0,  # Match training config
        min_size_pct=1.0,     # Match training config
        max_size_pct=25.0,    # Match training config
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
        raise FileNotFoundError(f"No model found in {results_dir}")
    
    # weights_only=False needed for PyTorch 2.6+ (checkpoint contains config objects)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    policy.inventory_agent.load_state_dict(checkpoint['inventory_agent'])
    policy.mm_agent.load_state_dict(checkpoint['mm_agent'])
    print(f"[Model] Loaded hierarchical policy from {model_path}")

    return env, policy


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main():
    print("\n" + "=" * 90)
    print("Hierarchical PPO Live Inference - PRODUCTION MODE")
    print("=" * 90)
    print("WARNING: This will execute real trades on the exchange!")
    print("=" * 90 + "\n")
    
    env, policy = load_model_and_env()
    logger = MetricLogger(print_interval=60)  # Log every minute

    obs, info = env.reset()
    policy.reset(NUM_ENVS)
    print("[Env] Connected and reset")
    
    step = 0
    cum_mm_reward = 0.0
    cum_inv_reward = 0.0
    
    # Tracking
    total_realized_pnl = 0.0
    total_fees = 0.0

    print(f"\n{'Step':>8} | {'MM.Rew':>8} | {'Inv.Rew':>8} | {'R.PnL':>10} | {'U.PnL':>10} | {'Fees':>8} | {'Lev':>6}")
    print("-" * 90)

    try:
        while step < MAX_STEPS:
            # Get action from hierarchical policy (deterministic for production)
            action, info_dict = policy.get_action(obs, deterministic=True)

            # Step environment (this sends orders to exchange!)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)

            # Extract rewards
            pnl = extract_pnl(info)
            mm_r = pnl.get('mm_reward', 0.0)
            inv_r = pnl.get('inv_reward', 0.0)
            
            cum_mm_reward += mm_r
            cum_inv_reward += inv_r
            step += 1

            # Track metrics
            total_realized_pnl = pnl['realized_pnl']
            total_fees = pnl['fees']

            # Print progress (overwrite line)
            print(f"{step:8d} | {mm_r:+8.2f} | {inv_r:+8.2f} | "
                  f"{pnl['realized_pnl']:+10.4f} | {pnl['unrealized_pnl']:+10.4f} | "
                  f"{pnl['fees']:8.4f} | {pnl['leverage']:6.2f}x", end="\r")

            if np.any(done):
                print(f"\n[Session End] steps={step}  MM.Rew={cum_mm_reward:.2f}  Inv.Rew={cum_inv_reward:.2f}  "
                      f"R.PnL={total_realized_pnl:.4f}")
                
                # In production, we typically just continue
                obs, info = env.reset()
                policy.reset(NUM_ENVS)
                cum_mm_reward = 0.0
                cum_inv_reward = 0.0
            else:
                obs = next_obs

    except KeyboardInterrupt:
        print("\n\n[CTRL+C] Shutting down gracefully...")

    finally:
        print("\n" + "=" * 90)
        print("Session Summary")
        print("=" * 90)
        print(f"Total steps:        {step}")
        print(f"MM reward:          {cum_mm_reward:.4f}")
        print(f"Inv reward:         {cum_inv_reward:.4f}")
        print(f"Realized PnL:       {total_realized_pnl:.4f}")
        print(f"Total fees:         {total_fees:.4f}")
        print("=" * 90)
        env.close()
        print("[Env] Closed. Goodbye!")


if __name__ == "__main__":
    main()
