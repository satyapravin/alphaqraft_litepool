#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test/evaluation script for trained Hierarchical PPO model.
Based on hierarchical_ppo.py structure, simplified for evaluation only.
"""

# Ensure we use the local litepool, not system-installed version
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels to project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import litepool
from hierarchical_policy import create_hierarchical_policy
from hierarchical_config import HierarchicalConfig

# === Device setup ===
device = torch.device("cpu")
print(f"Using device: {device}")

# === Configuration ===
NUM_ENVS = 1
MAX_STEPS = 7200 * 20  # 100 hours of testing
USE_DETERMINISTIC = True

config = HierarchicalConfig()


def _extract_info_value(env_info: dict, key: str, env_id: int, default=0.0) -> float:
    """Safely extract a value from env_info for a specific environment."""
    val = env_info.get(key, default)
    if isinstance(val, np.ndarray):
        return float(val[env_id]) if env_id < len(val) else default
    return float(val) if val is not None else default


def main():
    """Main evaluation loop."""
    # Create environment
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
        balance=config.balance,
        start=1,
        max_episode_steps=MAX_STEPS,
        base_spread_bps=config.base_spread_bps,
        min_size_pct=config.min_size_pct,
        max_size_pct=config.max_size_pct,
    )
    env.spec.id = "RlTrader-v0"

    # Create hierarchical policy (matching training config)
    policy = create_hierarchical_policy(
        inventory_update_freq=config.inventory_update_freq,
        device=str(device),
        target_range=config.target_range,  # Match training config
    )
    policy.eval()

    # Load trained model
    results_dir = Path("results/hierarchical")
    model_path = results_dir / "best_model.pt"
    if not model_path.exists():
        model_path = results_dir / "final_model.pt"
    if not model_path.exists():
        model_path = results_dir / "latest.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {results_dir}")
    
    # weights_only=False needed for PyTorch 2.6+ (checkpoint contains config objects)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    policy.inventory_agent.load_state_dict(checkpoint['inventory_agent'])
    policy.mm_agent.load_state_dict(checkpoint['mm_agent'])
    print(f"[Model] Loaded hierarchical policy from {model_path}")

    # Initialize
    obs, _ = env.reset()
    policy.reset(NUM_ENVS)
    
    # Per-env tracking (matching hierarchical_ppo.py structure)
    episode_steps = np.zeros(NUM_ENVS, dtype=np.int32)
    episode_mm_total = np.zeros(NUM_ENVS)
    episode_inv_total = np.zeros(NUM_ENVS)
    
    # Metrics tracking
    max_leverage = 0.0
    total_steps = 0
    total_realized_pnl = 0.0  # LIFO-based
    total_unrealized_pnl = 0.0  # LIFO-based
    total_realized_pnl_avg = 0.0  # Average cost-based (for comparison)
    total_unrealized_pnl_avg = 0.0  # Average cost-based (for comparison)
    total_trades = 0
    total_fees = 0.0
    last_env_info = None  # Store last env_info for final summary
    last_target_inventory = None  # Track target inventory changes
    target_change_count = 0  # Count how many times target changes
    
    print("\n" + "="*80)
    print("Hierarchical PPO Model Evaluation")
    print("="*80)
    print(f"Mode: {'DETERMINISTIC' if USE_DETERMINISTIC else 'EXPLORATION'} (matches training)")
    print(f"Max steps: {MAX_STEPS}")
    print("="*80)
    print(f"{'Step':>8} | {'MM.Rew':>8} | {'Inv.Rew':>8} | {'R.PnL':>10} | {'U.PnL':>10} | "
          f"{'Fees':>8} | {'Lev':>6} | {'Trades':>6} | {'Target':>12} | {'RiskAv':>8}")
    print("-"*100)
    
    try:
        while total_steps < MAX_STEPS:
            # Get action from hierarchical policy (deterministic=False matches training)
            action, info = policy.get_action(obs, deterministic=USE_DETERMINISTIC)

            # Step environment
            next_obs, reward, terminated, truncated, env_info = env.step(action)
            last_env_info = env_info  # Store for final summary
            done = terminated | truncated
            
            # Extract rewards (matching hierarchical_ppo.py structure)
            mm_reward = env_info.get('mm_reward', np.zeros(NUM_ENVS))
            inv_reward = env_info.get('inv_reward', np.zeros(NUM_ENVS))
            
            # Handle numpy array extraction
            if isinstance(mm_reward, np.ndarray):
                mm_reward = mm_reward.flatten()
            else:
                mm_reward = np.array([mm_reward] * NUM_ENVS)
            if isinstance(inv_reward, np.ndarray):
                inv_reward = inv_reward.flatten()
            else:
                inv_reward = np.array([inv_reward] * NUM_ENVS)
            
            # Accumulate rewards (matching hierarchical_ppo.py)
            episode_mm_total += mm_reward
            episode_inv_total += inv_reward
            episode_steps += 1
            total_steps += NUM_ENVS

            # Track metrics
            leverage = _extract_info_value(env_info, 'leverage', 0, 0.0)
            max_leverage = max(max_leverage, abs(leverage))
            target_inventory = _extract_info_value(env_info, 'target_inventory', 0, 0.0)
            risk_aversion = _extract_info_value(env_info, 'risk_aversion', 0, 0.5)
            
            # Track target inventory changes
            if last_target_inventory is not None:
                if abs(target_inventory - last_target_inventory) > 1e-6:
                    target_change_count += 1
                    if target_change_count <= 10:  # Print first 10 changes
                        print(f"[Target Change] Step {total_steps}: {last_target_inventory:.6f} -> {target_inventory:.6f}")
            last_target_inventory = target_inventory

            # Print progress
            if total_steps % 100 == 0:
                realized_pnl = _extract_info_value(env_info, 'realized_pnl', 0, 0.0)
                unrealized_pnl = _extract_info_value(env_info, 'lifo_unrealized_pnl', 0, 0.0)
                fees = _extract_info_value(env_info, 'fees', 0, 0.0)
                trade_count = int(_extract_info_value(env_info, 'trade_count', 0, 0.0))
                
                mm_r = float(episode_mm_total[0])
                inv_r = float(episode_inv_total[0])
                
                # Get raw actions from policy info (for debugging)
                # target_inventory is in [-target_range, +target_range], used directly in C++
                # risk_aversion is in [0, 1]
                raw_target = info.get('targets', np.array([0.0])) if 'targets' in info else np.array([0.0])
                raw_risk_aversion = info.get('risk_aversion', np.array([0.5])) if 'risk_aversion' in info else np.array([0.5])
                
                if isinstance(raw_target, np.ndarray):
                    raw_target = float(raw_target.item()) if raw_target.size > 0 else 0.0
                else:
                    raw_target = float(raw_target)
                
                if isinstance(raw_risk_aversion, np.ndarray):
                    raw_risk_aversion = float(raw_risk_aversion.item()) if raw_risk_aversion.size > 0 else 1.0
                else:
                    raw_risk_aversion = float(raw_risk_aversion)
                
                # C++ uses raw actions directly (no scaling)
                scaled_action = raw_target
                
                updated = info.get('updated_inventory', np.array([False]))
                if isinstance(updated, np.ndarray):
                    updated = bool(updated[0]) if len(updated) > 0 else False
                else:
                    updated = bool(updated)
                
                # Display: Tgt = EMA-smoothed target from env (actual value used in strategy)
                #         act = raw action (what C++ receives before EMA)
                #         RiskAv = risk aversion parameter (γ) for A-S model
                print(f"{total_steps:8d} | {mm_r:+8.2f} | {inv_r:+8.2f} | "
                      f"{realized_pnl:+10.4f} | {unrealized_pnl:+10.4f} | "
                      f"{fees:8.4f} | {leverage:6.2f}x | {trade_count:6d} | "
                      f"Tgt:{target_inventory:+.4f} (act:{scaled_action:+.4f}{'*' if updated else ''}) | "
                      f"γ:{risk_aversion:.3f}")
            
            # Debug: Print when inventory agent updates
            if 'updated_inventory' in info:
                updated = info.get('updated_inventory', np.array([False]))
                if isinstance(updated, np.ndarray):
                    updated = bool(updated.item()) if updated.size > 0 else False
                else:
                    updated = bool(updated)
                if updated:
                    raw_target = info.get('targets', np.array([0.0]))
                    raw_risk_aversion = info.get('risk_aversion', np.array([1.0]))
                    if isinstance(raw_target, np.ndarray):
                        raw_target = float(raw_target.item()) if raw_target.size > 0 else 0.0
                    else:
                        raw_target = float(raw_target)
                    if isinstance(raw_risk_aversion, np.ndarray):
                        raw_risk_aversion = float(raw_risk_aversion.item()) if raw_risk_aversion.size > 0 else 1.0
                    else:
                        raw_risk_aversion = float(raw_risk_aversion)
                    # C++ uses raw actions directly (no scaling)
                    current_target = raw_target  # Use raw value for display
                    current_risk_aversion = raw_risk_aversion
            
            # Handle episode ends (matching hierarchical_ppo.py structure)
            for env_id in range(NUM_ENVS):
                if done[env_id]:
                    # Extract terminal info
                    realized_pnl = _extract_info_value(env_info, 'final_realized_pnl', env_id, 0.0)  # LIFO
                    unrealized_pnl = _extract_info_value(env_info, 'final_unrealized_pnl', env_id, 0.0)  # LIFO
                    spread_capture = _extract_info_value(env_info, 'final_spread_capture', env_id, 0.0)
                    fees = _extract_info_value(env_info, 'final_fees', env_id, 0.0)
                    trade_count = int(_extract_info_value(env_info, 'final_trade_count', env_id, 0.0))
                    net_amount_btc = _extract_info_value(env_info, 'final_net_amount_btc', env_id, 0.0)
                    
                    # Calculate average cost-based PnL for comparison
                    # Use regular info keys (balance and initial_balance are in current info, not final_* keys)
                    balance = _extract_info_value(env_info, 'balance', env_id, 0.0)
                    initial_balance = _extract_info_value(env_info, 'initial_balance', env_id, 0.0)
                    unrealized_pnl_avg = _extract_info_value(env_info, 'unrealized_pnl', env_id, 0.0)  # Average cost
                    realized_pnl_avg = balance - initial_balance  # Average cost-based realized
                    
                    # Accumulate totals across all episodes
                    total_realized_pnl += realized_pnl  # LIFO
                    total_unrealized_pnl = unrealized_pnl  # LIFO (use final unrealized - current position)
                    total_realized_pnl_avg += realized_pnl_avg  # Average cost
                    total_unrealized_pnl_avg = unrealized_pnl_avg  # Average cost (use final - current position)
                    total_trades += trade_count
                    total_fees += fees
                    
                    print(f"\n[Episode End] Env {env_id} | "
                          f"Steps {int(episode_steps[env_id]):5d} | "
                          f"MM.Rew {float(episode_mm_total[env_id]):7.2f} | "
                          f"Inv.Rew {float(episode_inv_total[env_id]):7.2f} | "
                          f"R.PnL ${realized_pnl:7.2f} | "
                          f"SprdCap ${spread_capture:6.2f} | "
                          f"U.PnL ${unrealized_pnl:7.2f} | "
                          f"Net ${realized_pnl + unrealized_pnl:7.2f} | "
                          f"Trades {trade_count:4d} | "
                          f"Pos {net_amount_btc:+.5f} BTC")
                    print("-"*80)
                    
                    # Reset per-env tracking for the NEW episode
                    episode_steps[env_id] = 0
                    episode_mm_total[env_id] = 0
                    episode_inv_total[env_id] = 0
                    policy.reset_env(env_id)
            
                obs = next_obs

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")

    finally:
        # Get final state metrics if evaluation ended mid-episode
        if total_steps > 0 and last_env_info is not None:
            # Get current state from last env_info
            final_realized_pnl = _extract_info_value(last_env_info, 'realized_pnl', 0, 0.0)  # LIFO
            final_unrealized_pnl = _extract_info_value(last_env_info, 'lifo_unrealized_pnl', 0, 0.0)  # LIFO
            final_trade_count = int(_extract_info_value(last_env_info, 'trade_count', 0, 0.0))
            final_fees = _extract_info_value(last_env_info, 'fees', 0, 0.0)
            
            # Calculate average cost-based PnL for comparison
            final_balance = _extract_info_value(last_env_info, 'balance', 0, 0.0)
            final_initial_balance = _extract_info_value(last_env_info, 'initial_balance', 0, 0.0)
            final_unrealized_pnl_avg = _extract_info_value(last_env_info, 'unrealized_pnl', 0, 0.0)  # Average cost
            final_realized_pnl_avg = final_balance - final_initial_balance  # Average cost-based
            
            # If we haven't completed any episodes, use current state
            if total_trades == 0:
                total_realized_pnl = final_realized_pnl  # LIFO
                total_unrealized_pnl = final_unrealized_pnl  # LIFO
                total_realized_pnl_avg = final_realized_pnl_avg  # Average cost
                total_unrealized_pnl_avg = final_unrealized_pnl_avg  # Average cost
                total_trades = final_trade_count
                total_fees = final_fees
        
        net_pnl = total_realized_pnl + total_unrealized_pnl  # LIFO
        net_pnl_avg = total_realized_pnl_avg + total_unrealized_pnl_avg  # Average cost
        
        print("\n" + "="*80)
        print("Evaluation Summary")
        print("="*80)
        print(f"Total steps:        {total_steps}")
        print(f"Net PnL (LIFO):     ${net_pnl:10.4f} (R: ${total_realized_pnl:10.4f} + U: ${total_unrealized_pnl:10.4f})")
        print(f"Net PnL (Avg Cost): ${net_pnl_avg:10.4f} (R: ${total_realized_pnl_avg:10.4f} + U: ${total_unrealized_pnl_avg:10.4f})")
        print(f"Total trades:       {total_trades}")
        print(f"Total fees:         ${total_fees:10.4f}")
        print(f"Max leverage:       {max_leverage:.2f}x")
        if last_target_inventory is not None:
            print(f"Final target inv:   {last_target_inventory:.6f}")
            print(f"Target changes:     {target_change_count} times")
            if target_change_count == 0:
                print("  WARNING: Target inventory never changed during evaluation!")
        print("="*80)
        env.close()


if __name__ == "__main__":
    main()
