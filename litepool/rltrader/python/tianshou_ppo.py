# Copyright 2024 Alphaqraft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple PPO training for GLFT market making.
- 30-signal observations (13 market + 4 AMM flow + 8 trade + 5 agent state)
- 4-action outputs: 3 continuous quote parameters + 1 binary requote decision
  * Actions 0-2: bid_spread, ask_spread, target_inventory (continuous)
  * Action 3: should_requote (binary decision: >0 = requote, <=0 = continue)
- Order size uses min_size_pct by default (no RL control)
- Separate policy heads: Normal distribution for quotes, Bernoulli for requote
- Target inventory is smoothed with EMA to prevent flickering
- LSTM for temporal pattern recognition
"""
import numpy as np
import torch
from pathlib import Path
import gc
import litepool

from simple_actor_critic import SimpleActorCritic
from simple_ppo_policy import SimplePPOPolicy
from simple_collector import SimpleCollector
# Removed: HER-style goal manager not needed with dense rewards
from metric_logger import MetricLogger

# === Device setup (CPU for your machine) ===
device = torch.device("cpu")
print(f"Using device: {device}")

# === Configuration ===
NUM_ENVS = 6           # Match number of training data files
NUM_THREADS = 6        # Leave cores for main process (10 cores total)
N_STEPS = 2048         # Steps per rollout
UPDATE_EPOCHS = 3      # PPO epochs per update
MINIBATCH_SIZE = 128   # Minibatch size for updates
TOTAL_EPOCHS = 10000   # Total training epochs
LEARNING_RATE = 1e-4   # Reduced from 1e-4 to prevent gradient explosion
GAMMA = 0.995    
GAE_LAMBDA = 0.95
BASE_SPREAD_BPS = 1.0  # Base spread in basis points (1 bps = $10 on $100k BTC - room for spread capture)
MIN_SIZE_PCT = 1.0      # 1% per level × 5 levels = 5% total per side (ladder quoting)
MAX_SIZE_PCT = 5.0     # Same as MIN - fixed size, no RL control

# === Environment setup ===
print("Creating environment...")
env = litepool.make(
    "RlTrader-v0",
    env_type="gymnasium",
    num_envs=NUM_ENVS,
    batch_size=NUM_ENVS,
    num_threads=NUM_THREADS,
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
    foldername="/home/pravin/dev/alphaqraft_litepool/data/training/",
    balance=100000.0,  # Starting capital: $100,000 USD
                       # With BTC ~$100k, 2% of $100k = $2,000 = ~0.02 BTC per order
    start=360000,
    max_episode_steps=2048,  # 
    base_spread_bps=BASE_SPREAD_BPS,  # Base spread in basis points
    min_size_pct=MIN_SIZE_PCT,        # Minimum order size as % of balance
    max_size_pct=MAX_SIZE_PCT,        # Maximum order size as % of balance
)
env.spec.id = "RlTrader-v0"

# All observation signals are already bounded to [-1, 1]:
# - Market signals (13): all use tanh or are bounded by construction
# - AMM signals (3): all clamped or bounded to [-1, 1]
# No normalization needed!

# === Model setup ===
print("Creating model...")
torch.manual_seed(42)
np.random.seed(42)

# Model with separate heads:
# - Continuous head (Normal): 3 quote parameters (bid_spread, ask_spread, target_inventory)
# - Binary head (Bernoulli): 1 requote decision
# - 3 hidden layers with LayerNorm for stable training
model = SimpleActorCritic(
    obs_dim=30,  # 13 market + 4 AMM flow + 8 trade + 5 agent state
    action_dim=4,  # 3 quote params (bid_spread, ask_spread, target_inventory) + 1 requote decision
    hidden_dim=128,  # MLP hidden dimension
    lstm_hidden=64,  # LSTM hidden dimension for temporal patterns
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === Policy and Collector ===
policy = SimplePPOPolicy(
    model=model,
    lr=LEARNING_RATE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_eps=0.2,
    vf_coef=0.1,  # Reduced to prevent value function from exploding with large returns
    ent_coef=0.05,  # Higher entropy for more exploration (prevent policy collapse)
    max_grad_norm=0.5,  # Standard value for PPO
)

REWARD_SCALE = 1.0  # NO SCALING - let returns be naturally small
                    # C++ already scales by 10000, rewards are normalized by balance
                    # Returns will be small ([-3.5, 0.5]) which is CORRECT for the problem scale
                    # Value function can learn to predict small returns efficiently
                    # Scaling rewards was the wrong approach - it caused returns to explode
                    # and made training slow/unstable

collector = SimpleCollector(
    env=env,
    policy=policy,
    n_steps=N_STEPS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    reward_scale=REWARD_SCALE,
)

# Removed: goal_mgr (HER-style reward shaping not needed with dense rewards)

# === Metrics logger ===
metric_logger = MetricLogger(print_interval=N_STEPS * NUM_ENVS)  # Log every rollout

# === Directory setup ===
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
checkpoint_dir = results_dir / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True, parents=True)


def save_checkpoint(epoch, global_step):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
    }, checkpoint_dir / "checkpoint.pth")
    # Note: VecNormalize removed - all signals are already bounded to [-1, 1] in C++


def load_checkpoint():
    ckpt_path = checkpoint_dir / "checkpoint.pth"
    if not ckpt_path.exists():
        return None
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Note: VecNormalize removed - all signals are already bounded to [-1, 1] in C++
    return checkpoint['epoch'], checkpoint['global_step']


def train():
    """Main training loop."""
    # Try to resume
    resume = load_checkpoint()
    if resume:
        start_epoch, global_step = resume
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}, step {global_step}")
    else:
        start_epoch = 0
        global_step = 0
        print("Starting fresh training")
    
    best_reward = float('-inf')
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # === Collect rollout ===
        batch = collector.collect()
        
        # === PPO Update ===
        batch_size = batch['obs'].shape[0]
        indices = np.arange(batch_size)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_indices = indices[start:end]
                
                minibatch = {
                    'obs': batch['obs'][mb_indices],
                    'actions': batch['actions'][mb_indices],
                    'log_probs': batch['log_probs'][mb_indices],
                    'values': batch['values'][mb_indices],
                    'advantages': batch['advantages'][mb_indices],
                    'returns': batch['returns'][mb_indices],
                }
                
                loss_info = policy.learn(minibatch)
                
                total_loss += loss_info['loss']
                total_policy_loss += loss_info['policy_loss']
                total_value_loss += loss_info['value_loss']
                total_entropy += loss_info['entropy']
                n_updates += 1
        
        global_step += N_STEPS * NUM_ENVS
        
        # === Logging ===
        # Use average of completed episode rewards (cumulative episode totals)
        # This matches the episode-level rewards shown in logs
        completed_episode_rewards = batch.get('completed_episode_rewards', [])
        if completed_episode_rewards:
            # Average of cumulative episode rewards (sum of all steps per episode)
            avg_reward = float(np.mean(completed_episode_rewards))
        else:
            # Fallback: estimate from step rewards (sum per environment, then average)
            # This approximates cumulative episode rewards when no episodes completed
            step_rewards = batch['rewards']  # [n_steps, n_envs] - already scaled by 10000
            # Sum rewards per environment to get approximate episode totals
            env_totals = step_rewards.sum(dim=0)  # [n_envs] - cumulative per environment
            avg_reward = float(env_totals.mean().item())
        
        # Diagnostic: Check requote decisions and PnL
        actions_np = batch['actions'].detach().cpu().numpy()
        requote_actions = actions_np[:, 3]  # Last action (index 3) is requote decision
        requote_rate = (requote_actions > 0).mean()  # Fraction of steps that requoted
        
        # Check quote parameter statistics (only when requoting)
        requote_mask = actions_np[:, 3] > 0
        if requote_mask.sum() > 0:
            quote_actions = actions_np[requote_mask, :3]  # First 3 actions: bid_spread, ask_spread, target_inventory
            avg_bid_spread = quote_actions[:, 0].mean()
            avg_ask_spread = quote_actions[:, 1].mean()
            avg_target_inv = quote_actions[:, 2].mean()
        else:
            avg_bid_spread = 0.0
            avg_ask_spread = 0.0
            avg_target_inv = 0.0
        
        # Extract info for diagnostics
        # Use completed episode statistics for PnL (matches episode-level logs)
        # Fall back to last step info for trade counts and other metrics
        completed_realized_pnl = batch.get('completed_episode_realized_pnl', [])
        completed_unrealized_pnl = batch.get('completed_episode_unrealized_pnl', [])
        
        if completed_realized_pnl and len(completed_realized_pnl) > 0:
            # Use average from completed episodes (matches episode-level statistics)
            avg_realized_pnl = float(np.mean(completed_realized_pnl))
            avg_unrealized_pnl = float(np.mean(completed_unrealized_pnl))
        else:
            # Fallback: use last step info if no episodes completed
            avg_realized_pnl = 0.0
            avg_unrealized_pnl = 0.0
            if batch['infos'] and len(batch['infos']) > 0:
                last_info = batch['infos'][-1]
                def safe_get_avg(info, key, default=0.0):
                    values = []
                    if isinstance(info, list):
                        for env_info in info:
                            if isinstance(env_info, dict):
                                val = env_info.get(key, default)
                                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                                    values.extend([float(v) for v in val])
                                elif val is not None:
                                    values.append(float(val))
                    elif isinstance(info, dict):
                        val = info.get(key, default)
                        if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                            values = [float(v) for v in val]
                        elif val is not None:
                            values = [float(val)]
                    return np.mean(values) if values else default
                avg_realized_pnl = safe_get_avg(last_info, 'realized_pnl', 0.0)
                avg_unrealized_pnl = safe_get_avg(last_info, 'unrealized_pnl', 0.0)
        
        avg_trade_count = 0.0
        buy_delta = 0.0
        sell_delta = 0.0
        actual_placed_bid_spread_bps = 0.0
        actual_placed_ask_spread_bps = 0.0
        if batch['infos'] and len(batch['infos']) > 0:
            # Get first and last step's info to calculate deltas for trade counts
            first_info = batch['infos'][0]
            last_info = batch['infos'][-1]
            
            # Helper to extract values for delta calculation (sum across all envs)
            def safe_get_sum(info, key, default=0.0):
                values = []
                if isinstance(info, list):
                    # List of environment info dicts
                    for env_info in info:
                        if isinstance(env_info, dict):
                            val = env_info.get(key, default)
                            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                                values.extend([float(v) for v in val])
                            elif val is not None:
                                values.append(float(val))
                elif isinstance(info, dict):
                    val = info.get(key, default)
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        values = [float(v) for v in val]
                    elif val is not None:
                        values = [float(val)]
                
                return np.sum(values) if values else default
            
            # Get buy/sell counts to diagnose inventory accumulation
            def get_counts(info_list, key):
                """Extract count per environment for a given key."""
                counts = []
                if isinstance(info_list, list):
                    for env_info in info_list:
                        if isinstance(env_info, dict):
                            val = env_info.get(key, 0.0)
                            if isinstance(val, (list, np.ndarray)):
                                counts.extend([float(v) for v in val])
                            else:
                                counts.append(float(val))
                elif isinstance(info_list, dict):
                    val = info_list.get(key, 0.0)
                    if isinstance(val, (list, np.ndarray)):
                        counts = [float(v) for v in val]
                    else:
                        counts = [float(val)]
                return counts if counts else [0.0]
            
            # Handle resets for buy/sell counts (same logic as trade_count)
            buy_delta = 0.0
            sell_delta = 0.0
            first_buys = get_counts(first_info, 'buy_trades')
            last_buys = get_counts(last_info, 'buy_trades')
            first_sells = get_counts(first_info, 'sell_trades')
            last_sells = get_counts(last_info, 'sell_trades')
            
            n_envs = max(len(first_buys), len(last_buys))
            for i in range(n_envs):
                first_buy = first_buys[i] if i < len(first_buys) else 0.0
                last_buy = last_buys[i] if i < len(last_buys) else 0.0
                buy_delta += last_buy if last_buy < first_buy else (last_buy - first_buy)
                
                first_sell = first_sells[i] if i < len(first_sells) else 0.0
                last_sell = last_sells[i] if i < len(last_sells) else 0.0
                sell_delta += last_sell if last_sell < first_sell else (last_sell - first_sell)
            
            # Calculate trade count delta (trades in this batch/epoch)
            # trade_count is cumulative, but resets to 0 when episode ends
            # Need to handle per-environment to avoid negative deltas from resets
            def get_trade_counts(info_list):
                """Extract trade_count per environment, handling arrays and lists."""
                counts = []
                if isinstance(info_list, list):
                    for env_info in info_list:
                        if isinstance(env_info, dict):
                            val = env_info.get('trade_count', 0.0)
                            if isinstance(val, (list, np.ndarray)):
                                counts.extend([float(v) for v in val])
                            else:
                                counts.append(float(val))
                elif isinstance(info_list, dict):
                    val = info_list.get('trade_count', 0.0)
                    if isinstance(val, (list, np.ndarray)):
                        counts = [float(v) for v in val]
                    else:
                        counts = [float(val)]
                return counts if counts else [0.0]
            
            first_counts = get_trade_counts(first_info)
            last_counts = get_trade_counts(last_info)
            
            # Calculate delta per environment, handling resets (when count decreases)
            # If last < first, it means a reset happened - count only the new trades
            total_delta = 0.0
            n_envs = max(len(first_counts), len(last_counts))
            for i in range(n_envs):
                first_val = first_counts[i] if i < len(first_counts) else 0.0
                last_val = last_counts[i] if i < len(last_counts) else 0.0
                
                if last_val >= first_val:
                    # Normal case: no reset, or reset happened but we're counting new trades
                    total_delta += (last_val - first_val)
                else:
                    # Reset happened: last_val is from new episode, just count it
                    total_delta += last_val
            
            avg_trade_count = total_delta
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), results_dir / "best_model.pth")
        
        # Calculate actual quote parameters (for diagnostics)
        # bid_spread/ask_spread: [-1, 1] -> EXPONENTIAL mapping to [MIN_SPREAD_MULT, MAX_SPREAD_MULT]
        # Must match strategy.cc: center_mult * exp(action * log_ratio)
        # size: fixed at MIN_SIZE_PCT (no RL control)
        MIN_SPREAD_MULT = 0.5  # Must match strategy.h (0.5x base = 1.5bps floor)
        MAX_SPREAD_MULT = 3.0  # Must match strategy.h
        LOG_RATIO = np.log(MAX_SPREAD_MULT / MIN_SPREAD_MULT) / 2.0  # ~1.35
        CENTER_MULT = np.sqrt(MAX_SPREAD_MULT * MIN_SPREAD_MULT)     # ~0.77
        
        if requote_mask.sum() > 0:
            quote_actions = actions_np[requote_mask, :3]  # bid_spread, ask_spread, target_inventory
            # Bid spread: exponential mapping to match C++ calculation
            bid_spread_actions = quote_actions[:, 0]
            bid_spread_mult = CENTER_MULT * np.exp(bid_spread_actions * LOG_RATIO)
            actual_bid_spread_bps = BASE_SPREAD_BPS * bid_spread_mult.mean()
            
            # Ask spread: exponential mapping
            ask_spread_actions = quote_actions[:, 1]
            ask_spread_mult = CENTER_MULT * np.exp(ask_spread_actions * LOG_RATIO)
            actual_ask_spread_bps = BASE_SPREAD_BPS * ask_spread_mult.mean()
            
            # Average spread for logging (show bid/ask separately to diagnose)
            actual_spread_bps = (actual_bid_spread_bps + actual_ask_spread_bps) / 2.0
            size_pct = MIN_SIZE_PCT  # Fixed at minimum size
        else:
            actual_bid_spread_bps = BASE_SPREAD_BPS * CENTER_MULT
            actual_ask_spread_bps = BASE_SPREAD_BPS * CENTER_MULT
            actual_spread_bps = BASE_SPREAD_BPS * CENTER_MULT
            size_pct = MIN_SIZE_PCT
        
        # Calculate ACTUAL effective spreads from placed prices (accounts for skew and clamping)
        # This shows what was actually placed, not just what the action suggested
        actual_placed_bid_spread_bps = 0.0
        actual_placed_ask_spread_bps = 0.0
        if batch['infos'] and len(batch['infos']) > 0:
            last_info = batch['infos'][-1]
            mid_prices = []
            bid_prices = []
            ask_prices = []
            
            def extract_prices(info_list, key):
                prices = []
                if isinstance(info_list, list):
                    for env_info in info_list:
                        if isinstance(env_info, dict):
                            val = env_info.get(key, 0.0)
                            if isinstance(val, (list, np.ndarray)):
                                prices.extend([float(v) for v in val])
                            else:
                                prices.append(float(val))
                elif isinstance(info_list, dict):
                    val = info_list.get(key, 0.0)
                    if isinstance(val, (list, np.ndarray)):
                        prices = [float(v) for v in val]
                    else:
                        prices = [float(val)]
                return prices
            
            mid_prices = extract_prices(last_info, 'last_mid_price')
            bid_prices = extract_prices(last_info, 'last_bid_price')
            ask_prices = extract_prices(last_info, 'last_ask_price')
            
            # Calculate effective FULL spreads from actual placed prices
            # bid_spread = distance from mid to bid (full spread, not half)
            # ask_spread = distance from mid to ask (full spread, not half)
            valid_spreads = []
            for i in range(min(len(mid_prices), len(bid_prices), len(ask_prices))):
                mid = mid_prices[i]
                bid = bid_prices[i]
                ask = ask_prices[i]
                if mid > 0 and bid > 0 and ask > 0:
                    # Full spread: distance from mid to quote (in bps)
                    bid_spread_bps = ((mid - bid) / mid) * 10000.0  # Full spread from mid to bid
                    ask_spread_bps = ((ask - mid) / mid) * 10000.0  # Full spread from mid to ask
                    valid_spreads.append((bid_spread_bps, ask_spread_bps))
            
            if valid_spreads:
                actual_placed_bid_spread_bps = np.mean([s[0] for s in valid_spreads])
                actual_placed_ask_spread_bps = np.mean([s[1] for s in valid_spreads])
        
        print(f"Epoch {epoch:5d} | "
              f"Step {global_step:8d} | "
              f"Reward {avg_reward:8.4f} | "
              f"ReqRate {requote_rate:.2%} | "
              f"R.PnL {avg_realized_pnl:8.4f} | "
              f"U.PnL {avg_unrealized_pnl:8.4f} | "
              f"Trades {avg_trade_count:6.0f} (B:{buy_delta:.0f}/S:{sell_delta:.0f}) | "
              f"BidSprd {actual_bid_spread_bps:4.2f}bps AskSprd {actual_ask_spread_bps:4.2f}bps | "
              f"Actual: Bid {actual_placed_bid_spread_bps:4.2f}bps Ask {actual_placed_ask_spread_bps:4.2f}bps | "
              f"Size {size_pct:4.2f}% | "
              f"Loss {total_loss/n_updates:7.4f} | "
              f"PL {total_policy_loss/n_updates:7.4f} | "
              f"VL {total_value_loss/n_updates:7.4f} | "
              f"Ent {total_entropy/n_updates:6.4f} | "
              f"Std {loss_info['action_std']:5.3f}", flush=True)
        
        # Print episode statistics if any episodes completed in this epoch
        if completed_episode_rewards:
            print(f"\n  Completed {len(completed_episode_rewards)} episode(s) in this epoch\n", flush=True)
        
        # === Log detailed metrics ===
        if epoch % 10 == 0 or epoch == 0:
            # Prepare data for MetricLogger
            # MetricLogger expects: step_count, infos dict with 'infos' key, rewards array, policy
            # Our batch has: 'infos' as list of dicts (one per step), 'rewards' as tensor [n_steps, n_envs]
            rewards_array = batch['rewards']
            if isinstance(rewards_array, torch.Tensor):
                rewards_array = rewards_array.detach().cpu().numpy()
            
            # Convert infos list to dict format expected by MetricLogger
            # MetricLogger expects infos['infos'] to be a dict with arrays per metric, indexed by env_id
            if batch.get('infos') and len(batch['infos']) > 0:
                # Get last step's info from all environments
                last_infos = batch['infos'][-1]  # Last step, should be list of dicts (one per env)
                if isinstance(last_infos, list) and len(last_infos) > 0:
                    # Convert list of dicts to dict of arrays
                    # Each element in last_infos is a dict for one environment
                    def safe_get(inf, key, default=0.0):
                        val = inf.get(key, default)
                        if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                            return float(val[0])
                        return float(val) if val is not None else default
                    
                    info_dict = {
                        'realized_pnl': np.array([safe_get(inf, 'realized_pnl', 0.0) for inf in last_infos]),
                        'unrealized_pnl': np.array([safe_get(inf, 'unrealized_pnl', 0.0) for inf in last_infos]),
                        'fees': np.array([safe_get(inf, 'fees', 0.0) for inf in last_infos]),
                        'trade_count': np.array([safe_get(inf, 'trade_count', 0.0) for inf in last_infos]),
                        'drawdown': np.array([safe_get(inf, 'drawdown', 0.0) for inf in last_infos]),
                        'leverage': np.array([safe_get(inf, 'leverage', 0.0) for inf in last_infos]),
                    }
                    metric_logger.log(global_step, {'infos': info_dict}, rewards_array, policy)
        
        # CRITICAL: Clear infos from batch IMMEDIATELY after logging to prevent memory accumulation
        # infos_list contains 2048 dictionaries with numpy arrays - this is HUGE and causes hangs
        if 'infos' in batch:
            batch['infos'] = None
        
        # Force garbage collection to help with memory management
        # PyTorch's allocator caches memory, but Python GC can help free unreferenced objects
        gc.collect()
        
        # === Save checkpoint ===
        if epoch % 10 == 0:
            save_checkpoint(epoch, global_step)
    
    # Final save
    torch.save(model.state_dict(), results_dir / "final_model.pth")
    # Note: VecNormalize removed - all signals are already bounded to [-1, 1] in C++
    print(f"Training complete! Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Training for GLFT Market Making")
    print("=" * 60)
    print(f"Environments: {NUM_ENVS}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Steps per rollout: {N_STEPS}")
    print(f"Observations: 30 signals (13 market + 4 AMM flow + 8 trade + 5 agent state)")
    print(f"Actions: 4 (3 continuous quote params + 1 binary requote decision)")
    print(f"  Quote params: bid_spread, ask_spread, target_inventory (continuous)")
    print(f"  Order size: fixed at min_size_pct ({MIN_SIZE_PCT}%) - no RL control")
    print(f"  Requote: binary decision (>0 = requote, <=0 = continue)")
    print(f"  Note: target_inventory is smoothed with EMA (alpha=0.05) to prevent flickering")
    print(f"Strategy config: base_spread_bps={BASE_SPREAD_BPS} bps, min_size={MIN_SIZE_PCT}%, max_size={MAX_SIZE_PCT}%")
    print("=" * 60)
    train()
