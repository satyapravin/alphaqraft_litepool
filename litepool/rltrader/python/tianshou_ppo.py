"""
Simple PPO training for GLFT market making.
- 13-signal observations
- 7-action outputs: 6 continuous quote parameters + 1 binary requote decision
  * Actions 0-5: bid_spread, ask_spread, bid_size, ask_size, target_inventory, skew
  * Action 6: should_requote (binary decision: >0 = requote, <=0 = continue)
- Separate policy heads: Normal distribution for quotes, Bernoulli for requote
- CPU-friendly, no recurrence, no OT
"""
import numpy as np
import torch
from pathlib import Path
import litepool

from simple_actor_critic import SimpleActorCritic
from simple_ppo_policy import SimplePPOPolicy
from simple_collector import SimpleCollector
from vec_normalizer import VecNormalize
from goal_manager import NetPnlGoalManager
from metric_logger import MetricLogger

# === Device setup (CPU for your machine) ===
device = torch.device("cpu")
print(f"Using device: {device}")

# === Configuration ===
NUM_ENVS = 6           # Match number of training data files
NUM_THREADS = 6        # Leave cores for main process (10 cores total)
N_STEPS = 4096         # Steps per rollout (increased from 1024 for more stable training)
UPDATE_EPOCHS = 10     # PPO epochs per update
MINIBATCH_SIZE = 256   # Minibatch size for updates
TOTAL_EPOCHS = 10000   # Total training epochs
LEARNING_RATE = 1e-4
GAMMA = 0.997  # ~300 step horizon for 3-5 minute trading (was 0.99 = 100 steps)
GAE_LAMBDA = 0.95

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
    maker_fee=0,
    taker_fee=0.0005,
    foldername="/home/pravin/dev/alphaqraft_litepool/data/training/",
    balance=100000.0,  # Starting capital: $100,000 USD
                       # With BTC ~$100k, 2% of $100k = $2,000 = ~0.02 BTC per order
    start=1,
    max=20480,  # Match rollout: 20480 ticks / 5 = 4096 RL steps = N_STEPS
)
env.spec.id = "RlTrader-v0"

# Observation normalization
env = VecNormalize(
    env,
    device=device,
    num_envs=NUM_ENVS,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=GAMMA,
)

# === Model setup ===
print("Creating model...")
torch.manual_seed(42)
np.random.seed(42)

# Model with separate heads:
# - Continuous head (Normal): 6 quote parameters
# - Binary head (Bernoulli): 1 requote decision
model = SimpleActorCritic(
    obs_dim=16,  # 13 market signals + 3 AMM flow signals
    action_dim=4,  # 3 quote params (spread, size, skew) + 1 requote decision
    hidden_dim=64,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === Policy and Collector ===
policy = SimplePPOPolicy(
    model=model,
    lr=LEARNING_RATE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,  # Reduced - log_std is clamped now to prevent entropy explosion
    max_grad_norm=1.0,  # Increased for stability
)

REWARD_SCALE = 1000.0  # Scale up tiny rewards (~0.0001) to meaningful magnitude

collector = SimpleCollector(
    env=env,
    policy=policy,
    n_steps=N_STEPS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    reward_scale=REWARD_SCALE,
)

goal_mgr = NetPnlGoalManager(device)

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
    # Save reward normalization stats
    env.save(results_dir / "vecnorm.pth")


def load_checkpoint():
    ckpt_path = checkpoint_dir / "checkpoint.pth"
    if not ckpt_path.exists():
        return None
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load reward normalization stats
    vecnorm_path = results_dir / "vecnorm.pth"
    if vecnorm_path.exists():
        env.load(vecnorm_path)
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
        
        # === HER-style reward shaping ===
        goal_mgr.relabel_rewards(batch)
        
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
        avg_reward = batch['rewards'].mean().item()
        
        # Diagnostic: Check requote decisions and PnL
        actions_np = batch['actions'].detach().cpu().numpy()
        requote_actions = actions_np[:, 3]  # Last action (index 3) is requote decision
        requote_rate = (requote_actions > 0).mean()  # Fraction of steps that requoted
        
        # Check quote parameter statistics (only when requoting)
        requote_mask = actions_np[:, 3] > 0
        if requote_mask.sum() > 0:
            quote_actions = actions_np[requote_mask, :3]  # First 3 actions: spread, size, skew
            avg_spread = quote_actions[:, 0].mean()
            avg_size = quote_actions[:, 1].mean()
            avg_skew = quote_actions[:, 2].mean()
        else:
            avg_spread = 0.0
            avg_size = 0.0
            avg_skew = 0.0
        
        # Extract info for diagnostics - average across all environments from last step
        avg_realized_pnl = 0.0
        avg_unrealized_pnl = 0.0
        avg_trade_count = 0.0
        if batch['infos'] and len(batch['infos']) > 0:
            # Get last step's info (should have values from all environments)
            last_info = batch['infos'][-1]
            
            # Helper to extract and average value across all environments
            def safe_get_avg(info, key, default=0.0):
                values = []
                if isinstance(info, list):
                    # List of environment info dicts
                    for env_info in info:
                        if isinstance(env_info, dict):
                            val = env_info.get(key, default)
                            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                                # Array of values (one per env) - collect all
                                values.extend([float(v) for v in val])
                            elif val is not None:
                                values.append(float(val))
                elif isinstance(info, dict):
                    # Single dict - check if values are arrays (one per env)
                    val = info.get(key, default)
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        values = [float(v) for v in val]
                    elif val is not None:
                        values = [float(val)]
                
                return np.mean(values) if values else default
            
            avg_realized_pnl = safe_get_avg(last_info, 'realized_pnl', 0.0)
            avg_unrealized_pnl = safe_get_avg(last_info, 'unrealized_pnl', 0.0)
            avg_trade_count = safe_get_avg(last_info, 'trade_count', 0.0)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), results_dir / "best_model.pth")
        
        # Calculate actual quote parameters (for diagnostics)
        # spread: [-1, 1] -> multiplier [0.5, 1.5] on base_spread_bps (0.0 bps, min 1 tick enforced)
        # size: [-1, 1] -> [min_size_pct, max_size_pct] (0.5% to 2%)
        if requote_mask.sum() > 0:
            quote_actions = actions_np[requote_mask, :3]
            # Spread: action [-1,1] -> multiplier [0.5, 1.5] -> actual spread = 0.0 bps * multiplier = 0
            # But code enforces minimum 1 tick spread, so actual will be ~1 tick
            spread_mult = 1.0 + quote_actions[:, 0].mean() * 0.5
            actual_spread_bps = 0.0 * spread_mult  # Will be clamped to 1 tick minimum in C++
            # Size: action [-1,1] -> [0.5%, 2%] of balance
            size_pct = 0.5 + (1.0 + quote_actions[:, 1].mean()) * 0.5 * (2.0 - 0.5)
        else:
            actual_spread_bps = 0.0
            size_pct = 0.0
        
        print(f"Epoch {epoch:5d} | "
              f"Step {global_step:8d} | "
              f"Reward {avg_reward:8.4f} | "
              f"ReqRate {requote_rate:.2%} | "
              f"R.PnL {avg_realized_pnl:8.4f} | "
              f"U.PnL {avg_unrealized_pnl:8.4f} | "
              f"Trades {avg_trade_count:6.0f} | "
              f"Spread {actual_spread_bps:4.2f}bps | "
              f"Size {size_pct:4.2f}% | "
              f"Loss {total_loss/n_updates:7.4f} | "
              f"PL {total_policy_loss/n_updates:7.4f} | "
              f"VL {total_value_loss/n_updates:7.4f} | "
              f"Ent {total_entropy/n_updates:6.4f} | "
              f"Std {loss_info['action_std']:5.3f}")
        
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
            if batch['infos'] and len(batch['infos']) > 0:
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
        
        # === Save checkpoint ===
        if epoch % 10 == 0:
            save_checkpoint(epoch, global_step)
    
    # Final save
    torch.save(model.state_dict(), results_dir / "final_model.pth")
    env.save(results_dir / "vecnorm.pth")
    print(f"Training complete! Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Training for GLFT Market Making")
    print("=" * 60)
    print(f"Environments: {NUM_ENVS}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Steps per rollout: {N_STEPS}")
    print(f"Observations: 13 signals (6 spread + 4 volume + 3 volatility)")
    print(f"Actions: 7 (6 continuous quote params + 1 binary requote decision)")
    print(f"  Quote params: bid_spread, ask_spread, bid_size, ask_size, target_inventory, skew")
    print(f"  Requote: binary decision (>0 = requote, <=0 = continue)")
    print("=" * 60)
    train()
