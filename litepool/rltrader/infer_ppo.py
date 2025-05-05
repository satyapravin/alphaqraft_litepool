import torch
import numpy as np
from pathlib import Path
import litepool
from vec_normalizer import VecNormalize
from recurrent_actor_critic import RecurrentActorCritic
from recurrent_ppo_policy import RecurrentPPOPolicy
from metric_logger import MetricLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_envs = 1  # Single environment for inference

# === Environment setup for out-of-sample testing ===
env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=num_of_envs, batch_size=num_of_envs,
    num_threads=1, is_prod=False, is_inverse_instr=True, api_key="", api_secret="",
    symbol="BTC-PERPETUAL", hedge_symbol='BTC-18APR25', tick_size=0.1, min_amount=10,
    maker_fee=-0.00005, taker_fee=0.0005, foldername="./test_files/",
    balance=1.0, start=1, max=36000 * 24  # Out-of-sample range
)
env.spec.id = 'RlTrader-v0'

# === Initialize VecNormalize ===
env = VecNormalize(
    env,
    device=device,
    num_envs=num_of_envs,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.,
    clip_reward=10.,
    gamma=0.999
)

# === Load VecNormalize state ===
results_dir = Path("results")
vec_norm_path = results_dir / "vecnorm.pth"
env.load(vec_norm_path)  # Load the saved normalization state
#env.set_training(False)
#env.eval()

# === Model initialization ===
model = RecurrentActorCritic(
    action_dim=env.action_space.shape[0],
    hidden_dim=128,
    gru_hidden_dim=128,
    num_layers=2
).to(device)

# === Load trained model weights ===
model_path = results_dir / "final_model_inference.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.train()  # Keep Bayesian layers in sampling mode

# === Policy ===
policy = RecurrentPPOPolicy(model=model)
#policy.eval()

# === Metric Logger ===
metric_logger = MetricLogger(print_interval=512)

# === Inference Loop ===
obs, info = env.reset()
hidden_state = policy.init_hidden_state(batch_size=num_of_envs)
hidden_state = policy._to_device(hidden_state)

step_count = 0
cumulative_reward = 0.0
episode_rewards = []
episode_infos = []

print(f"\n=== Starting Out-of-Sample Bayesian Inference ===\n")

# === Helper for Bayesian sampling ===
@torch.no_grad()
def sample_bayesian_action(model, obs_tensor, hidden_state, n_samples=20):
    model.train()  # Ensure stochastic sampling
    samples = []
    for _ in range(n_samples):
        dist, _, _ = model(obs_tensor, hidden_state)
        action = dist.sample()
        samples.append(action)
    all_samples = torch.stack(samples)  # [n_samples, batch, action_dim]
    mean_action = all_samples.mean(dim=0)
    std_action = all_samples.std(dim=0)
    return mean_action, std_action

# === Main loop ===
while step_count < 36000 * 2:  # Limit to out-of-sample range
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

    # === Bayesian Action Inference ===
    mean_action, std_action = sample_bayesian_action(model, obs_tensor, hidden_state, n_samples=20)
    action_np = mean_action.cpu().numpy()

    next_obs, reward, terminated, truncated, info = env.step(action_np)

    # Accumulate rewards and infos
    reward_value = reward.sum().item()
    cumulative_reward += reward_value
    episode_rewards.append(reward_value)
    episode_infos.append(info)

    # Handle reset
    if np.logical_or(terminated, truncated).any():
        obs, info = env.reset()
        hidden_state = policy.init_hidden_state(batch_size=num_of_envs)
        hidden_state = policy._to_device(hidden_state)
        print(f"\n[INFO] Environment reset at step {step_count}.")

        # Log episode metrics
        metric_logger.log(step_count, {
            "infos": episode_infos,
        }, np.array(episode_rewards), policy)
        episode_rewards = []
        episode_infos = []

    else:
        obs = next_obs

    step_count += 1

    # === Get uncertainty values ===
    std_val = std_action.cpu().numpy()[0] if std_action is not None else np.zeros_like(action_np[0])

    # === Log step metrics ===
    if isinstance(info, list) and len(info) > 0:
        env_info = info[0]
        realized_pnl = env_info.get('realized_pnl', [0.0])[0]
        unrealized_pnl = env_info.get('unrealized_pnl', [0.0])[0]
        leverage = env_info.get('leverage', [0.0])[0]
        print(f"Step {step_count:06d} | "
              f"Reward: {reward_value:+.6f} | "
              f"Cumulative Reward: {cumulative_reward:+.6f} | "
              f"Realized PnL: {realized_pnl:+.6f} | "
              f"Unrealized PnL: {unrealized_pnl:+.6f} | "
              f"Leverage: {leverage:.2f}x | "
              f"Action Std: {std_val}")

# === Final logging ===
if episode_rewards:
    metric_logger.log(step_count, {
        "infos": episode_infos,
    }, np.array(episode_rewards), policy)

print(f"\n=== Out-of-Sample Testing Complete | Total Steps: {step_count} | Cumulative Reward: {cumulative_reward:.6f} ===\n")
