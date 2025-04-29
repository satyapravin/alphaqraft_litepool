import torch
import numpy as np
from pathlib import Path
import litepool

from vec_normalizer import VecNormalize
from recurrent_actor_critic import RecurrentActorCritic
from recurrent_ppo_policy import RecurrentPPOPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_envs = 1  # Production = 1 env

# === Environment setup ===
env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=num_of_envs, batch_size=num_of_envs,
    num_threads=1, is_prod=True, is_inverse_instr=True, api_key="", api_secret="",
    symbol="BTC-PERPETUAL", hedge_symbol='BTC-18APR25', tick_size=0.5, min_amount=10,
    maker_fee=-0.0001, taker_fee=0.0005, foldername="./prod_files/",
    balance=1.0, start=0, max=int(1e9)
)
env.spec.id = 'RlTrader-v0'

# === Load VecNormalize ===
vec_norm_path = Path("results/vecnorm.pth")
env = VecNormalize.load(vec_norm_path, env)
env.set_training(False)
env.eval()

# === Model initialization ===
model = RecurrentActorCritic(
    action_dim=env.action_space.shape[0],
    hidden_dim=128,
    gru_hidden_dim=128,
    num_layers=2
).to(device)

# === Load trained model weights ===
model_path = Path("results/final_model_inference.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.train()  # Important: keep Bayesian layers in sampling mode

# === Policy ===
policy = RecurrentPPOPolicy(model=model)
policy.eval()

# === Inference Loop ===
obs, info = env.reset()
hidden_state = policy.init_hidden_state(batch_size=num_of_envs)
hidden_state = policy._to_device(hidden_state)

step_count = 0
cumulative_reward = 0.0

print(f"\n=== Starting Continuous Bayesian Inference ===\n")

# === Helper for Bayesian sampling ===
@torch.no_grad()
def sample_bayesian_action(model, obs_tensor, hidden_state, n_samples=20):
    model.train()  # Ensure stochastic sampling is active (BayesianLinear)
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
while True:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

    # === Bayesian Action Inference ===
    mean_action, std_action = sample_bayesian_action(model, obs_tensor, hidden_state, n_samples=20)
    action_np = mean_action.cpu().numpy()

    next_obs, reward, terminated, truncated, info = env.step(action_np)

    # Handle reset
    if np.logical_or(terminated, truncated).any():
        obs, info = env.reset()
        hidden_state = policy.init_hidden_state(batch_size=num_of_envs)
        hidden_state = policy._to_device(hidden_state)
        print("\n[INFO] Environment automatically reset.")
    else:
        obs = next_obs

    reward_value = reward.sum().item()
    cumulative_reward += reward_value
    step_count += 1

    # === Get uncertainty values ===
    std_val = std_action.cpu().numpy()[0] if std_action is not None else np.zeros_like(action_np[0])

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
