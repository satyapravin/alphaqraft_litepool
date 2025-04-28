import torch
import numpy as np
from pathlib import Path
import litepool

from vec_normalizer import VecNormalize
from recurrent_actor_critic import RecurrentActorCritic
from recurrent_ppo_policy import RecurrentPPOPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_envs = 1  # Usually 1 env for production

# === Environment setup ===
env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=num_of_envs, batch_size=num_of_envs,
    num_threads=1, is_prod=True, is_inverse_instr=True, api_key="", api_secret="",
    symbol="BTC-PERPETUAL", hedge_symbol='BTC-18APR25', tick_size=0.5, min_amount=10,
    maker_fee=-0.0001, taker_fee=0.0005, foldername="./prod_files/",
    balance=1.0, start=0, max=int(1e9)  # Large max for continuous
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
model.eval()

# === Policy ===
policy = RecurrentPPOPolicy(model=model)
policy.eval()

# === Inference Loop ===
obs, info = env.reset()
hidden_state = policy.init_hidden_state(batch_size=num_of_envs)
hidden_state = policy._to_device(hidden_state)

step_count = 0
cumulative_reward = 0.0

print(f"\n=== Starting Continuous Inference ===\n")

while True:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        action, _, _, hidden_state = policy.forward(obs_tensor, hidden_state)

    action_np = action.cpu().numpy()
    next_obs, reward, terminated, truncated, info = env.step(action_np)

    # Handle reset if needed (for safety)
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

    # === Print step info ===
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
              f"Leverage: {leverage:.2f}x")

