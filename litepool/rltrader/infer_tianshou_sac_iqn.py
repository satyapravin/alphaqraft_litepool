import torch
import numpy as np
from pathlib import Path
from tianshou.data import Batch
from litepool import make
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_dir = Path("results")
model_path = results_dir / "final_model.pth"

env = litepool.make("RlTrader-v0", env_type="gymnasium",
                          num_envs=1, batch_size=1,
                          num_threads=1,
                          is_prod=True,
                          is_inverse_instr=True,
                          api_key="1VOwN5_G",
                          api_secret="GE1U3j-05bHT-zRZ3ZzqtSLRsEyD2_Jobf_JuCbh4l8",
                          symbol="BTC-PERPETUAL",
                          tick_size=0.5,
                          min_amount=10,
                          maker_fee=-0.0001,
                          taker_fee=0.0005,
                          foldername="./testfiles/",
                          balance=0.001,
                          start=1,
                          max=72000001,
                          depth=20)


env.spec.id = "RlTrader-v0"
env = DummyVectorEnv([lambda: env])

from tianshou.data import VectorReplayBuffer
from tianshou.policy import BasePolicy

from tianshou_sac_iqn import RecurrentActor, IQNCritic, CustomSACPolicy  # Importing from the training script

# Define actor and critic models
actor = RecurrentActor().to(device)
critic = IQNCritic().to(device)

policy = CustomSACPolicy(
    actor=actor,
    critic=critic,
    actor_optim=torch.optim.Adam(actor.parameters(), lr=1e-3),
    critic_optim=torch.optim.Adam(critic.parameters(), lr=1e-3),
    tau=0.005,
    gamma=0.997,
    alpha=2.0,
    action_space=env.action_space,
).to(device)

# Load the trained model
if model_path.exists():
    print(f"Loading trained model from {model_path}")
    saved_model = torch.load(model_path, map_location=device)
    policy.load_state_dict(saved_model["policy_state_dict"])
    policy.actor_optim.load_state_dict(saved_model["actor_optim_state_dict"])
    policy.critic_optim.load_state_dict(saved_model["critic_optim_state_dict"])
    policy.alpha_optim.load_state_dict(saved_model["alpha_optim_state_dict"])
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Inference function
def run_inference(environment, policy):
    obs, _ = environment.reset()
    done = False
    episode_reward = 0.0
    step_count = 0

    print("Starting inference...")
    while not np.any(done):
        obs_batch = Batch(obs=torch.as_tensor(obs, device=device, dtype=torch.float32))

        with torch.no_grad():
            result = policy(obs_batch)
            action = result.act.cpu().numpy()

        obs, reward, terminated, truncated, info = environment.step(action)
        
        print(f"Balance: {info['balance'][0]:.6f}, "
                          f"Realized PnL: {info['realized_pnl'][0]:.6f}, "
                          f"Unrealized PnL: {info['unrealized_pnl'][0]:.6f}, "
                          f"Fees: {info['fees'][0]:.6f}, "
                          f"Trade Count: {info['trade_count'][0]}, "
                          f"Drawdown: {info['drawdown'][0]:.6f}, "
                          f"Leverage: {info['leverage'][0]:.4f}")
        done = terminated or truncated
        step_count += 1


run_inference(env, policy)
