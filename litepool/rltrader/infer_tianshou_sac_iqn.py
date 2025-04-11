import numpy as np
import torch
from pathlib import Path
import litepool
from tianshou.data import VectorReplayBuffer
from vec_normalizer import VecNormalize
from recurrent_actor import RecurrentActor
from iqn_critic import IQNCritic
from custom_sac_policy import CustomSACPolicy
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, env_action_space):
    """Load the trained SAC-IQN model from checkpoint"""
    # Initialize model components
    actor = RecurrentActor(device).to(device)
    critic1 = IQNCritic(action_dim=3, num_quantiles=32, hidden_dim=128, 
                       quantile_embedding_dim=32, gru_hidden_dim=128, num_layers=2).to(device)
    critic2 = IQNCritic(action_dim=3, num_quantiles=32, hidden_dim=128, 
                       quantile_embedding_dim=32, gru_hidden_dim=128, num_layers=2).to(device)

    # Initialize policy
    policy = CustomSACPolicy(
        device=device,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic1_optim=torch.optim.Adam(critic1.parameters(), lr=3e-4),
        critic2_optim=torch.optim.Adam(critic2.parameters(), lr=3e-4),
        tau=0.01, gamma=0.99, init_alpha=5.0,
        action_space=env_action_space
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Set to evaluation mode
    policy.eval()
    
    return policy

def run_inference(policy, env):
    """Run inference on the environment"""
    policy.eval()
    
    obs, info = env.reset()
    done = False
    hidden_state = None
    step = 0    
    
    while not done:
        # Get action from policy
        with torch.no_grad():
            if hidden_state is None:
                action, hidden_state = policy(obs)
            else:
                action, hidden_state = policy(obs, hidden_state=hidden_state)
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
   
        if step % 10 == 0:
           print(info)

        step += 1
        

if __name__ == "__main__":
    # Environment setup (single environment for inference)
    env = litepool.make(
        "RlTrader-v0", 
        env_type="gymnasium", 
        num_envs=1, 
        batch_size=1,
        num_threads=1, 
        is_prod=True, 
        is_inverse_instr=True, 
        api_key="",  # Add your API key if needed
        api_secret="",  # Add your API secret if needed
        symbol="BTC-PERPETUAL", 
        tick_size=0.5, 
        min_amount=10,
        maker_fee=-0.0001, 
        taker_fee=0.0005, 
        foldername="./inference_files/",
        balance=1.0, 
        start=3601 * 10, 
        max=7200 * 10
    )
    
    # Normalize the environment
    env = VecNormalize(
        env,
        device=device,
        num_envs=1,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99
    )
    
    # Load the trained model
    model_path = Path("results/final_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    policy = load_model(model_path, env.action_space)
    
    # Run inference
    print("Starting inference...")
    run_inference(policy, env, num_episodes=3)
