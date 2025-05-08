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

def load_model_and_env():
    """Load trained model and environment with proper error handling"""
    # Initialize environment
    env = litepool.make(
        "RlTrader-v0", env_type="gymnasium", num_envs=num_of_envs, batch_size=num_of_envs,
        num_threads=1, is_prod=False, is_inverse_instr=True, api_key="", api_secret="",
        symbol="BTC-PERPETUAL", hedge_symbol='BTC-18APR25', tick_size=0.1, min_amount=10,
        maker_fee=-0.00005, taker_fee=0.0005, foldername="./test_files/",
        balance=1.0, start=1, max=36000 * 24  # Out-of-sample range
    )
    env.spec.id = 'RlTrader-v0'

    # Initialize VecNormalize with modified reset behavior
    class SafeVecNormalize(VecNormalize):
        def reset(self, env_id=None):
            """Modified reset to avoid in-place operation conflicts"""
            if env_id is None:
                obs, info = self.env.reset()
                self.returns = torch.zeros_like(self.returns)  # Create new tensor
            else:
                env_id = np.array([env_id] if np.isscalar(env_id) else env_id)
                obs, info = self.env.reset(env_id)
                self.returns[env_id] = 0.0  # This is safe now
            
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs = self.normalize_obs(obs)
            return obs.cpu().numpy(), info

    env = SafeVecNormalize(
        env,
        device=device,
        num_envs=num_of_envs,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.999
    )

    # Load normalization stats
    results_dir = Path("results")
    vec_norm_path = results_dir / "vecnorm.pth"
    if vec_norm_path.exists():
        env.load(vec_norm_path)
    else:
        print(f"Warning: VecNormalize stats not found at {vec_norm_path}")

    # Initialize model
    model = RecurrentActorCritic(
        action_dim=env.action_space.shape[0],
        hidden_dim=128,
        gru_hidden_dim=128,
        num_layers=2
    ).to(device)

    # Load model weights
    model_path = results_dir / "final_model_inference.pth"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    return env, model

@torch.no_grad()
def sample_bayesian_action(model, obs_tensor, hidden_state, n_samples=10):
    """Sample multiple actions and return mean and uncertainty"""
    samples = []
    for _ in range(n_samples):
        dist, _, new_hidden_state = model(obs_tensor, hidden_state)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)  # Proper squashing
        samples.append(action)
    
    all_samples = torch.stack(samples)  # [n_samples, batch, action_dim]
    mean_action = all_samples.mean(dim=0)
    std_action = all_samples.std(dim=0)
    return mean_action, std_action, new_hidden_state

def extract_pnl_values(info):
    """Helper function to safely extract PnL values from info dict"""
    if isinstance(info, list):
        info = info[0]
    
    def get_value(key):
        val = info.get(key, 0.0)
        if isinstance(val, (list, np.ndarray)):
            return float(val[0])
        return float(val)
    
    return {
        'realized_pnl': get_value('realized_pnl'),
        'unrealized_pnl': get_value('unrealized_pnl'),
        'fees': get_value('fees'),
        'leverage': get_value('leverage')
    }

def main():
    # Load environment and model
    env, model = load_model_and_env()
    policy = RecurrentPPOPolicy(model=model)
    metric_logger = MetricLogger(print_interval=512)

    # Initialize environment
    obs, info = env.reset()
    hidden_state = policy.init_hidden_state(batch_size=num_of_envs)
    if isinstance(hidden_state, tuple):
        hidden_state = tuple(h.to(device) for h in hidden_state)
    else:
        hidden_state = hidden_state.to(device)

    # Tracking variables
    step_count = 0
    cumulative_reward = 0.0
    episode_rewards = []
    episode_infos = []
    current_episode_length = 0

    print("\n=== Starting Out-of-Sample Bayesian Inference ===\n")
    print("Step   | Reward | Cum Reward | Realized PnL | Unrealized PnL | Fees | Leverage | Action Uncertainty")

    try:
        while step_count < 36000 * 2:  # Limit to out-of-sample range
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim if needed
            
            # Get Bayesian action with uncertainty
            mean_action, std_action, hidden_state = sample_bayesian_action(
                model, obs_tensor, hidden_state, n_samples=10
            )
            action_np = mean_action.cpu().numpy() # Remove batch dim

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = np.logical_or(terminated, truncated)

            # Update tracking
            reward_value = float(reward[0] if isinstance(reward, np.ndarray) else reward)
            cumulative_reward += reward_value
            episode_rewards.append(reward_value)
            episode_infos.append(info)
            current_episode_length += 1
            step_count += 1

            # Extract PnL values safely
            pnl_values = extract_pnl_values(info)
            
            # Print step info
            print(f"{step_count:06d} | {reward_value:+.4f} | {cumulative_reward:+.2f} | "
                  f"{pnl_values['realized_pnl']:+.5f} | {pnl_values['unrealized_pnl']:+.5f} | "
                  f"{pnl_values['fees']:+.5f}  |" 
                  f"{pnl_values['leverage']:.2f}x | {std_action.mean().item():.4f}", end='\r')

            # Handle episode end
            if np.any(done):
                # Log episode metrics
                metric_logger.log(step_count, {
                    "infos": episode_infos,
                    "episode_length": current_episode_length
                }, np.array(episode_rewards), policy)
                
                # Reset environment and tracking
                obs, info = env.reset()
                hidden_state = policy.init_hidden_state(batch_size=num_of_envs)
                if isinstance(hidden_state, tuple):
                    hidden_state = tuple(h.to(device) for h in hidden_state)
                else:
                    hidden_state = hidden_state.to(device)
                
                print(f"\n[Episode End] Steps: {current_episode_length} | Total Reward: {sum(episode_rewards):.2f}")
                episode_rewards = []
                episode_infos = []
                current_episode_length = 0
            else:
                obs = next_obs

    except KeyboardInterrupt:
        print("\n\nInference interrupted by user!")
    finally:
        # Final logging if episode was in progress
        if episode_rewards:
            metric_logger.log(step_count, {
                "infos": episode_infos,
                "episode_length": current_episode_length
            }, np.array(episode_rewards), policy)

        print(f"\n=== Inference Complete ===")
        print(f"Total Steps: {step_count}")
        print(f"Cumulative Reward: {cumulative_reward:.2f}")
        env.close()

if __name__ == "__main__":
    main()
