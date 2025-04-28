import numpy as np
import torch

def flatten_obs(obs):
    """Flatten a single observation into a flat float32 numpy array."""
    if isinstance(obs, (tuple, list)):
        obs = np.concatenate([np.asarray(part, dtype=np.float32).flatten() for part in obs], axis=0)
    else:
        obs = np.asarray(obs, dtype=np.float32)
    return obs

def reset_hidden_state(hidden_state, done):
    """Reset hidden states where the episode is done."""
    if hidden_state is None:
        return None

    done_tensor = torch.as_tensor(done, dtype=torch.bool, device=hidden_state[0].device if isinstance(hidden_state, tuple) else hidden_state.device)

    if isinstance(hidden_state, tuple):
        # For LSTM (h, c)
        return tuple(
            h.masked_fill(done_tensor.view(1, -1, 1), 0.0)
            for h in hidden_state
        )
    else:
        # For GRU or RNN (single tensor)
        return hidden_state.masked_fill(done_tensor.view(1, -1, 1), 0.0)

class PPOCollector:
    def __init__(self, env, policy, rollout_len, device='cpu'):
        """
        env: Vectorized environment (e.g., EnvPool or gym.vector)
        policy: Your recurrent policy model
        rollout_len: Number of steps to collect (per environment)
        device: 'cpu' or 'cuda'
        """
        self.env = env
        self.policy = policy
        self.rollout_len = rollout_len
        self.device = device

    def collect(self):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        hidden_state = None

        obs, _ = self.env.reset()
        obs = flatten_obs(obs)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)  # Ensure batch dimension if single env

        n_envs = obs.shape[0]

        for step in range(self.rollout_len):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                action, log_prob, value, hidden_state = self.policy.forward(obs_tensor, hidden_state)

                # Detach outputs to CPU
                action = action.detach().cpu()
                log_prob = log_prob.detach().cpu()
                value = value.detach().cpu()

            # Step environment
            action_env = action.numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_env)
            
            done = np.logical_or(terminated, truncated)

            next_obs = flatten_obs(next_obs)
            if len(next_obs.shape) == 1:
                next_obs = np.expand_dims(next_obs, axis=0)

            # Reset hidden states where episodes ended
            hidden_state = reset_hidden_state(hidden_state, done)

            # Save rollout step
            obs_buf.append(torch.as_tensor(obs, device='cpu'))  # (n_envs, feature_dim)
            act_buf.append(action)                              # (n_envs, action_dim)
            logp_buf.append(log_prob)                           # (n_envs, 1) or (n_envs,)
            rew_buf.append(torch.as_tensor(reward, dtype=torch.float32))  # (n_envs,)
            done_buf.append(torch.as_tensor(done, dtype=torch.float32))   # (n_envs,)
            val_buf.append(value)                               # (n_envs, 1)

            obs = next_obs

        # Stack rollout buffers
        batch = {
            'obs': torch.stack(obs_buf),    # (rollout_len, n_envs, feature_dim)
            'act': torch.stack(act_buf),    # (rollout_len, n_envs, action_dim)
            'logp': torch.stack(logp_buf),  # (rollout_len, n_envs)
            'rew': torch.stack(rew_buf),    # (rollout_len, n_envs)
            'done': torch.stack(done_buf),  # (rollout_len, n_envs)
            'val': torch.stack(val_buf),    # (rollout_len, n_envs)
        }

        return batch
