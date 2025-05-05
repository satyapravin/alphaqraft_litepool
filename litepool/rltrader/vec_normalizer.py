import numpy as np
import torch
from typing import Optional, Union

class RunningMeanStd:
    def __init__(self, device, shape=(), epsilon=3e-4):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class VecNormalize:
    def __init__(self, env, num_envs, device, norm_obs=True, norm_reward=True, clip_obs=10.,
                 clip_reward=1, gamma=0.99, epsilon=1e-8):
        self.env = env
        self.num_envs = num_envs
        self.device = device
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        self.time_steps = 2
        self.feature_dim = 242
        self.flat_dim = self.time_steps * self.feature_dim

        self.obs_rms = RunningMeanStd(shape=(self.feature_dim,), device=self.device)
        self.ret_rms = RunningMeanStd(shape=(), device=self.device)
        self.returns = torch.zeros(self.num_envs, device=device)

    def save(self, path):
        """Save normalization stats to a file."""
        state = {
            'obs_mean': self.obs_rms.mean.cpu(),
            'obs_var': self.obs_rms.var.cpu(),
            'obs_count': self.obs_rms.count.cpu(),
            'ret_mean': self.ret_rms.mean.cpu(),
            'ret_var': self.ret_rms.var.cpu(),
            'ret_count': self.ret_rms.count.cpu()
        }
        torch.save(state, path)
        print(f"Saved normalization stats to {path}")

    def load(self, path):
        """Load normalization stats from a file."""
        state = torch.load(path, map_location=self.device)
        self.obs_rms.mean = state['obs_mean'].to(self.device)
        self.obs_rms.var = state['obs_var'].to(self.device)
        self.obs_rms.count = state['obs_count'].to(self.device)
        self.ret_rms.mean = state['ret_mean'].to(self.device)
        self.ret_rms.var = state['ret_var'].to(self.device)
        self.ret_rms.count = state['ret_count'].to(self.device)
        print(f"Loaded normalization stats from {path}")

    def normalize_obs(self, obs):
        if not self.norm_obs:
            return obs
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        batch_size = obs.shape[0]
        obs = obs.view(batch_size, self.time_steps, self.feature_dim)
        flat_obs = obs.view(-1, self.feature_dim)
        self.obs_rms.update(flat_obs)
        mean = self.obs_rms.mean.view(1, 1, self.feature_dim)
        var = self.obs_rms.var.view(1, 1, self.feature_dim)
        normed = (obs - mean) / torch.sqrt(var + self.epsilon)
        normed = torch.clamp(normed, -self.clip_obs, self.clip_obs)
        return normed.view(batch_size, self.flat_dim)

    def __len__(self):
        return self.num_envs

    def close(self):
        return self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space  # Should be Box(shape=(242*2,))

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    def seed(self, seed=None):
        return self.env.seed(seed)

    def normalize_reward(self, reward):
        if not self.norm_reward:
            return reward
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)  # Shape: [num_envs]
        self.ret_rms.update(self.returns)
        normed = reward / torch.sqrt(self.ret_rms.var + self.epsilon)
        return torch.clamp(normed, -self.clip_reward, self.clip_reward)

    def step(self, actions):
        obs, rews, terminations, truncations, infos = self.env.step(actions)

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # Shape: [num_envs, 242*2]
        rews = torch.as_tensor(rews, dtype=torch.float32, device=self.device)  # Shape: [num_envs]

        self.returns = self.returns * self.gamma + rews

        obs = self.normalize_obs(obs)
        rews = self.normalize_reward(rews)

        dones = torch.logical_or(torch.as_tensor(terminations), torch.as_tensor(truncations))
        self.returns[dones] = 0.0

        return obs.cpu().numpy(), rews.cpu().numpy(), terminations, truncations, infos

    def reset(self, env_id: Optional[Union[int, np.ndarray]] = None):
        """Reset environments, optionally specifying which envs to reset"""
        if env_id is None:
            obs, info = self.env.reset()
        else:
            # Convert env_id to numpy array if it isn't already
            env_id = np.array([env_id] if np.isscalar(env_id) else env_id)
            obs, info = self.env.reset(env_id)
    
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.returns[env_id] = 0.0 if env_id is not None else self.returns.zero_()
        obs = self.normalize_obs(obs)
        return obs.cpu().numpy(), info

    def __getattr__(self, name):
        return getattr(self.env, name)
