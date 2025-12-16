"""
Vectorized environment wrapper with reward normalization.

Observations: Already bounded via tanh/normalization, so just clip for safety.
Rewards: Use running std normalization (standard PPO practice).
"""
import numpy as np
import torch
from typing import Optional, Union


class RunningMeanStd:
    """Tracks running mean and std using Welford's algorithm."""
    
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon  # Small initial count for stability
    
    def update(self, x: np.ndarray):
        """Update with a batch of values."""
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Parallel algorithm for combining statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class VecNormalize:
    """
    Vectorized environment wrapper.
    
    Observations: Bounded signals, just clip for safety.
    Rewards: Running std normalization (divide by std, don't subtract mean).
    
    Args:
        env: Vectorized environment
        num_envs: Number of parallel environments
        norm_reward: Whether to normalize rewards by running std
        clip_obs: Clip observations to [-clip_obs, clip_obs]
        clip_reward: Clip normalized rewards to [-clip_reward, clip_reward]
        gamma: Discount factor for return tracking
    """
    
    def __init__(
        self, 
        env, 
        num_envs, 
        device=None,
        obs_dim=16,
        norm_obs=True,
        norm_reward=True,  # Enable reward normalization by default
        clip_obs=10.0,
        clip_reward=10.0, 
        gamma=0.99, 
        epsilon=1e-8
    ):
        self.env = env
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.training = True
        
        # Return tracking
        self.returns = np.zeros(num_envs, dtype=np.float32)
        
        # Running stats for reward normalization
        self.ret_rms = RunningMeanStd()

    def train(self):
        """Training mode - update running stats."""
        self.training = True
    
    def eval(self):
        """Evaluation mode - don't update stats."""
        self.training = False

    def save(self, path):
        """Save reward normalization stats."""
        state = {
            'ret_mean': self.ret_rms.mean,
            'ret_var': self.ret_rms.var,
            'ret_count': self.ret_rms.count,
        }
        torch.save(state, path)

    def load(self, path):
        """Load reward normalization stats."""
        try:
            state = torch.load(path, map_location='cpu')
            self.ret_rms.mean = float(state.get('ret_mean', 0.0))
            self.ret_rms.var = float(state.get('ret_var', 1.0))
            self.ret_rms.count = float(state.get('ret_count', 1e-4))
        except Exception:
            pass  # Use defaults if load fails

    def _clip_obs(self, obs):
        """Clip observations for safety."""
        if self.norm_obs and self.clip_obs > 0:
            return np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def _normalize_reward(self, reward):
        """Normalize rewards by running std (not mean)."""
        if self.norm_reward:
            # Update running stats during training
            if self.training:
                self.ret_rms.update(reward)
            # Normalize by std only (don't subtract mean - changes optimization target)
            normalized = reward / self.ret_rms.std
            # Clip for stability
            if self.clip_reward > 0:
                normalized = np.clip(normalized, -self.clip_reward, self.clip_reward)
            return normalized
        return reward

    def step(self, actions):
        """Take a step in all environments."""
        obs, rews, terminations, truncations, infos = self.env.step(actions)
        
        # Track returns
        self.returns = self.returns * self.gamma + rews
        
        # Process observations and rewards
        obs = self._clip_obs(obs)
        rews = self._normalize_reward(rews)
        
        # Reset returns for done environments
        dones = np.logical_or(terminations, truncations)
        self.returns[dones] = 0.0
        
        return obs, rews, terminations, truncations, infos

    def reset(self, env_id: Optional[Union[int, np.ndarray]] = None):
        """Reset environments."""
        if env_id is None:
            obs, info = self.env.reset()
            self.returns[:] = 0.0
        else:
            env_id = np.array([env_id] if np.isscalar(env_id) else env_id)
            obs, info = self.env.reset(env_id)
            self.returns[env_id] = 0.0
        
        obs = self._clip_obs(obs)
        return obs, info

    def __len__(self):
        return self.num_envs

    def close(self):
        return self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)
