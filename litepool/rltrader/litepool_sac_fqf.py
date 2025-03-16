import numpy as np
import pandas as pd
from typing import Optional, Type
from packaging import version

import gymnasium
from gymnasium import spaces

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import (
  VecEnvObs,
  VecEnvStepReturn,
)
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise


import litepool
from litepool.python.protocol import LitePool


device = torch.device("cuda")

# ---------------------------
# 1. Recurrent Actor with Multi-Layer LSTM
# ---------------------------
class RecurrentActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, lstm_hidden_dim=128, num_layers=2):
        super(RecurrentActor, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim, device=device)
        return (h_0, c_0)

    def forward(self, state, hidden=None):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)

        x = F.relu(self.fc1(state))

        batch_size = state.shape[0] if state.dim() > 1 else 1
        if hidden is None or isinstance(hidden, bool):
            hidden = self.init_hidden(batch_size, state.device)

        x, hidden = self.lstm(x.unsqueeze(0) if state.dim() == 1 else x.unsqueeze(1), hidden)
        x = F.relu(self.fc2(x.squeeze(0) if state.dim() == 1 else x.squeeze(1)))

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()

        return mean, std, hidden 

    def sample(self, state, hidden=None, deterministic=False):
        mean, std, hidden = self.forward(state, hidden)
        normal_dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else normal_dist.rsample()
        action = torch.tanh(action)  
        return action, hidden

    def action_log_prob(self, state, hidden=None):
        mean, std, hidden = self.forward(state, hidden)
        normal_dist = torch.distributions.Normal(mean, std)

        raw_action = normal_dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = normal_dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


    def set_training_mode(self, mode: bool):
        self.train(mode)


# ---------------------------
# 2. Transformer-Based Twin Critic
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerCritic(nn.Module):
    def __init__(self, state_dim, action_dim, transformer_dim=128, num_heads=4):
        super(TransformerCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, transformer_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc2 = nn.Linear(transformer_dim, 256)
        self.q_1_out = nn.Linear(256, 1)  
        self.q_2_out = nn.Linear(256, 1)  

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x)).unsqueeze(0)  
        x = self.transformer(x).squeeze(0)
        x = F.relu(self.fc2(x))
        return self.q_1_out(x), self.q_2_out(x)

    def set_training_mode(self, mode: bool):
        self.train(mode) 


# ---------------------------
# 3. Custom FQF-DSAC Policy with Proper Initialization
# ---------------------------
class CustomFQFDSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomFQFDSACPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        # Initialize LSTM-based Actor
        self.actor = RecurrentActor(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor.optimizer = self.actor_optimizer
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=100000, eta_min=1e-5)

        # Initialize Transformer-based Critic
        self.critic = TransformerCritic(state_dim, action_dim)
        self.critic_target = TransformerCritic(state_dim, action_dim)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic.optimizer = self.critic_optimizer
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=100000, eta_min=1e-5)

        # Copy weights to target network
        self.critic_target.load_state_dict(self.critic.state_dict())

    def forward(self, obs, hidden, deterministic=False):
        return self.actor.sample(obs, hidden, deterministic)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if state is None:
            batch_size = observation.shape[0] if isinstance(observation, np.ndarray) else 1
            state = self.actor.init_hidden(batch_size, device=observation.device if torch.is_tensor(observation) else "cuda")

        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=next(self.actor.parameters()).device)

        action, new_state = self.actor.sample(observation, state, deterministic=deterministic)
        return action.detach().cpu().numpy(), new_state 

    @property
    def optimizer(self):
        return self.actor_optimizer 

# ---------------------------
# 4. Improved Training Step with Persistent Hidden States
# ---------------------------
class CustomFQFDSAC(SAC):
    def train_step(self, replay_buffer, batch_size=256):
        replay_data = replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        states = replay_data.observations
        actions = replay_data.actions
        rewards = replay_data.rewards
        next_states = replay_data.next_observations
        dones = replay_data.dones

        hidden_actor = None
        hidden_critic = None

        with torch.no_grad():
            next_actions, next_log_probs, hidden_actor = self.actor.sample(next_states, hidden_actor)
            next_q_values = self.critic_target(next_states, next_actions)
            target_quantiles = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_values = self.critic(states, actions)
        critic_loss = F.smooth_l1_loss(current_q_values, target_quantiles)

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        new_actions, log_probs, hidden_actor = self.actor.sample(states, hidden_actor)
        q_values_new = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_probs - q_values_new.mean()).mean()

        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        self.policy.optimizer.step()
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "entropy_loss": log_probs.mean().item()}


# ---------------------------
# 5. Ornstein-Uhlenbeck Noise for Continuous Exploration
# ---------------------------
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2):
        super().__init__()
        self.mu = np.array(mu)  # Mean value (usually 0)
        self.sigma = sigma  # Standard deviation
        self.theta = theta  # Mean reversion speed
        self.dt = dt  # Time step
        self.x_prev = np.zeros_like(self.mu)  # Initialize noise state

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)


class VecAdapter(VecEnvWrapper):
  def __init__(self, venv: LitePool):
    venv.num_envs = venv.spec.config.num_envs
    super().__init__(venv=venv)
    self.steps = 0
    self.header = True
    self.action_env_ids = np.arange(self.venv.num_envs, dtype=np.int32)

  def step_async(self, actions: np.ndarray) -> None:
      self.actions = actions 
      self.venv.send(self.actions, self.action_env_ids)

  def reset(self) -> VecEnvObs:
      self.steps = 0
      return self.venv.reset(self.action_env_ids)[0]

  def seed(self, seed: Optional[int] = None) -> None:
     if seed is not None:
          self.venv.seed(seed) 

  def step_wait(self) -> VecEnvStepReturn:
      obs, rewards, terms, truncs, info_dict = self.venv.recv()
      if (np.isnan(obs).any() or np.isinf(obs).any()):
          print("NaN in OBS...................")

      if (np.isnan(rewards).any() or np.isinf(rewards).any()):
          print("NaN in REWARDS...................")
          print(rewards)

      dones = terms + truncs
      infos = []
      for i in range(len(info_dict["env_id"])):
          infos.append({
              key: info_dict[key][i]
              for key in info_dict.keys()
              if isinstance(info_dict[key], np.ndarray)
          })


          if self.steps % 500  == 0 or dones[i]:
              print("id:{0}, steps:{1}, fees:{2:.8f}, balance:{3:.6f}, unreal:{4:.8f}, real:{5:.8f}, drawdown:{6:.8f}, leverage:{7:.4f}, count:{8}".format(
                    infos[i]["env_id"],  self.steps, infos[i]['fees'], infos[i]['balance'] - infos[i]['fees'], infos[i]['unrealized_pnl'], 
                    infos[i]['realized_pnl'], infos[i]['drawdown'], infos[i]['leverage'], infos[i]['trade_count']))
          
          if dones[i]:
              infos[i]["terminal_observation"] = obs[i]
              obs[i] = self.venv.reset(np.array([i]))[0]

      self.steps += 1
      return obs, rewards, dones, infos

import os
if os.path.exists('temp.csv'):
    os.remove('temp.csv')
env = litepool.make("RlTrader-v0", env_type="gymnasium", 
                          num_envs=32, batch_size=32,
                          num_threads=32,
                          is_prod=False,
                          is_inverse_instr=True,
                          api_key="",
                          api_secret="",
                          symbol="BTC-PERPETUAL",
                          tick_size=0.5,
                          min_amount=10,
                          maker_fee=-0.0001,
                          taker_fee=0.0005,
                          foldername="./train_files/", 
                          balance=1.0,
                          start=7200,
                          max=3601*5)

env.spec.id = 'RlTrader-v0'

env = VecAdapter(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecMonitor(env)

action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape))

model = CustomFQFDSAC(
    CustomFQFDSACPolicy,
    env,
    batch_size=256,
    buffer_size=1000000,                
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    learning_starts=100,        
    train_freq=64,           
    gradient_steps=64,    
    ent_coef='auto',                
    verbose=1,
    action_noise=action_noise,
    device=device)

if os.path.exists("sac_fqf_rltrader.zip"):
    model = CustomFQFDSAC.load("sac_fqf_rltrader.zip", env=env, device=device)
    model.ent_coef = "auto"
    model.log_ent_coef = torch.tensor(0.0, requires_grad=True, device=model.device)
    model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=1e-4)

    if os.path.exists("replay_fqf_buffer.pkl"):
        model.load_replay_buffer("replay_fqf_buffer.pkl")
    print("saved fqf model loaded")

for i in range(0, 50):
    model.learn(3605*32)
    model.save("sac_fqf_rltrader")
    model.save_replay_buffer("replay_fqf_buffer.pkl")
