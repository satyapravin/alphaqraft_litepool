import numpy as np
import os
from typing import Optional, Type
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

from stable_baselines3.common.vec_env.base_vec_env import (
  VecEnvObs,
  VecEnvStepReturn,
)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import ActorCriticPolicy

import litepool
from litepool.python.protocol import LitePool

device = torch.device("cpu")

# ------------------
# 1. GRU Feature Extractor
# ------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GRUFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_dim=64, gru_hidden_dim=128, num_layers=2, num_envs=64):
        super(GRUFeatureExtractor, self).__init__(observation_space, feature_dim)

        self.num_envs = num_envs 
        self.state_dim = 242
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers

        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = self.state_dim - (self.position_dim + self.trade_dim)

        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.trade_fc = nn.Sequential(
            nn.Linear(self.trade_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 32), nn.ReLU()
        )

        self.gru = nn.GRU(64, gru_hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fusion_fc = nn.Sequential(
            nn.Linear(gru_hidden_dim + 32, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, feature_dim), nn.ReLU()
        )

        self.hidden_states = None

    def forward(self, observations, env_indices=None):
        batch_size, flat_dim = observations.shape
        time_steps = flat_dim // self.state_dim
        assert flat_dim % self.state_dim == 0, "Incorrect observation shape!"
        
        observations = observations.view(batch_size, time_steps, self.state_dim)

        position_state = observations[:, -1, :self.position_dim]  
        trade_state = observations[:, :, self.position_dim:self.position_dim + self.trade_dim]
        market_state = observations[:, :, self.position_dim + self.trade_dim:]

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)

        x = torch.cat([trade_out, market_out], dim=-1)

        if self.hidden_states is None or self.hidden_states.shape[1] != batch_size:
            self.hidden_states = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=observations.device)

        self.hidden_states = self.hidden_states.detach()

        x, self.hidden_states = self.gru(x, self.hidden_states)
        x = x[:, -1, :]  # Take last time step output
        x = torch.cat([x, position_out], dim=-1)

        return self.fusion_fc(x)

    def reset_hidden_states(self, env_indices):
        if self.hidden_states is not None:
            self.hidden_states[:, env_indices, :] = 0 

class VecAdaptor(VecEnvWrapper):
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
      envs_to_reset = []

      for i in range(len(info_dict["env_id"])):
          infos.append({
              key: info_dict[key][i]
              for key in info_dict.keys()
              if isinstance(info_dict[key], np.ndarray)
          })


          if self.steps % 1000  == 0 or dones[i]:
              print("id:{0}, steps:{1}, fees:{2:.8f}, balance:{3:.6f}, unreal:{4:.8f}, real:{5:.8f}, drawdown:{6:.8f}, leverage:{7:.4f}, count:{8}".format(
                    infos[i]["env_id"],  self.steps, infos[i]['fees'], infos[i]['balance'] - infos[i]['fees'] + infos[i]['unrealized_pnl'],
                    infos[i]['unrealized_pnl'], infos[i]['realized_pnl'], infos[i]['drawdown'], infos[i]['leverage'], infos[i]['trade_count']))

          if dones[i]:
              envs_to_reset.append(i) 
              infos[i]["terminal_observation"] = obs[i]
              obs[i] = self.venv.reset(np.array([i]))[0]

      if envs_to_reset:
          model.policy.features_extractor.reset_hidden_states(envs_to_reset)

      self.steps += 1
      return obs, rewards, dones, infos

env = litepool.make("RlTrader-v0", env_type="gymnasium", num_envs=64, batch_size=64, num_threads=64,
                    is_prod=False, is_inverse_instr=True, api_key="", api_secret="",
                    symbol="BTC-PERPETUAL", tick_size=0.5, min_amount=10, maker_fee=-0.0001,
                    taker_fee=0.0005, foldername="./train_files/", balance=1.0, start=360000, max=9001*10)

env.spec.id = 'RlTrader-v0'
env = VecNormalize(VecMonitor(VecAdaptor(env)), norm_obs=True, norm_reward=True)

model = PPO(
    'MlpPolicy',
    env,
    batch_size=64,  # Reduce batch size for better GRU training
    learning_rate=0.0003,
    gamma=0.99,  # More discounting for better long-term decisions
    gae_lambda=0.95,
    clip_range=0.2,  # Slightly lower clip for stable updates
    ent_coef=0.05,  # Reduce entropy coefficient slightly
    vf_coef=1.0,  # Increase value loss importance
    n_epochs=1,   # More epochs per update
    n_steps=900,  # Larger rollout buffer
    verbose=1,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[128, 64, 32]),
        features_extractor_class=GRUFeatureExtractor,
        features_extractor_kwargs=dict(feature_dim=64, num_envs=env.num_envs)
    ),
    device=device
)

if os.path.exists("ppo_rltrader.zip"):
    model = PPO.load("ppo_rltrader.zip", env=env, device=device)
    print("saved ppo model loaded")

for i in range(0, 50):
    model.learn(9005*64)
    model.save("ppo_rltrader")

