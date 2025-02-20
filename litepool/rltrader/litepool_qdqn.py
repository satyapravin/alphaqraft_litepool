import pandas as pd
import gymnasium
from typing import Optional
import numpy as np
import torch as th
import torch.nn as nn
from packaging import version
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import (
  VecEnvObs,
  VecEnvStepReturn,
)
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import QRDQN
import litepool

from litepool.python.protocol import LitePool

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type
from gymnasium import spaces

device = torch.device("cuda")



class ResetHiddenStateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        dones = self.locals["dones"]  
        for env_index, done in enumerate(dones):
            if done:  
                self.model.policy.quantile_net.features_extractor.reset_hidden_state_for_env(env_index)
        return True


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, lstm_hidden_size: int = 16):
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim=lstm_hidden_size)
        self.lstm_hidden_size = lstm_hidden_size
        self.n_input_channels = 98
        self.lstm = nn.LSTM(self.n_input_channels, lstm_hidden_size, batch_first=True, bidirectional=False).to(device)
        self.hidden = None
        
    def reset(self):
        self.hidden = None

    def reset_hidden_state_for_env(self, env_idx: int):
        if self.hidden is not None:
            self.hidden = (
                self.hidden[0].detach(),
                self.hidden[1].detach()
            )
            self.hidden[0][:, env_idx, :] = 0  # Reset hidden state (h_0)
            self.hidden[1][:, env_idx, :] = 0  # Reset cell state (c_0)


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lstm_input = observations  
        batch_size = observations.shape[0]
        lstm_input = lstm_input.view(batch_size, 5, 98)  
        if self.hidden is None or lstm_input.shape[0] != self.hidden[0].shape[1]:
            self.hidden = (
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(observations.device),
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(observations.device),
            )
        else:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden)  # (batch_size, seq_len, hidden_size)
        final_output = lstm_out[:, -1, :]
        return final_output

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
                          balance=0.01,
                          start=20000,
                          max=360001)

env.spec.id = 'RlTrader-v0'
env = VecAdapter(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecMonitor(env)

kwargs = dict(use_sde=True, sde_sample_freq=4)

policy_kwargs = {
    'features_extractor_class': LSTMFeatureExtractor,
    'features_extractor_kwargs': {'lstm_hidden_size': 64},
    'activation_fn': th.nn.ReLU,
    'net_arch': [64, 64, 64],
    'n_quantiles': 50,
    'normalize_images': False
}

import os


if os.path.exists("litepool_qrdqn.zip"):
    model = QRDQN.load("litepool_qrdqn", env)
    print("QRDQN saved model loaded")
else:
    model = QRDQN("MlpPolicy", env, train_freq=32, batch_size=64, target_update_interval=1000, policy_kwargs=policy_kwargs, verbose=1)
model.learn(30000000, callback=ResetHiddenStateCallback())
model.save("litepool_qrdqn")
