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

device = torch.device("cuda")

class GRUFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_dim=64, gru_hidden_dim=128, num_layers=2):
        super(GRUFeatureExtractor, self).__init__(observation_space, feature_dim)

        self.time_steps = 10  
        self.state_dim = 242 
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers

        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = self.state_dim - (self.position_dim + self.trade_dim)

        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.trade_fc = nn.Sequential(
            nn.Linear(self.trade_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        self.gru = nn.GRU(64, gru_hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        
        self.fusion_fc = nn.Sequential(
                nn.Linear(gru_hidden_dim + 32, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, feature_dim),
                nn.ReLU()
        )

    def forward(self, observations, hidden_states=None):
        batch_size, flat_dim = observations.shape

        assert flat_dim == self.time_steps * self.state_dim, "Incorrect observation shape!"
        observations = observations.view(batch_size, self.time_steps, self.state_dim)

        position_state = observations[:, -1, :self.position_dim] 
        trade_state = observations[:, :, self.position_dim:self.position_dim + self.trade_dim]
        market_state = observations[:, :, self.position_dim + self.trade_dim:]

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)

        x = torch.cat([trade_out, market_out], dim=-1)  

        if hidden_states is None:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=observations.device)

        x, new_hidden_states = self.gru(x, hidden_states)

        x = x[:, -1, :]  
        x = torch.cat([x, position_out], dim=-1)

        return F.relu(self.fusion_fc(x)), new_hidden_states

class RecurrentActor(nn.Module):
    def __init__(self, feature_dim=256, action_dim=3):
        super(RecurrentActor, self).__init__()

        self.mean = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Linear(feature_dim, action_dim)

    def forward(self, extracted_features):
        mean = self.mean(extracted_features)
        log_std = self.log_std(extracted_features).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, extracted_features, deterministic=False):
        mean, std = self.forward(extracted_features)
        normal_dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else normal_dist.rsample()
        return torch.tanh(action)

class FQFValueCritic(nn.Module):
    def __init__(self, feature_dim=64, num_quantiles=32):
        super(FQFValueCritic, self).__init__()
        self.num_quantiles = num_quantiles

        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out_fc = nn.Linear(128, num_quantiles) 

        self.quantile_fractions = nn.Parameter(torch.rand(num_quantiles))

    def forward(self, extracted_features):
        x = F.relu(self.fc1(extracted_features))
        x = F.relu(self.fc2(x))
        quantile_values = self.out_fc(x) 

        sorted_fractions = torch.sort(torch.sigmoid(self.quantile_fractions))[0]
        return quantile_values, sorted_fractions

from stable_baselines3.ppo.policies import ActorCriticPolicy

class FQFPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(FQFPPOPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.features_extractor = GRUFeatureExtractor(observation_space)
        self.actor = RecurrentActor(action_dim=action_space.shape[0])
        self.critic = FQFValueCritic()

    def forward(self, obs, deterministic=False):
        extracted_features, _ = self.features_extractor(obs)
        actions = self.actor.sample(extracted_features, deterministic)
        quantile_values, quantile_fractions = self.critic(extracted_features)
        values = quantile_values.mean(dim=1)
        mean, std = self.actor.forward(extracted_features)
        log_probs = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
        return actions, values, log_probs, quantile_values, quantile_fractions

def quantile_huber_loss(quantile_values, target_values, quantile_fractions):
    td_error = target_values.unsqueeze(1) - quantile_values 
    huber_loss = F.smooth_l1_loss(td_error, torch.zeros_like(td_error), reduction='none')

    quantile_weights = torch.abs(quantile_fractions - (td_error.detach() < 0).float())
    loss = (quantile_weights * huber_loss).mean()
    return loss

from stable_baselines3 import PPO

class FQFPPO(PPO):
    def train(self):
        for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
            obs, actions, rewards, next_obs, dones = rollout_data.observations, rollout_data.actions, rollout_data.rewards, rollout_data.next_observations, rollout_data.dones

            _, values, log_probs, quantile_values, quantile_fractions = self.policy(obs)
            _, next_values, _, next_quantile_values, _ = self.policy(next_obs)

            target_values = rewards + self.gamma * (1 - dones) * next_quantile_values.mean(dim=1)
            critic_loss = quantile_huber_loss(quantile_values, target_values, quantile_fractions)

            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

        super().train()

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
              infos[i]["terminal_observation"] = obs[i]
              obs[i] = self.venv.reset(np.array([i]))[0]

      self.steps += 1
      return obs, rewards, dones, infos

# ---------------------------
# 5. Optimized PPO Training with Adaptive Learning Rate
# ---------------------------
env = litepool.make("RlTrader-v0", env_type="gymnasium", num_envs=64, batch_size=64, num_threads=64,
                    is_prod=False, is_inverse_instr=True, api_key="", api_secret="",
                    symbol="BTC-PERPETUAL", tick_size=0.5, min_amount=10, maker_fee=-0.0001,
                    taker_fee=0.0005, foldername="./train_files/", balance=1.0, start=1, max=72001*10)

env.spec.id = 'RlTrader-v0'
env = VecNormalize(VecMonitor(VecAdaptor(env)), norm_obs=True, norm_reward=True)

model = PPO(
    FQFPPOPolicy,
    env,
    batch_size=64,  # Reduce batch size for better GRU training
    learning_rate=0.0003,
    gamma=0.995,  # More discounting for better long-term decisions
    gae_lambda=0.97,
    clip_range=0.3,  # Slightly lower clip for stable updates
    ent_coef=0.005,  # Reduce entropy coefficient slightly
    vf_coef=2.0,  # Increase value loss importance
    n_epochs=1,   # More epochs per update
    n_steps=4096,  # Larger rollout buffer
    verbose=1,
    device=device
)

if os.path.exists("ppo_rltrader.zip"):
    model = PPO.load("ppo_rltrader.zip", env=env, device=device)
    print("saved ppo model loaded")

for i in range(0, 50):
    model.learn(72005*64)
    model.save("ppo_rltrader")

