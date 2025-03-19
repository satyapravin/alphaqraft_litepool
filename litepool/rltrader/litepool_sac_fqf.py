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

from stable_baselines3.common.buffers import ReplayBuffer
from tianshou.data import ReplayBuffer as TianshouReplayBuffer, Batch

device = torch.device("cuda")


class RecurrentReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, sequence_length=10, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.sequence_length = sequence_length
        self.tianshou_buffer = TianshouReplayBuffer(size=self.buffer_size)

    def add(self, obs, next_obs, action, reward, done, infos):
        num_envs = len(done)
        done = np.asarray(done, dtype=bool)  
        reward = np.asarray(reward, dtype=np.float32)
        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        action = np.asarray(action)
        terminated = np.copy(done)
        truncated = np.array([False for info in infos], dtype=bool)
       
        for env_id in range(0, num_envs):
            batch = Batch(
                obs=obs[env_id],
                act=action[env_id],
                rew=reward[env_id],
                done=done[env_id],  
                obs_next=next_obs[env_id],
                terminated=terminated[env_id],  
                truncated=truncated[env_id],  
            )

            self.tianshou_buffer.add(batch)


    def sample(self, batch_size, env=None):
        indices = np.random.choice(len(self.tianshou_buffer), batch_size)
        sequences = []

        for idx in indices:
            start_idx = max(0, idx - self.sequence_length + 1)
            end_idx = idx + 1

            batch = self.tianshou_buffer[start_idx:end_idx]
            sequences.append(batch)

        sampled_batch = Batch.cat(sequences)
        sampled_batch.observations = sampled_batch.obs 
        sampled_batch.next_observations = sampled_batch.obs_next
        sampled_batch.rewards = torch.tensor(sampled_batch.rew, dtype=torch.float32, device=device)
        sampled_batch.dones = torch.tensor(sampled_batch.done, dtype=torch.float32, device=device)
        sampled_batch.actions = torch.tensor(sampled_batch.act, dtype=torch.float32, device=device)
        return sampled_batch

    def __len__(self):
        return len(self.tianshou_buffer)

# ------------------
# 1. Recurrent Actor 
# ------------------
class RecurrentActor(nn.Module):
    def __init__(self, state_dim=2420, action_dim=3, hidden_dim=256, gru_hidden_dim=128, num_layers=2):
        super(RecurrentActor, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers
        self.feature_dim = 242
        self.time_steps = 10  

        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = 242 - (18 + 6) 

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

        self.gru = nn.GRU(64, gru_hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fusion_fc = nn.Linear(gru_hidden_dim + 32, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=device)

    def forward(self, state, hidden=None):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)

        batch_size = state.shape[0]
        state = state.view(batch_size, self.time_steps, self.feature_dim)  # Reshape to (batch, time_steps, 242)

        market_state = state[:, :, :self.market_dim]  # First 218 values → Market signals
        position_state = state[:, -1, self.market_dim:self.market_dim + self.position_dim]  # Next 18 values → Position signals
        trade_state = state[:, :, self.market_dim + self.position_dim:]  # Last 6 values → Trade signals

        position_out = self.position_fc(position_state) 
        trade_out = self.trade_fc(trade_state)  
        market_out = self.market_fc(market_state)  

        x = torch.cat([trade_out, market_out], dim=-1) 

        if hidden is None or hidden.shape[1] != batch_size:
            hidden = self.init_hidden(batch_size, state.device)

        x, hidden = self.gru(x, hidden)  
        x = x[:, -1, :]  
        x = torch.cat([x, position_out], dim=-1) 

        x = F.relu(self.fusion_fc(x))

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std, hidden

    def set_training_mode(self, mode: bool):
        self.train(mode)


    def sample(self, state, hidden=None, deterministic=False):
        mean, std, hidden = self.forward(state, hidden)
        normal_dist = torch.distributions.Normal(mean, std)
        raw_action = mean if deterministic else normal_dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal_dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True) 
        return action, log_prob, hidden

    def action_log_prob(self, state, hidden=None):
        mean, std, hidden = self.forward(state, hidden)
        normal_dist = torch.distributions.Normal(mean, std)

        raw_action = normal_dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = normal_dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


# ---------------------------
# 2. Transformer-Based Twin Critic
# ---------------------------
class TransformerCritic(nn.Module):
    def __init__(self, state_dim=2420, action_dim=3, transformer_dim=128, num_heads=4):
        super(TransformerCritic, self).__init__()
        self.time_steps = 10 
        self.feature_dim = 242  # 242 per timestep

        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = 242 - (18 + 6)  # 218 market features

        # Process position features separately
        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Trade and market features go through Transformer
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

        # Transformer processes time-dependent trade & market signals
        self.fc1 = nn.Linear(64 + action_dim, transformer_dim)  # 64 = trade (32) + market (32)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        # Fix: Adjust `self.fc2` input size to match concatenated features
        self.fc2 = nn.Linear(transformer_dim + 32, 256)  # Transformer output (128) + Position features (32)
        
        self.q_1_out = nn.Linear(256, 1)
        self.q_2_out = nn.Linear(256, 1)

    def forward(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)

        batch_size = state.shape[0]
        state = state.view(batch_size, self.time_steps, self.feature_dim)  

        # Correct feature extraction
        market_state = state[:, :, :self.market_dim]  # First 218 values → Market signals
        position_state = state[:, -1, self.market_dim:self.market_dim + self.position_dim]  # Last timestep position
        trade_state = state[:, :, self.market_dim + self.position_dim:]  # Last 6 values → Trade signals

        position_out = self.position_fc(position_state)  
        trade_out = self.trade_fc(trade_state) 
        market_out = self.market_fc(market_state) 

        x = torch.cat([trade_out, market_out], dim=-1)  
        action = action.unsqueeze(1).expand(-1, self.time_steps, -1)
        x = torch.cat([x, action], dim=-1)  

        x = F.relu(self.fc1(x))
        x = self.transformer(x)

        x = torch.cat([x[:, -1, :], position_out], dim=-1) 
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

        state_dim = observation_space.shape[0]  # Expecting 2420 (10 * 242)
        action_dim = action_space.shape[0]

        self.actor = RecurrentActor(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor.optimizer = self.actor_optimizer

        self.critic = TransformerCritic(state_dim, action_dim)
        self.critic_target = TransformerCritic(state_dim, action_dim)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic.optimizer = self.critic_optimizer
        self.critic_target.load_state_dict(self.critic.state_dict())

    def forward(self, obs, hidden, deterministic=False):
        return self.actor.sample(obs, hidden, deterministic)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if state is None:
            batch_size = observation.shape[0] if torch.is_tensor(observation) else 1
            state = self.actor.init_hidden(batch_size, device=device)

        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=next(self.actor.parameters()).device)

        action, log_prob, new_state = self.actor.sample(observation, state, deterministic=deterministic)
        return action.detach().cpu().numpy(), new_state


# ---------------------------
# 4. Improved Training Step with Persistent Hidden States
# ---------------------------
class CustomFQFDSAC(SAC):
    def __init__(self, policy, env, learning_rate=3e-4, alpha=0.2, **kwargs):
        super().__init__(policy, env, learning_rate=learning_rate, **kwargs)
        self.alpha = alpha
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        print("DEBUG: Custom train() is running!")  # Ensure this prints

        for _ in range(gradient_steps):
            # Sample batch from replay buffer
            replay_buffer = self.replay_buffer
            if replay_buffer is None or len(replay_buffer) < batch_size:
                print("DEBUG: Not enough samples in replay buffer, skipping training step.")
                return

            # Manually call our custom train_step()
            losses = self.train_step(replay_buffer, batch_size)

            print(f"DEBUG: Training step completed. Losses: {losses}")

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
            next_q1_values, next_q2_values = self.critic_target(next_states, next_actions)
            # Take the minimum Q-value to reduce overestimation bias
            next_q_values = torch.min(next_q1_values, next_q2_values)

            # Debugging: Print shapes before modifying
            print(f"next_q_values shape: {next_q_values.shape}")  # Should be [batch_size, num_quantiles]
            print(f"rewards shape before view: {rewards.shape}")  # Should be [batch_size]
            print(f"dones shape before view: {dones.shape}")  # Should be [batch_size]

            # Fix: Ensure rewards and dones have correct shape
            rewards = rewards.view(-1, 1)  # Ensures [batch_size, 1]
            dones = dones.view(-1, 1)  # Ensures [batch_size, 1]

            print(f"rewards shape after view: {rewards.shape}")  # Should be [batch_size, 1]
            print(f"dones shape after view: {dones.shape}")  # Should be [batch_size, 1]

            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            print(f"target_q_values shape: {target_q_values.shape}")  # Should be [batch_size, num_quantiles]

        # Get current Q-values
        current_q1_values, current_q2_values = self.critic(states, actions)

        # Take the minimum Q-value to prevent overestimation
        current_q_values = torch.min(current_q1_values, current_q2_values)
        print(f"current_q_values shape: {current_q_values.shape}")  # Should be [batch_size, num_quantiles]

        # Fix: Ensure target_q_values and current_q_values have the same shape
        target_q_values = target_q_values.view_as(current_q_values)

        print(f"Final target_q_values shape: {target_q_values.shape}")  # Should match current_q_values

        # Compute critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        # Actor update
        new_actions, log_probs, hidden_actor = self.actor.sample(states, hidden_actor)
        q_values_new = self.critic(states, new_actions)
        alpha = self.alpha if isinstance(self.alpha, float) else self.log_alpha.exp()
        actor_loss = (alpha * log_probs - q_values_new.mean()).mean()

        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        self.policy.optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy_loss": log_probs.mean().item(),
        }
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


          if self.steps % 50  == 0 or dones[i]:
              print("id:{0}, steps:{1}, fees:{2:.8f}, balance:{3:.6f}, unreal:{4:.8f}, real:{5:.8f}, drawdown:{6:.8f}, leverage:{7:.4f}, count:{8}".format(
                    infos[i]["env_id"],  self.steps, infos[i]['fees'], infos[i]['balance'] - infos[i]['fees'] + infos[i]['unrealized_pnl'], 
                    infos[i]['unrealized_pnl'], infos[i]['realized_pnl'], infos[i]['drawdown'], infos[i]['leverage'], infos[i]['trade_count']))
          
          if dones[i]:
              infos[i]["terminal_observation"] = obs[i]
              obs[i] = self.venv.reset(np.array([i]))[0]

      self.steps += 1
      return obs, rewards, dones, infos

import os
if os.path.exists('temp.csv'):
    os.remove('temp.csv')
env = litepool.make("RlTrader-v0", env_type="gymnasium", 
                          num_envs=64, batch_size=64,
                          num_threads=64,
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
                          start=360000,
                          max=7201*10)

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
    learning_rate=5e-4,
    gamma=0.99,
    tau=0.005,
    learning_starts=100,        
    train_freq=64,           
    gradient_steps=64,    
    ent_coef='auto',                
    verbose=1,
    replay_buffer_class=RecurrentReplayBuffer,  
    replay_buffer_kwargs={"sequence_length": 90},  
    action_noise=action_noise,
    device=device)

if os.path.exists("sac_fqf_rltrader.zip"):
    model = CustomFQFDSAC.load("sac_fqf_rltrader.zip", prioritized_replay=True, env=env, device=device)
    model.ent_coef = "auto"
    model.log_ent_coef = torch.tensor(0.0, requires_grad=True, device=model.device)
    model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=1e-4)

    if os.path.exists("replay_fqf_buffer.pkl"):
        model.load_replay_buffer("replay_fqf_buffer.pkl")
    print("saved fqf model loaded")

for i in range(0, 500):
    model.learn(7205*64)
    model.save("sac_fqf_rltrader")
    model.save_replay_buffer("replay_fqf_buffer.pkl")
