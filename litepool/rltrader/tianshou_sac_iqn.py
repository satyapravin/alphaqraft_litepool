import numpy as np
import time
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.optim import Adam
import copy
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import litepool
from tianshou.env import BaseVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.amp import autocast, GradScaler
from tianshou.env import DummyVectorEnv

device = torch.device("cuda")

#-------------------------------------
# Make environment
#------------------------------------
num_of_envs=32

env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=num_of_envs, batch_size=num_of_envs,
    num_threads=num_of_envs, is_prod=False, is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL", tick_size=0.5, min_amount=10,
    maker_fee=-0.0001, taker_fee=0.0005, foldername="./train_files/",
    balance=1.0, start=1, max=72001*10
)

env.spec.id = 'RlTrader-v0'

class VecNormalize:
    def __init__(self, env, num_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        self.env = env
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_env = num_env
        self.obs_rms = RunningMeanStd(shape=self.env.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_env)

    def __len__(self):
        return self.num_env

    @property
    def num_envs(self):
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

    def normalize_obs(self, obs):
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                         -self.clip_obs, self.clip_obs)
        return obs

    def normalize_reward(self, reward):
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon),
                           -self.clip_reward, self.clip_reward)
        return reward

    def step(self, actions):
        obs, rews, terminateds, truncateds, infos = self.env.step(actions)

        if self.norm_obs:
            self.obs_rms.update(obs)
            obs = self.normalize_obs(obs)

        if self.norm_reward:
            self.returns = self.returns * self.gamma + rews
            self.ret_rms.update(self.returns)
            rews = self.normalize_reward(rews)

        dones = np.logical_or(terminateds, truncateds)
        self.returns[dones] = 0.0

        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.norm_obs:
            self.obs_rms.update(obs)
            obs = self.normalize_obs(obs)

        return obs, info

    def __getattr__(self, name):
        return getattr(self.env, name)

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

# Wrap environment with VecNormalize
env = VecNormalize(
    env,
    num_env=num_of_envs,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.,
    clip_reward=10.,
    gamma=0.99
)

env_action_space = env.action_space

#-----------------------------
# Custom SAC policy
#----------------------------
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SACSummary:
    loss: float
    loss_actor: float
    loss_critic: float
    loss_alpha: float
    alpha: float
    train_time: float
    def get_loss_stats_dict(self):
        return {
            "loss": self.loss,
            "loss_actor": self.loss_actor,
            "loss_critic": self.loss_critic,
            "loss_alpha": self.loss_alpha,
            "alpha": self.alpha
        }

import copy


class CustomSACPolicy(SACPolicy):
    def __init__(
        self,
        actor,
        critic,
        actor_optim,
        critic_optim,
        action_space=None,
        tau=0.005,
        gamma=0.99,
        alpha=2.0,  # Directly using alpha instead of log_alpha
        **kwargs
    ):
        super().__init__(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic,
            critic_optim=critic_optim,
            action_space=action_space,
            tau=tau,
            gamma=gamma,
            alpha=alpha,  # Directly passing alpha
            **kwargs
        )

        # Directly use alpha as a trainable parameter
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32, device=device), requires_grad=True)
        self.target_entropy = -np.prod(action_space.shape).item()

        # Optimizer for alpha (no log transformation)
        self.alpha_optim = torch.optim.Adam([self.alpha], lr=1e-4)

        # Learning rate scheduler for alpha
        self.alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.alpha_optim, T_max=1000000, eta_min=1e-5
        )

        self.critic_target = copy.deepcopy(critic)
        self.scaler = GradScaler()

    @property
    def get_alpha(self):
        return self.alpha.clamp_min(1e-5)  

    def forward(self, batch: Batch, state=None, **kwargs):
        obs = batch.obs
        loc, scale, h = self.actor(obs, state=state)
        dist = Independent(Normal(loc, scale), 1)
        act = dist.rsample()
        log_prob = dist.log_prob(act)

        # Apply tanh squashing
        act = torch.tanh(act)
        log_prob = log_prob - torch.sum(torch.log(1 - act.pow(2) + 1e-6), dim=-1)

        return Batch(act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch: Batch, **kwargs):
        start_time = time.time()

        # Convert batch data to tensors
        for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
            if isinstance(getattr(batch, key), np.ndarray):
                setattr(batch, key, torch.as_tensor(getattr(batch, key), dtype=torch.float32).to(device))

        self.training = True
        
        # Temperature for Q-value scaling
        temp = 0.1  

        with autocast(device_type="cuda"):
            # Actor forward pass
            loc, scale, _ = self.actor(batch.obs)
            dist = Independent(Normal(loc, scale), 1)
            act_pred = dist.rsample()
            log_prob = dist.log_prob(act_pred)
            act_pred = torch.tanh(act_pred)
            log_prob = log_prob - torch.sum(torch.log(1 - act_pred.pow(2) + 1e-6), dim=-1)

            # Current Q values with temperature scaling
            current_q1, curr_taus1 = self.critic(batch.obs, act_pred, return_taus=True)
            current_q2, curr_taus2 = self.critic(batch.obs, act_pred, return_taus=True)
            
            if current_q1.dim() == 3:
                current_q1 = current_q1.squeeze(-1)
            if current_q2.dim() == 3:
                current_q2 = current_q2.squeeze(-1)
                
            q_min = torch.min(current_q1, current_q2).mean(dim=-1) * temp
            actor_loss = (self.get_alpha * log_prob - q_min).mean()

            # Actor optimization with gradient clipping
            self.actor_optim.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optim)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.scaler.step(self.actor_optim)

            # Alpha optimization with adjusted target entropy
            target_entropy = -0.5 * np.prod(self.actor.action_dim)  # Less negative target entropy
            alpha_loss = -(self.alpha * (log_prob.detach() + self.target_entropy)).mean()
            
            self.alpha_optim.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.unscale_(self.alpha_optim)
            torch.nn.utils.clip_grad_norm_([self.alpha], max_norm=0.5)
            self.scaler.step(self.alpha_optim)
            self.scaler.update()

            # Target Q computation with temperature scaling
            with torch.no_grad():
                next_loc, next_scale, _ = self.actor(batch.obs_next)
                next_dist = Independent(Normal(next_loc, next_scale), 1)
                next_act = next_dist.rsample()
                next_act = torch.tanh(next_act)
                next_log_prob = next_dist.log_prob(next_act)
                next_log_prob = next_log_prob - torch.sum(torch.log(1 - next_act.pow(2) + 1e-6), dim=-1)

                target_q1, target_taus = self.critic_target(batch.obs_next, next_act, return_taus=True)
                target_q2, _ = self.critic_target(batch.obs_next, next_act, return_taus=True)
                
                if target_q1.dim() == 3:
                    target_q1 = target_q1.squeeze(-1)
                if target_q2.dim() == 3:
                    target_q2 = target_q2.squeeze(-1)
                    
                target_q = torch.min(target_q1, target_q2) * temp
                target_q = target_q - (self.get_alpha.detach() * next_log_prob.unsqueeze(1))
                
                reward_scale = 0.1
                target_q = batch.rew.unsqueeze(1) * reward_scale + self.gamma * (1 - batch.done.unsqueeze(1)) * target_q
                target_q = torch.clamp(target_q, -10.0, 10.0)  # Tighter clipping

            # Current Q values for critic loss
            current_q1, curr_taus1 = self.critic(batch.obs, batch.act, return_taus=True)
            current_q2, curr_taus2 = self.critic(batch.obs, batch.act, return_taus=True)

            if current_q1.dim() == 3:
                current_q1 = current_q1.squeeze(-1)
            if current_q2.dim() == 3:
                current_q2 = current_q2.squeeze(-1)

            # Compute critic losses with Huber loss
            critic_loss1 = quantile_huber_loss(current_q1, target_q, curr_taus1, kappa=1.0)
            critic_loss2 = quantile_huber_loss(current_q2, target_q, curr_taus2, kappa=1.0)
            critic_loss = critic_loss1 + critic_loss2

            # Critic optimization with gradient clipping
            self.critic_optim.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optim)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.scaler.step(self.critic_optim)
            self.scaler.update()

            # Soft update of target network with rate limiting
            self.soft_update(self.critic_target, self.critic, min(self.tau, 0.005))

        mean_q1 = current_q1.mean().item()
        mean_q2 = current_q2.mean().item()
        mean_target_q = target_q.mean().item()

        print("\nDetailed Training Stats:")
        print(f"Actor Loss: {actor_loss.item():.6f}")
        print(f"Critic Loss 1: {critic_loss1.item():.6f}")
        print(f"Critic Loss 2: {critic_loss2.item():.6f}")
        print(f"Total Critic Loss: {critic_loss.item():.6f}")
        print(f"Alpha Loss: {alpha_loss.item():.6f}")
        print(f"Alpha: {self.get_alpha.item():.6f}")
        print(f"Mean Q1: {mean_q1:.6f}")
        print(f"Mean Q2: {mean_q2:.6f}")
        print(f"Mean Target Q: {mean_target_q:.6f}")
        print(f"Log Prob Mean: {log_prob.mean().item():.6f}")
        print("-" * 50)

        return SACSummary(
            loss=critic_loss.item() + actor_loss.item(),
            loss_actor=actor_loss.item(),
            loss_critic=critic_loss.item(),
            loss_alpha=alpha_loss.item(),
            alpha=self.get_alpha.item(),
            train_time=time.time() - start_time
        )
# ---------------------------
# Custom Models for SAC + IQN
# ---------------------------

class RecurrentActor(nn.Module):
    def __init__(self, state_dim=2420, action_dim=12, hidden_dim=64, gru_hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers
        self.feature_dim = 242
        self.time_steps = 10
        self.max_action = 1
        self.position_dim = 18
        self.trade_dim = 6  # Keep original trade_dim
        self.market_dim = 218
        self.action_dim = action_dim

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

        self.gru = nn.GRU(64, gru_hidden_dim, num_layers=num_layers, batch_first=True)
        self.fusion_fc = nn.Sequential(
            nn.Linear(gru_hidden_dim + 32, hidden_dim * 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.log_std.weight.data.uniform_(-3, -2)
        self.log_std.bias.data.uniform_(-3, -2)

    def forward(self, obs, state=None, info=None):
        if isinstance(obs, Batch):
            obs = obs.obs

        obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        batch_size = obs.shape[0]

        # Reshape the input to (batch_size, time_steps, feature_dim)
        obs = obs.view(batch_size, self.time_steps, -1)

        # Split the features
        market_state = obs[:, :, :self.market_dim]
        position_state = obs[:, -1, self.market_dim:self.market_dim + self.position_dim]
        trade_state = obs[:, -1, self.market_dim + self.position_dim:self.market_dim + self.position_dim + self.trade_dim]

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state[:, -1])  # Use only the last timestep

        x = torch.cat([trade_out, market_out], dim=-1)

        if state is None:
            state = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=obs.device)
        elif isinstance(state, list):
            state = state[0]

        if state.shape == (batch_size, self.num_layers, self.gru_hidden_dim):
            state = state.transpose(0, 1).contiguous()

        x = x.unsqueeze(1)  # Add time dimension for GRU
        x, new_state = self.gru(x, state)
        x = x[:, -1, :]
        x = torch.cat([x, position_out], dim=-1)
        x = self.fusion_fc(x)

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp() + 1e-6

        loc = mean
        scale = std

        if loc.dim() == 1:
            loc = loc.unsqueeze(0)
        if scale.dim() == 1:
            scale = scale.unsqueeze(0)

        return loc, scale, new_state.detach().transpose(0, 1)


class IQNCritic(nn.Module):
    def __init__(self, state_dim=2420, action_dim=12, hidden_dim=128, num_quantiles=32, gru_hidden_dim=128, num_layers=2):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.num_layers = num_layers
        self.gru_hidden_dim = gru_hidden_dim
        self.feature_dim = 242
        self.time_steps = 10
        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = 218  

        # Initialize layers first
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

        self.gru = nn.GRU(64, gru_hidden_dim, num_layers=num_layers, batch_first=True)

        self.fusion_fc = nn.Sequential(
            nn.Linear(gru_hidden_dim + 32 + action_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.q_values = nn.Sequential(
            nn.Linear(hidden_dim, num_quantiles),
            nn.LayerNorm(num_quantiles)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, state, action, taus=None, return_taus=False):
        device = next(self.parameters()).device

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device)

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        
        batch_size = state.shape[0]
        if action.shape[0] != batch_size:
            action = action.expand(batch_size, -1)

        expected_flat_dim = self.time_steps * self.feature_dim
        if state.dim() == 2 and state.shape[1] == expected_flat_dim:
            state = state.view(batch_size, self.time_steps, self.feature_dim)

        market_state = state[:, :, :self.market_dim]
        position_state = state[:, -1, self.market_dim:self.market_dim + self.position_dim] 
        trade_state = state[:, -1, self.market_dim + self.position_dim:]  # Changed to use last timestep

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state[:, -1])  # Use last timestep for market state too

        x = torch.cat([trade_out, market_out], dim=-1)
        x = x.unsqueeze(1)  # Add time dimension for GRU [batch_size, 1, feature_dim]

        state_h = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=device)
        x, _ = self.gru(x, state_h)
        x = x[:, -1, :]  # Take last timestep output

        x = torch.cat([x, position_out, action], dim=-1)
        x = self.fusion_fc(x)
        
        # Output shape: [batch_size, num_quantiles]
        q_values = self.q_values(x)
        
        if taus is None:
            taus = torch.rand(batch_size, self.num_quantiles, device=device)
        
        return (q_values, taus) if return_taus else q_values


def quantile_huber_loss(ip, target, taus, kappa=1.0):
    # Ensure proper dimensions
    if ip.dim() == 3:
        ip = ip.squeeze(-1)  # [batch_size, num_quantiles]
    if target.dim() == 3:
        target = target.squeeze(-1)  # [batch_size, num_quantiles]
    elif target.dim() == 1:
        target = target.unsqueeze(1)  # [batch_size, 1]
    
    # Now target is [batch_size, 1] or [batch_size, num_quantiles]
    # and ip is [batch_size, num_quantiles]
    if target.shape[1] == 1:
        target = target.expand_as(ip)  # Expand to match ip dimensions
    
    # Calculate TD error
    td_error = target - ip  # [batch_size, num_quantiles]
    
    # Calculate Huber loss
    huber_loss = F.huber_loss(ip, target, delta=kappa, reduction="none")  # [batch_size, num_quantiles]
    
    # Ensure taus has correct shape [batch_size, num_quantiles]
    if taus.dim() == 3:
        taus = taus.squeeze(-1)
    
    # Calculate quantile weights
    quantile_weight = (taus - (td_error.detach() < 0).float()).abs()
    
    # Apply asymmetric loss
    asymmetric_factor = 1.2
    weighted_loss = torch.where(td_error < 0,
                              huber_loss * asymmetric_factor * quantile_weight,
                              huber_loss * quantile_weight)
    
    return weighted_loss.mean()


from tianshou.data import Collector, Batch
from collections import namedtuple
import time
from dataclasses import dataclass

class MetricLogger:
    def __init__(self, print_interval=1000):
        self.print_interval = print_interval
        self.last_print_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def log(self, step_count, info, rew, policy):
        if step_count % self.print_interval == 0:
            print(f"\nStep: {step_count}")
            print("Env | Net_PnL   | R_PnL     | UR_PnL    | Fees       | Trades   | Drawdown | Leverage | Reward")
            print("-" * 80)
            
            # Print one line per environment
            for ii in info['env_id']:
                net_pnl = info['realized_pnl'][ii] + info['unrealized_pnl'][ii] - info['fees'][ii]
                print(f"{ii:3d} | {net_pnl:+7.6f} | {info['realized_pnl'][ii]:+6.6f} | "
                      f"{info['unrealized_pnl'][ii]:+6.6f} | {info['fees'][ii]:+6.6f} | "
                      f"{info['trade_count'][ii]} | {info['drawdown'][ii]:+8.6f} | "
                      f"{info['leverage'][ii]:+8.6f} | {rew[ii]:+8.6f}")
            
            if hasattr(policy, 'get_alpha'):
                print(f"\nAlpha: {policy.get_alpha.item():.6f}")
            print("-" * 80)

@dataclass
class CollectStats:
    n_ep: int
    n_st: int
    rews: np.ndarray
    lens: np.ndarray
    rew: float
    len: float
    rew_std: float
    len_std: float
    n_collected_steps: int
    n_collected_episodes: int
    returns_stat: object
    lens_stat: object
    rewards_stat: object
    episodes: int
    reward_sum: float
    length_sum: int
    collect_time: float
    step_time: float
    returns: np.ndarray
    lengths: np.ndarray
    continuous_step: int

class GPUCollector(Collector):
    def __init__(self, policy, env, buffer=None, preprocess_fn=None, device='cuda', **kwargs):
        super().__init__(policy, env, buffer, **kwargs)
        self.device = device
        self.preprocess_fn = preprocess_fn
        self.env_active = False  # Track if environments are already running
        self.continuous_step_count = 0
        self.data = Batch()
        self.reset_batch_data()

    def reset_batch_data(self):
        """Reset the internal batch data but maintain environment state"""
        if not self.env_active:
            self.data.obs_next = None
            self.data.obs = None
        self.data.act = None
        self.data.rew = None
        self.data.done = None
        self.data.terminated = None
        self.data.truncated = None
        self.data.info = None
        self.data.policy = Batch()
        self.data.state = None

    def reset_env(self, gym_reset_kwargs=None):
        """Reset the environment only if not already active"""
        if not self.env_active:
            gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
            obs = self.env.reset(**gym_reset_kwargs)
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = None

            if isinstance(obs, torch.Tensor):
                obs = obs.to(self.device)
            else:
                obs = torch.as_tensor(obs, device=self.device)

            self.data.obs = obs
            self.data.info = info
            self.env_active = True

        return self.data.obs, self.data.info

    def _collect(self, n_step=None, n_episode=None, random=False, render=None, gym_reset_kwargs=None):
        if n_step is not None:
            assert n_episode is None, "Only one of n_step or n_episode is allowed"
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        
        start_time = time.time()
        local_step_count = 0  # Local counter for this collection cycle
        episode_count = 0
        episode_rews = []
        episode_lens = []

        while True:
            if self.data.obs is None:
                obs, info = self.reset_env()
                self.data.obs = obs
                self.data.info = info

            # Convert observation to tensor on device and wrap in Batch
            if isinstance(self.data.obs, torch.Tensor):
                obs_batch = Batch(obs=self.data.obs.to(self.device))
            else:
                obs_batch = Batch(obs=torch.as_tensor(self.data.obs, device=self.device))

            with torch.no_grad():
                result = self.policy(obs_batch, state=self.data.state)

            self.data.act = result.act
            self.data.state = result.state if hasattr(result, 'state') else None

            # Convert action to numpy array before stepping
            if isinstance(self.data.act, torch.Tensor):
                action = self.data.act.cpu().numpy()
            else:
                action = np.array(self.data.act)
            
            # Step the environment with numpy action
            result = self.env.step(action)
            if len(result) == 5:  # gymnasium style
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            else:  # old gym style
                obs_next, rew, done, info = result
                terminated = done
                truncated = False
            
            if hasattr(self, 'logger'):
                self.logger.log(self.continuous_step_count, info, rew, self.policy)

            if isinstance(obs_next, torch.Tensor):
                obs_next = obs_next.to(self.device)
            else:
                obs_next = torch.as_tensor(obs_next, device=self.device)
                
            if isinstance(rew, torch.Tensor):
                rew = rew.to(self.device)
            else:
                rew = torch.as_tensor(rew, device=self.device)

            self.data.obs_next = obs_next
            self.data.rew = rew
            self.data.done = done
            self.data.terminated = terminated
            self.data.truncated = truncated
            self.data.info = info

            if self.preprocess_fn:
                self.data = self.preprocess_fn(self.data)

            # Convert tensors to numpy for buffer
            if isinstance(self.data.obs, torch.Tensor):
                obs = self.data.obs.cpu().numpy()
            else:
                obs = self.data.obs

            if isinstance(self.data.act, torch.Tensor):
                act = self.data.act.cpu().numpy()
            else:
                act = self.data.act

            if isinstance(self.data.rew, torch.Tensor):
                rew = self.data.rew.cpu().numpy()
            else:
                rew = self.data.rew

            if isinstance(self.data.obs_next, torch.Tensor):
                obs_next = self.data.obs_next.cpu().numpy()
            else:
                obs_next = self.data.obs_next

            batch = Batch(
                obs=obs,
                act=act,
                rew=rew,
                done=self.data.done,
                terminated=self.data.terminated,
                truncated=self.data.truncated,
                obs_next=obs_next,
                info=self.data.info,
                policy=self.data.policy if hasattr(self.data, 'policy') else None,
                state=self.data.state if hasattr(self.data, 'state') else None
            )

            # Save to buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(batch)

            local_step_count += 1
            self.continuous_step_count += 1  # Update continuous counter

            # Handle vectorized environment episode completion
            if isinstance(ep_len, np.ndarray):
                for idx, (l, r) in enumerate(zip(ep_len, ep_rew)):
                    if l is not None:  # episode ended
                        episode_lens.append(l)
                        episode_rews.append(r)
                        episode_count += 1
                        if n_episode and episode_count >= n_episode:
                            break
            else:
                if ep_len is not None:  # single environment episode ended
                    episode_lens.append(ep_len)
                    episode_rews.append(ep_rew)
                    episode_count += 1

            if n_episode and episode_count >= n_episode:
                break

            if n_step and local_step_count >= n_step:
                break

            self.data.obs = self.data.obs_next

        class StatClass:
            def __init__(self, mean_val, std_val):
                self.mean = mean_val
                self._std = std_val
            
            def std(self):
                return self._std

        if episode_count > 0:
            rews, lens = list(map(np.array, [episode_rews, episode_lens]))
            mean_rew = rews.mean()
            mean_len = lens.mean()
            std_rew = rews.std()
            std_len = lens.std()
            
            # Basic statistics with mean and std
            return_stat = StatClass(mean_rew, std_rew)
            return_stat.n_ep = episode_count
            return_stat.n_st = local_step_count
            return_stat.rews = rews
            return_stat.lens = lens
            return_stat.rew = mean_rew
            return_stat.len = mean_len
            return_stat.rew_std = std_rew
            return_stat.len_std = std_len
            
            # Length statistics
            lens_stat = StatClass(mean_len, std_len)
            lens_stat.n_ep = episode_count
            lens_stat.n_st = local_step_count
            lens_stat.lens = lens
            lens_stat.len = mean_len
            lens_stat.len_std = std_len
            
            # Reward statistics
            rewards_stat = StatClass(mean_rew, std_rew)
            rewards_stat.n_ep = episode_count
            rewards_stat.n_st = local_step_count
            rewards_stat.rews = rews
            rewards_stat.rew = mean_rew
            rewards_stat.rew_std = std_rew
        else:
            empty_arr = np.array([])
            return_stat = StatClass(0.0, 0.0)
            return_stat.n_ep = episode_count
            return_stat.n_st = local_step_count
            
            lens_stat = StatClass(0.0, 0.0)
            lens_stat.n_ep = episode_count
            lens_stat.n_st = local_step_count
            lens_stat.lens = empty_arr
            lens_stat.len = 0.0
            lens_stat.len_std = 0.0
            
            rewards_stat = StatClass(0.0, 0.0)
            rewards_stat.n_ep = episode_count
            rewards_stat.n_st = local_step_count
            rewards_stat.rews = empty_arr
            rewards_stat.rew = 0.0
            rewards_stat.rew_std = 0.0

        collect_time = time.time() - start_time
        step_time = collect_time / local_step_count if local_step_count else 0

        return CollectStats(
            n_ep=episode_count,
            n_st=local_step_count,
            rews=np.array(episode_rews) if episode_rews else empty_arr,
            lens=np.array(episode_lens) if episode_lens else empty_arr,
            rew=mean_rew if episode_count > 0 else 0.0,
            len=mean_len if episode_count > 0 else 0.0,
            rew_std=std_rew if episode_count > 0 else 0.0,
            len_std=std_len if episode_count > 0 else 0.0,
            n_collected_steps=local_step_count,
            n_collected_episodes=episode_count,
            returns_stat=return_stat,
            lens_stat=lens_stat,
            rewards_stat=rewards_stat,
            episodes=episode_count,
            reward_sum=float(np.sum(episode_rews)) if episode_rews else 0.0,
            length_sum=int(np.sum(episode_lens)) if episode_lens else 0,
            collect_time=collect_time,
            step_time=step_time,
            returns=np.array(episode_rews) if episode_rews else empty_arr,
            lengths=np.array(episode_lens) if episode_lens else empty_arr,
            continuous_step=self.continuous_step_count
        )

def save_checkpoint_fn(epoch, env_step, gradient_step):
    try:
        checkpoint_dir = results_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Save model checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{env_step}.pth"
        torch.save({
            'epoch': epoch,
            'env_step': env_step,
            'gradient_step': gradient_step,
            'policy_state_dict': policy.state_dict(),
            'alpha_optim_state_dict': policy.alpha_optim.state_dict(),
            'actor_optim_state_dict': policy.actor_optim.state_dict(),
            'critic_optim_state_dict': policy.critic_optim.state_dict(),
            'alpha_optim_state_dict': policy.alpha_optim.state_dict()
        }, checkpoint_path)

        # Save buffer at epoch intervals
        if env_step % (7201*num_of_envs) == 0:  # Save every epoch
            buffer_path = checkpoint_dir / f"buffer_epoch_{epoch}_step_{env_step}.h5"
            buffer.save_hdf5(buffer_path)

        print(f"Saved checkpoint at epoch {epoch}, step {env_step}")
        return True

    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False


# ---------------------------
# Training Setup
# ---------------------------
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Initialize models and optimizers
actor = RecurrentActor().to(device)
critic = IQNCritic().to(device)
critic_optim = Adam(critic.parameters(), lr=1e-4)

policy = CustomSACPolicy(
    actor=actor,
    critic=critic,
    actor_optim=Adam(actor.parameters(), lr=1e-4),
    critic_optim=critic_optim,
    tau=0.005, gamma=0.99, alpha=2.0,
    action_space=env_action_space
)

policy = policy.to(device)

# Load model if exists
final_checkpoint_path = results_dir / "final_model.pth"
final_buffer_path = results_dir / "final_buffer.h5"
start_epoch = 0

if final_checkpoint_path.exists():
    print(f"Loading model from {final_checkpoint_path}")
    saved_model = torch.load(final_checkpoint_path)
    policy.load_state_dict(saved_model['policy_state_dict'])
    policy.alpha_optim.load_state_dict(saved_model['alpha_optim_state_dict'])
    policy.actor_optim.load_state_dict(saved_model['actor_optim_state_dict'])
    policy.critic_optim.load_state_dict(saved_model['critic_optim_state_dict'])
    start_epoch = saved_model.get('epoch', 0)
    print(f"Resumed from epoch {start_epoch}")
else:
    print(f"Could not find model {final_checkpoint_path}")
    
if final_buffer_path.exists():
    print(f"Loading buffer from {final_buffer_path}")
    buffer = VectorReplayBuffer.load_hdf5(final_buffer_path, device=device)
    print(f"Buffer loaded with {len(buffer)} transitions")
else:
    print(f"Could not find buffer {final_buffer_path}")
    buffer = VectorReplayBuffer(
        total_size=100000,
        buffer_num=num_of_envs,
        device=device
    )

logger = MetricLogger(print_interval=1000)
collector = GPUCollector(policy, env, buffer, device=device, exploration_noise=True)
collector.logger = logger

trainer = OffpolicyTrainer(
    policy=policy,
    train_collector=collector,
    max_epoch=30,
    step_per_epoch=72,
    step_per_collect=32*16,
    update_per_step=0.01,
    episode_per_test=0,
    batch_size=num_of_envs,
    test_in_train=False,
    verbose=True,
    save_checkpoint_fn=save_checkpoint_fn,
    resume_from_log=True
)

trainer.run()

# Save final results
torch.save({
    'policy_state_dict': policy.state_dict(),
    'actor_optim_state_dict': policy.actor_optim.state_dict(),
    'critic_optim_state_dict': policy.critic_optim.state_dict(),
    'alpha_optim_state_dict': policy.alpha_optim.state_dict()
}, final_checkpoint_path)

buffer.save_hdf5(final_buffer_path)
