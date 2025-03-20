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
env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=32, batch_size=32,
    num_threads=32, is_prod=False, is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL", tick_size=0.5, min_amount=10,
    maker_fee=-0.0001, taker_fee=0.0005, foldername="./train_files/",
    balance=1.0, start=1, max=36001*10
)

env.spec.id = 'RlTrader-v0'
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
        alpha=0.2, 
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
            alpha=alpha,
            **kwargs
        )
        
        self.target_entropy = -np.prod(action_space.shape).item()
        self.alpha = nn.Parameter(torch.tensor([alpha], device=device))
        self.alpha_optim = torch.optim.Adam([self.alpha], lr=3e-4)
        self.critic_target = copy.deepcopy(critic)
        self.scaler = GradScaler()

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
        batch.to_torch(device=device)

        for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
            if isinstance(getattr(batch, key), np.ndarray):
                setattr(batch, key, torch.as_tensor(getattr(batch, key)).to(device))

        obs = batch.obs
        obs_next = batch.obs_next
        act = batch.act
        rew = batch.rew
        done = batch.done

        self.training = True
        start_time = time.time()

        with autocast(device_type="cuda"):
            loc, scale, _ = self.actor(obs)
            dist = Independent(Normal(loc, scale), 1)
            act_pred = dist.rsample()
            log_prob = dist.log_prob(act_pred)
            act_pred = torch.tanh(act_pred)
            log_prob = log_prob - torch.sum(torch.log(1 - act_pred.pow(2) + 1e-6), dim=-1)

            current_q1 = self.critic(obs, act_pred)
            current_q2 = self.critic(obs, act_pred)
            q_min = torch.min(current_q1, current_q2).mean(dim=-1)
            actor_loss = (self.alpha * log_prob - q_min).mean()
            batch.to_torch(device=device)
            self.training = True
            start_time = time.time()


        # Actor optimization with scaler
        self.actor_optim.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optim)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.scaler.step(self.actor_optim)

        # Alpha update
        with autocast(device_type="cuda"):
            alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_optim)

        # Critic update
        with autocast(device_type="cuda"):
            with torch.no_grad():
                next_loc, next_scale, _ = self.actor(obs_next)
                next_dist = Independent(Normal(next_loc, next_scale), 1)
                next_act = next_dist.rsample()
                next_act = torch.tanh(next_act)
                next_log_prob = next_dist.log_prob(next_act)
                next_log_prob = next_log_prob - torch.sum(torch.log(1 - next_act.pow(2) + 1e-6), dim=-1)

                target_q = self.critic_target(obs_next, next_act)
                target_q = target_q - self.alpha.detach() * next_log_prob.unsqueeze(-1)
                target_q = rew.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * target_q

            current_q1 = self.critic(obs, act)
            current_q2 = self.critic(obs, act)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optim)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.scaler.step(self.critic_optim)
        self.scaler.update()
        

        self.soft_update(self.critic_target, self.critic, self.tau)

        loss = critic_loss.item() + actor_loss.item()
        
        return SACSummary(
            loss=loss,
            loss_actor=actor_loss.item(),
            loss_critic=critic_loss.item(),
            loss_alpha=alpha_loss.item(),
            alpha=self.alpha.item(),
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
            nn.Linear(gru_hidden_dim + 32 + action_dim, hidden_dim * 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU()
        )

        self.q_values = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, state, action):
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
        trade_state = state[:, :, self.market_dim + self.position_dim:]

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)

        x = torch.cat([trade_out, market_out], dim=-1)

        state_h = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=device)
        x, _ = self.gru(x, state_h)
        x = x[:, -1, :]

        x = torch.cat([x, position_out, action], dim=-1)
        x = self.fusion_fc(x)
        return self.q_values(x)


def quantile_huber_loss(ip, target, taus, kappa=1.0):
    target = target.unsqueeze(-1).expand_as(ip)  
    td_error = target - ip  
    huber_loss = F.huber_loss(ip, target, delta=kappa, reduction="none")
    loss = (taus - (td_error.detach() < 0).float()).abs() * huber_loss
    return loss.mean()

# ---------------------------
# Training Setup
# ---------------------------
actor = RecurrentActor().to(device)
critic = IQNCritic().to(device)
critic_optim = Adam(critic.parameters(), lr=3e-4)

policy = CustomSACPolicy(
    actor=actor,
    critic=critic,
    actor_optim=Adam(actor.parameters(), lr=3e-4),
    critic_optim=critic_optim,
    tau=0.005, gamma=0.99, alpha=0.2,
    action_space=env_action_space
)

policy = policy.to(device)

buffer = VectorReplayBuffer(
    total_size=1000000,  # 1M transitions in total
    buffer_num=32,       # Number of environments
    device=device
)

from tianshou.data import Collector, Batch
from collections import namedtuple
import time
from dataclasses import dataclass

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

class GPUCollector(Collector):
    def __init__(self, policy, env, buffer=None, preprocess_fn=None, device='cuda', **kwargs):
        super().__init__(policy, env, buffer, **kwargs)
        self.device = device
        self.preprocess_fn = preprocess_fn
        self.data = Batch()
        self.reset_batch_data()

    def reset_batch_data(self):
        """Reset the internal batch data"""
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
        """Reset the environment and return initial observation."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs = self.env.reset(**gym_reset_kwargs)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = None
        
        # Convert observation to tensor on device
        if isinstance(obs, torch.Tensor):
            obs = obs.to(self.device)
        else:
            obs = torch.as_tensor(obs, device=self.device)
            
        self.reset_batch_data()
        return obs, info

    def _collect(self, n_step=None, n_episode=None, random=False, render=None, gym_reset_kwargs=None):
        if n_step is not None:
            assert n_episode is None, "Only one of n_step or n_episode is allowed"
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        start_time = time.time()
        step_count = 0
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
                
            # Convert new observations and rewards to tensors on device
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

            step_count += 1
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

            if n_step and step_count >= n_step:
                break

            self.data.obs = self.data.obs_next

        self.reset_batch_data()

        collect_time = time.time() - start_time
        step_time = collect_time / step_count if step_count else 0

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
            return_stat.n_st = step_count
            return_stat.rews = rews
            return_stat.lens = lens
            return_stat.rew = mean_rew
            return_stat.len = mean_len
            return_stat.rew_std = std_rew
            return_stat.len_std = std_len
            
            # Length statistics
            lens_stat = StatClass(mean_len, std_len)
            lens_stat.n_ep = episode_count
            lens_stat.n_st = step_count
            lens_stat.lens = lens
            lens_stat.len = mean_len
            lens_stat.len_std = std_len
            
            # Reward statistics
            rewards_stat = StatClass(mean_rew, std_rew)
            rewards_stat.n_ep = episode_count
            rewards_stat.n_st = step_count
            rewards_stat.rews = rews
            rewards_stat.rew = mean_rew
            rewards_stat.rew_std = std_rew

            return CollectStats(
                n_ep=episode_count,
                n_st=step_count,
                rews=rews,
                lens=lens,
                rew=mean_rew,
                len=mean_len,
                rew_std=std_rew,
                len_std=std_len,
                n_collected_steps=step_count,
                n_collected_episodes=episode_count,
                returns_stat=return_stat,
                lens_stat=lens_stat,
                rewards_stat=rewards_stat,
                episodes=episode_count,
                reward_sum=float(rews.sum()),
                length_sum=int(lens.sum()),
                collect_time=collect_time,
                step_time=step_time,
                returns=rews,
                lengths=lens
            )
        else:
            empty_arr = np.array([])
            
            return_stat = StatClass(0.0, 0.0)
            return_stat.n_ep = episode_count
            return_stat.n_st = step_count
            
            lens_stat = StatClass(0.0, 0.0)
            lens_stat.n_ep = episode_count
            lens_stat.n_st = step_count
            lens_stat.lens = empty_arr
            lens_stat.len = 0.0
            lens_stat.len_std = 0.0
            
            rewards_stat = StatClass(0.0, 0.0)
            rewards_stat.n_ep = episode_count
            rewards_stat.n_st = step_count
            rewards_stat.rews = empty_arr
            rewards_stat.rew = 0.0
            rewards_stat.rew_std = 0.0

            return CollectStats(
                n_ep=episode_count,
                n_st=step_count,
                rews=empty_arr,
                lens=empty_arr,
                rew=0.0,
                len=0.0,
                rew_std=0.0,
                len_std=0.0,
                n_collected_steps=step_count,
                n_collected_episodes=episode_count,
                returns_stat=return_stat,
                lens_stat=lens_stat,
                rewards_stat=rewards_stat,
                episodes=episode_count,
                reward_sum=0.0,
                length_sum=0,
                collect_time=collect_time,
                step_time=step_time,
                returns=empty_arr,
                lengths=empty_arr
            )

collector = GPUCollector(policy, env, buffer, device=device, exploration_noise=True)


trainer = OffpolicyTrainer(
    policy=policy,
    train_collector=collector,
    max_epoch=10,
    step_per_epoch=36002*32,
    step_per_collect=360*32,  # Collect 360 steps per environment
    update_per_step=0.1,
    episode_per_test=0,
    batch_size=32,  
    test_in_train=False,
    verbose=True
)

trainer.run()

# Save results
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

torch.save({
    'policy_state_dict': policy.state_dict(),
    'actor_optim_state_dict': policy.actor_optim.state_dict(),
    'critic_optim_state_dict': policy.critic_optim.state_dict(),
    'alpha_optim_state_dict': policy.alpha_optim.state_dict()
}, results_dir / "sac_iqn_rltrader.pth")

buffer.save_hdf5(results_dir / "replay_iqn_buffer.h5")
