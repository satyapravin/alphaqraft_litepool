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

class VecNormalize:
    def __init__(self, env, num_envs, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        self.env = env
        self.num_envs = num_envs
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        self.time_steps = 10
        self.feature_dim = 242

        self.obs_rms = RunningMeanStd(shape=(self.feature_dim,))
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = torch.zeros(self.num_envs, device=device)

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

    def normalize_obs(self, obs):
        if not self.norm_obs:
            return obs

        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        batch_size = obs.shape[0]

        # Reshape to [B, T, F]
        obs = obs.view(batch_size, self.time_steps, self.feature_dim)
        flat_obs = obs.view(-1, self.feature_dim)

        self.obs_rms.update(flat_obs)
        normed = (flat_obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)
        normed = torch.clamp(normed, -self.clip_obs, self.clip_obs)

        normed = normed.view(batch_size, self.time_steps, self.feature_dim)
        return normed.view(batch_size, -1)  # Flatten back to [B, 2420]

    def normalize_reward(self, reward):
        if not self.norm_reward:
            return reward
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        self.ret_rms.update(self.returns)
        normed = reward / torch.sqrt(self.ret_rms.var + self.epsilon)
        return torch.clamp(normed, -self.clip_reward, self.clip_reward)

    def step(self, actions):
        obs, rews, terminateds, truncateds, infos = self.env.step(actions)

        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        rews = torch.as_tensor(rews, dtype=torch.float32, device=device)

        self.returns = self.returns * self.gamma + rews

        obs = self.normalize_obs(obs)
        rews = self.normalize_reward(rews)

        dones = torch.logical_or(torch.as_tensor(terminateds), torch.as_tensor(truncateds))
        self.returns[dones] = 0.0

        return obs.cpu().numpy(), rews.cpu().numpy(), terminateds, truncateds, infos

    def reset(self, indices=None, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        self.returns.zero_()
        obs = self.normalize_obs(obs)
        return obs.cpu().numpy(), info

    def __getattr__(self, name):
        return getattr(self.env, name)

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
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
        self.alpha_optim = torch.optim.Adam([self.alpha], lr=1e-3)

        # Learning rate scheduler for alpha
        self.alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.alpha_optim, T_max=1000000, eta_min=1e-3
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
        temp = 1.0  

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

                # Sample new taus for target Q
                taus_target = torch.rand(batch.obs.shape[0], self.critic.num_quantiles, device=device)
                target_q1, target_taus1 = self.critic_target(batch.obs_next, next_act, taus=taus_target, return_taus=True)
                target_q2, target_taus2 = self.critic_target(batch.obs_next, next_act, taus=taus_target, return_taus=True)

                target_q = torch.min(target_q1, target_q2) * temp
                target_q = target_q - (self.get_alpha.detach() * next_log_prob.unsqueeze(1))

                reward_scale = 1.0
                target_q = batch.rew.unsqueeze(1) * reward_scale + self.gamma * (1 - batch.done.unsqueeze(1)) * target_q
                target_q = torch.clamp(target_q, -10.0, 10.0) 

            # Current Q values for critic loss
            current_q1, curr_taus1 = self.critic(batch.obs, batch.act, return_taus=True)
            current_q2, curr_taus2 = self.critic(batch.obs, batch.act, return_taus=True)

            if current_q1.dim() == 3:
                current_q1 = current_q1.squeeze(-1)
            if current_q2.dim() == 3:
                current_q2 = current_q2.squeeze(-1)

            # Compute critic losses with Huber loss
            critic_loss1 = quantile_huber_loss(current_q1, target_q, curr_taus1, target_taus1, kappa=1.0)
            critic_loss2 = quantile_huber_loss(current_q2, target_q, curr_taus2, target_taus2, kappa=1.0)
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
    def __init__(self, state_dim=2420, action_dim=3, hidden_dim=64, gru_hidden_dim=128, num_layers=2):
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

        self.log_std.weight.data.uniform_(-1, 0)
        self.log_std.bias.data.uniform_(-1, 0)

    def forward(self, obs, state=None, info=None):
        if isinstance(obs, Batch):
            obs = obs.obs

        obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        batch_size = obs.shape[0]

        # Reshape the input to (batch_size, time_steps, feature_dim)
        obs = obs.view(batch_size, self.time_steps, -1)

        # Split the features
        market_state = obs[:, :, :self.market_dim]
        position_state = obs[:, :, self.market_dim:self.market_dim + self.position_dim]
        trade_state = obs[:, :, self.market_dim + self.position_dim:self.market_dim + self.position_dim + self.trade_dim]

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)  

        x = torch.cat([trade_out, market_out], dim=-1)

        if state is None:
            state = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=obs.device)
        elif isinstance(state, list):
            state = state[0]

        if state.shape == (batch_size, self.num_layers, self.gru_hidden_dim):
            state = state.transpose(0, 1).contiguous()

        x, new_state = self.gru(x, state)
        x = x[:, -1, :]
        position_out = position_out[:, -1, :]
        x = torch.cat([x, position_out], dim=-1)
        x = self.fusion_fc(x)

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-10, 2)
        std = log_std.exp() + 1e-6

        loc = mean
        scale = std

        if loc.dim() == 1:
            loc = loc.unsqueeze(0)
        if scale.dim() == 1:
            scale = scale.unsqueeze(0)

        return loc, scale, new_state.detach().transpose(0, 1)


class IQNCritic(nn.Module):
    def __init__(self, state_dim=2420, action_dim=3, hidden_dim=128, num_quantiles=64, gru_hidden_dim=128, num_layers=2):
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
        position_state = state[:, :, self.market_dim:self.market_dim + self.position_dim] 
        trade_state = state[:, :, self.market_dim + self.position_dim:]  # Changed to use last timestep

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)  

        x = torch.cat([trade_out, market_out], dim=-1)
        state_h = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=device)
        x, _ = self.gru(x, state_h)
        x = x[:, -1, :]  # Take last timestep output
        position_out = position_out[:, -1, :]
        x = torch.cat([x, position_out, action], dim=-1)
        x = self.fusion_fc(x)
        
        # Output shape: [batch_size, num_quantiles]
        q_values = self.q_values(x)
        
        if taus is None:
            taus = torch.rand(batch_size, self.num_quantiles, device=device)
        
        return (q_values, taus) if return_taus else q_values


def quantile_huber_loss(pred, target, taus_pred, taus_target, kappa=1.0):
    """
    pred: [B, N] - predicted quantile values (critic output)
    target: [B, M] - target quantile values
    taus_pred: [B, N] - quantile fractions for predicted quantiles
    taus_target: [B, M] - quantile fractions for target quantiles
    """
    B, N = pred.shape
    _, M = target.shape

    # Expand dims for pairwise differences
    pred = pred.unsqueeze(2)        # [B, N, 1]
    target = target.unsqueeze(1)    # [B, 1, M]
    td_error = target - pred        # [B, N, M]

    # Compute Huber loss
    huber = F.smooth_l1_loss(pred.expand(-1, -1, M), target.expand(-1, N, -1), reduction="none")
    
    # Quantile weighting
    taus_pred = taus_pred.unsqueeze(2)  # [B, N, 1]
    weight = torch.abs(taus_pred - (td_error.detach() < 0).float())  # [B, N, M]

    quantile_loss = weight * huber  # [B, N, M]
    return quantile_loss.mean()



torch.manual_seed(42)
np.random.seed(42)

results_dir = Path("results")
model_path = results_dir / "final_model.pth"

env = litepool.make("RlTrader-v0", env_type="gymnasium",
                          num_envs=1, batch_size=1,
                          num_threads=1,
                          is_prod=True,
                          is_inverse_instr=True,
                          api_key="",
                          api_secret="",
                          symbol="BTC-PERPETUAL",
                          tick_size=0.5,
                          min_amount=10,
                          maker_fee=-0.0001,
                          taker_fee=0.0005,
                          foldername="./prodfiles/",
                          balance=0.1,
                          start=1,
                          max=72000001)


env.spec.id = "RlTrader-v0"
env = VecNormalize(
    env,
    num_envs=1,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10000000.,
    clip_reward=1000.,
    gamma=0.99
)

env_action_space = env.action_space


actor = RecurrentActor().to(device)
critic = IQNCritic().to(device)

policy = CustomSACPolicy(
    actor=actor,
    critic=critic,
    actor_optim=torch.optim.Adam(actor.parameters(), lr=1e-3),
    critic_optim=torch.optim.Adam(critic.parameters(), lr=1e-3),
    tau=0.005,
    gamma=0.997,
    alpha=2.0,
    action_space=env.action_space,
).to(device)

# Load the trained model
if model_path.exists():
    print(f"Loading trained model from {model_path}")
    saved_model = torch.load(model_path, map_location=device)
    policy.load_state_dict(saved_model["policy_state_dict"])
    policy.actor_optim.load_state_dict(saved_model["actor_optim_state_dict"])
    policy.critic_optim.load_state_dict(saved_model["critic_optim_state_dict"])
    policy.alpha_optim.load_state_dict(saved_model["alpha_optim_state_dict"])
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Inference function
def run_inference(environment, policy):
    obs, _ = environment.reset()
    done = False
    episode_reward = 0.0
    step_count = 0

    print("Starting inference...")
    while not np.any(done):
        obs_batch = Batch(obs=torch.as_tensor(obs, device=device, dtype=torch.float32))

        with torch.no_grad():
            result = policy(obs_batch)
            action = result.act.cpu().numpy()

        obs, reward, terminated, truncated, info = environment.step(action)
        
        print(f"Balance: {info['balance'][0]:.6f}, "
                          f"Realized PnL: {info['realized_pnl'][0]:.6f}, "
                          f"Unrealized PnL: {info['unrealized_pnl'][0]:.6f}, "
                          f"Fees: {info['fees'][0]:.6f}, "
                          f"Trade Count: {info['trade_count'][0]}, "
                          f"Drawdown: {info['drawdown'][0]:.6f}, "
                          f"Leverage: {info['leverage'][0]:.4f}")
        done = terminated or truncated
        step_count += 1


run_inference(env, policy)
