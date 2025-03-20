import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.optim import Adam
import copy
from pathlib import Path

import litepool
from tianshou.env import BaseVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.cuda.amp import autocast, GradScaler

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


env.spec.id = "RlTrader-v0"
env_action_space = env.action_space

env = gym.wrappers.TransformObservation(
    env, lambda obs: torch.as_tensor(obs, device=device)
)
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
        self.training = True
        start_time = time.time()  # Add this line

        # Convert numpy arrays to tensors
        obs = batch.obs
        obs_next = batch.obs_next
        act = batch.act
        rew = batch.rew
        done = batch.done
        batch_size = obs.shape[0]

        # Update actor
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
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        # Update alpha
        alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Update critic
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

        # Ensure all tensors have matching batch dimensions
        if current_q1.shape[0] != target_q.shape[0]:
            target_q = target_q.expand(current_q1.shape[0], -1)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # Update target networks
        self.soft_update(self.critic_target, self.critic, self.tau)

        # Create loss statistics
        loss = critic_loss.item() + actor_loss.item()
        
        # Return LogInfo object
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
        self.trade_dim = 6
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

        # Initialize log_std to a reasonable value
        self.log_std.weight.data.uniform_(-3, -2)
        self.log_std.bias.data.uniform_(-3, -2)

    def forward(self, obs, state=None, info=None):
        if isinstance(obs, Batch):
            obs = obs.obs

        obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        batch_size = obs.shape[0]

        expected_flat_dim = self.time_steps * self.feature_dim
        if obs.dim() == 2 and obs.shape[1] == expected_flat_dim:
            obs = obs.view(batch_size, self.time_steps, self.feature_dim)

        market_state = obs[:, :, :self.market_dim]
        position_state = obs[:, -1, self.market_dim:self.market_dim + self.position_dim]
        trade_state = obs[:, :, self.market_dim + self.position_dim:]

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
        x = torch.cat([x, position_out], dim=-1)
        x = self.fusion_fc(x)

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp() + 1e-6

        # Return in the format expected by Tianshou's SAC implementation
        loc = mean
        scale = std

        # Ensure proper dimensions
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
    total_size=1000000,
    buffer_num=32
)

class GPUCollector(Collector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collect(self, *args, **kwargs):
        result = super().collect(*args, **kwargs)
        # Move batch data to GPU
        if self.buffer is not None:
            self.buffer.to_torch(device=device)
        return result

collector = GPUCollector(policy, env, buffer, exploration_noise=True)

buffer = VectorReplayBuffer(
    total_size=1000000,  # 1M transitions in total
    buffer_num=32,       # Number of environments
    device=device
)


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
