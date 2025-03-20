import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import litepool
from tianshou.env import BaseVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer

device = torch.device("cuda")

#-------------------------------------
# Make environment
#------------------------------------
env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=32, batch_size=32,
    num_threads=32, is_prod=False, is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL", tick_size=0.5, min_amount=10,
    maker_fee=-0.00001, taker_fee=0.0005, foldername="./train_files/",
    balance=1.0, start=1, max=3601*10
)
env.spec.id = "RlTrader-v0"
env_action_space = env.action_space


#-----------------------------
# Custom SAC policy
#----------------------------

class CustomSACPolicy(SACPolicy):
    """Custom SAC Policy using Quantile Huber Loss for IQN-based critics."""

    def learn(self, batch: Batch, **kwargs):
        """Override SAC's critic loss with Quantile Huber Loss."""

        # 🔥 Debugging: Print batch type
        print(f"Batch.obs type: {type(batch.obs)}")  # Check if it's NumPy or Tensor
        print(f"Batch.act type: {type(batch.act)}")

        # Convert to PyTorch tensor if necessary
        batch_obs_tensor = torch.as_tensor(batch.obs, device=device)

        # Compute actor output
        action, log_prob = self.actor(batch_obs_tensor)

        # Compute critic Q-values
        current_q1a = self.critic(batch_obs_tensor, batch.act)
        current_q2a = self.critic(batch_obs_tensor, batch.act)

        print(f"Log_prob shape: {log_prob.flatten().shape}")
        print(f"Q1a shape: {current_q1a.shape}, Q2a shape: {current_q2a.shape}")

        # 🔹 Ensure batch size matches between actor and critic
        if current_q1a.shape[0] != batch_obs_tensor.shape[0]:
            raise ValueError(f"Critic batch size mismatch! Expected: {batch_obs_tensor.shape[0]}, Got: {current_q1a.shape[0]}")

        # Generate quantile fractions (taus) for IQN
        taus = torch.linspace(0, 1, self.critic.num_quantiles + 1, device=batch_obs_tensor.device)[:-1]  # ✅ Fixed

        # Compute target Q-value using minimum of two critics
        target_q = torch.min(current_q1a, current_q2a).detach()

        # Compute Quantile Huber Loss
        critic_loss = quantile_huber_loss(current_q1a, target_q, taus) + quantile_huber_loss(current_q2a, target_q, taus)

        return Batch(critic_loss=critic_loss)  # ✅ Corrected

# ---------------------------
# 3. Custom Models for SAC + IQN
# ---------------------------
class RecurrentActor(nn.Module):
    """GRU-based Actor with Feature Extraction"""

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
        self.log_std.weight.data.fill_(-0.5)

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
            state = state[0]  # ✅ Extract tensor from list if needed

        # ✅ Fix: Convert `state` back to `[num_layers, batch_size, hidden_dim]` if needed
        if state.shape == (batch_size, self.num_layers, self.gru_hidden_dim):
            state = state.transpose(0, 1).contiguous()  # Convert `[batch_size, num_layers, hidden_dim]` → `[num_layers, batch_size, hidden_dim]`


        x, new_state = self.gru(x, state)  # ✅ GRU expects `[num_layers, batch_size, hidden_dim]`

        x = x[:, -1, :]
        x = torch.cat([x, position_out], dim=-1)
        x = self.fusion_fc(x)

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 0)

        # ✅ Fix: Return `new_state` as `[batch_size, num_layers, hidden_dim]` for Tianshou
        return (torch.tanh(mean) * self.max_action, log_std.exp()), new_state.detach().transpose(0, 1)  # `[num_layers, batch_size, hidden_dim]` → `[batch_size, num_layers, hidden_dim]`

class IQNCritic(nn.Module):
    """IQN-based Critic with Feature Extraction"""

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

        # 🔹 Feature extraction layers (same as in Actor)
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

        # 🔹 GRU for time-series processing
        self.gru = nn.GRU(64, gru_hidden_dim, num_layers=num_layers, batch_first=True)

        # 🔹 Fusion layer for combining extracted features
        self.fusion_fc = nn.Sequential(
            nn.Linear(gru_hidden_dim + 32 + action_dim, hidden_dim * 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU()
        )

        # 🔹 Q-value output layer
        self.q_values = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, state, action):
        """Processes the observation and action to compute Q-values."""

        # Ensure state is a tensor
        state = torch.as_tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        batch_size = state.shape[0]

        # 🔹 Ensure state is reshaped correctly
        expected_flat_dim = self.time_steps * self.feature_dim
        if state.dim() == 2 and state.shape[1] == expected_flat_dim:
            state = state.view(batch_size, self.time_steps, self.feature_dim)

        # 🔹 Extract different parts of the input
        market_state = state[:, :, :self.market_dim]
        position_state = state[:, -1, self.market_dim:self.market_dim + self.position_dim] 
        trade_state = state[:, :, self.market_dim + self.position_dim:]

        # 🔹 Process each part using feature extractors
        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)

        # 🔹 Combine trade and market features
        x = torch.cat([trade_out, market_out], dim=-1)

        # 🔹 Initialize GRU hidden state
        state_h = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=state.device)

        # 🔹 Process with GRU
        x, _ = self.gru(x, state_h)  # Output shape: (batch, time_steps, gru_hidden_dim)

        # 🔹 Take last timestep
        x = x[:, -1, :]

        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.float32, device=state.device)

        x = torch.cat([x, position_out, action], dim=-1)

        # 🔹 Fusion layer
        x = self.fusion_fc(x)

        # 🔹 Compute Q-values
        return self.q_values(x)

# ---------------------------
# 4. Define SAC Policy with IQN
# ---------------------------
actor = RecurrentActor().to(device)
critic = IQNCritic().to(device)

def quantile_huber_loss(input, target, taus, kappa=1.0):
    td_error = target - input  # Temporal difference error
    huber_loss = F.huber_loss(input, target, delta=kappa, reduction="none") 
    loss = (taus - (td_error.detach() < 0).float()).abs() * huber_loss
    return loss.mean()

policy = CustomSACPolicy(
    actor=actor,
    actor_optim=Adam(actor.parameters(), lr=3e-4),
    critic=critic,
    critic_optim=Adam(critic.parameters(), lr=3e-4),
    tau=0.005, gamma=0.99, alpha=0.2,
    action_space=env_action_space
)

policy = policy.to(device)
# ---------------------------
# 5. Training Setup
# ---------------------------
buffer = VectorReplayBuffer(total_size=100000, buffer_num=32)
collector = Collector(policy, env, buffer, exploration_noise=True)

trainer = OffpolicyTrainer(
    policy=policy,
    train_collector=collector,  
    max_epoch=100,
    step_per_epoch=3605*32,
    step_per_collect=100*32,  
    update_per_step=0.1,
    batch_size=64,
    episode_per_test=0,
)
trainer.run()

torch.save(policy.state_dict(), "sac_iqn_rltrader.pth")
replay_buffer.save_hdf5("replay_iqn_buffer.h5")
