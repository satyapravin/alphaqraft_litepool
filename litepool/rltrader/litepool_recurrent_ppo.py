import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecEnvWrapper

import litepool
from litepool.python.protocol import LitePool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_steps = 10
feature_dim = 242

class TradingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 32, num_envs: int = 32):
        super().__init__(observation_space, features_dim)
        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = feature_dim - (18 + 6)
        self.hidden_dim = 32
        self.num_envs = num_envs
        self.num_layers = 1

        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.trade_fc = nn.Sequential(
            nn.Linear(self.trade_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.final_linear = nn.Linear(self.hidden_dim, features_dim)
        self.output_dim = features_dim
        
        # Initialize hidden states
        self.reset_all_hidden_states()

    def reset_hidden_states(self, env_indices):
        """Reset hidden states for specific environments"""
        device = next(self.parameters()).device
        for idx in env_indices:
            self.hidden_states[0][:, idx] = torch.zeros(self.num_layers, self.hidden_dim, device=device)
            self.hidden_states[1][:, idx] = torch.zeros(self.num_layers, self.hidden_dim, device=device)

    def reset_all_hidden_states(self):
        """Reset all hidden states"""
        device = next(self.parameters()).device
        self.hidden_states = (
            torch.zeros(self.num_layers, self.num_envs, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, self.num_envs, self.hidden_dim, device=device)
        )

    def forward(self, obs: torch.Tensor, env_indices=None, dones=None) -> torch.Tensor:
        device = obs.device
        
        # Handle different input shapes
        if len(obs.shape) == 3:  # (batch_size, seq_len, feature_dim)
            batch_size, seq_len, _ = obs.shape
            # Take only the last timestep
            obs = obs[:, -1, :]
        else:  # (batch_size, feature_dim)
            batch_size = obs.shape[0]
        
        # Move hidden states to the correct device
        if self.hidden_states[0].device != device:
            self.hidden_states = (
                self.hidden_states[0].to(device),
                self.hidden_states[1].to(device)
            )

        # Reset hidden states for done episodes
        if dones is not None:
            done_indices = torch.where(dones)[0]
            if len(done_indices) > 0:
                self.reset_hidden_states(done_indices)

        # Split features
        market = obs[:, :self.market_dim]
        position = obs[:, self.market_dim:self.market_dim + self.position_dim]
        trade = obs[:, self.market_dim + self.position_dim:]

        # Process features
        trade_out = self.trade_fc(trade)
        market_out = self.market_fc(market)
        position_out = self.position_fc(position)

        # Combine features
        combined = torch.cat([market_out, trade_out, position_out], dim=-1)
        
        # Add sequence dimension for LSTM
        combined = combined.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Get relevant hidden states
        if env_indices is not None:
            h0 = self.hidden_states[0][:, env_indices]
            c0 = self.hidden_states[1][:, env_indices]
            hidden_states = (h0, c0)
        else:
            hidden_states = self.hidden_states

        # Process through LSTM and immediately detach the new hidden states
        lstm_out, (h_n, c_n) = self.lstm(combined, hidden_states)
        
        # Always detach the hidden states after each forward pass
        if env_indices is not None:
            self.hidden_states[0][:, env_indices] = h_n.detach()
            self.hidden_states[1][:, env_indices] = c_n.detach()
        else:
            self.hidden_states = (h_n.detach(), c_n.detach())

        # Remove sequence dimension and apply final linear layer
        output = self.final_linear(lstm_out.squeeze(1))

        return output


class VecAdapter(VecEnvWrapper):
    def __init__(self, venv: LitePool):
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)
        self.steps = 0
        self.action_env_ids = np.arange(self.venv.num_envs, dtype=np.int32)

        # Observation space should be for a single timestep
        high = np.inf * np.ones(feature_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.steps = 0
        obs = self.venv.reset(self.action_env_ids)[0]
        # Take only the last timestep from each environment
        return obs.reshape(self.num_envs, time_steps, feature_dim)[:, -1, :]

    def step_wait(self):
        obs, rewards, terms, truncs, info_dict = self.venv.recv()

        # Take only the last timestep from each environment
        obs = obs.reshape(self.num_envs, time_steps, feature_dim)[:, -1, :]

        if (np.isnan(obs).any() or np.isinf(obs).any()):
            print("NaN in OBS...................")

        if (np.isnan(rewards).any() or np.isinf(rewards).any()):
            print("NaN in REWARDS...................")
            print(rewards)

        dones = terms + truncs
        infos = []

        for i in range(len(info_dict["env_id"])):
            info = {
                key: info_dict[key][i]
                for key in info_dict.keys()
                if isinstance(info_dict[key], np.ndarray)
            }

            if self.steps % 50 == 0 or dones[i]:
                print("id:{0}, steps:{1}, fees:{2:.8f}, balance:{3:.6f}, unreal:{4:.8f}, real:{5:.8f}, drawdown:{6:.8f}, leverage:{7:.4f}, count:{8}".format(
                    info["env_id"], self.steps, info['fees'],
                    info['balance'] - info['fees'] + info['unrealized_pnl'],
                    info['unrealized_pnl'], info['realized_pnl'],
                    info['drawdown'], info['leverage'], info['trade_count']
                ))

            if dones[i]:
                info["terminal_observation"] = obs[i]
                reset_obs = self.venv.reset(np.array([i]))[0]
                obs[i] = reset_obs.reshape(time_steps, feature_dim)[-1]

            infos.append(info)

        self.steps += 1
        return obs, rewards, dones, infos
# -----------------------
# Create and Wrap Env
# -----------------------
num_envs = 64
env = litepool.make("RlTrader-v0", env_type="gymnasium",
    num_envs=num_envs, batch_size=num_envs,
    num_threads=num_envs, is_prod=False,
    is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL",
    tick_size=0.5, min_amount=10,
    maker_fee=-0.00001, taker_fee=0.0005,
    foldername="./train_files/",
    balance=1.0, start=36000*10, max=7201* 10)

env.spec.id = 'RlTrader-v0'
env = VecAdapter(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecMonitor(env)

# -----------------------
# Create RecurrentPPO Model
# -----------------------
model = RecurrentPPO(
    policy="MlpLstmPolicy",
    policy_kwargs={
        "features_extractor_class": TradingFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 64,
            "num_envs": num_envs  
        },
        "lstm_hidden_size": 64,
        "n_lstm_layers": 1,
    },
    env=env,
    learning_rate=2e-4,
    n_steps=2048,          
    batch_size=256,       
    n_epochs=10,
    gamma=0.99,          
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,        
    max_grad_norm=0.5, 
    device=device,
    verbose=1
)
# -----------------------
# Load Previous Model if Exists
# -----------------------
if os.path.exists("recurrent_ppo_rltrader.zip"):
    model = RecurrentPPO.load("recurrent_ppo_rltrader.zip", env=env, device=device)
    print("Loaded existing RecurrentPPO model.")

# -----------------------
# Training Loop
# -----------------------
for i in range(500):
    model.learn(
        total_timesteps=36005 * 64,
        reset_num_timesteps=False,
        progress_bar=True
    )
    model.save("recurrent_ppo_rltrader")
