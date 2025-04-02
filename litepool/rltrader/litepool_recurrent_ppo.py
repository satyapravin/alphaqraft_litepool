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
    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = feature_dim - (18 + 6)
        self.hidden_dim = 32
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
        self.hidden_states = None

    def reset_hidden_states(self, env_indices=None):
        """Reset hidden states for specific environments or all if env_indices is None"""
        if self.hidden_states is None:
            return

        device = next(self.parameters()).device
        if env_indices is None:
            batch_size = self.hidden_states[0].shape[1]
            self.hidden_states = (
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        else:
            for idx in env_indices:
                self.hidden_states[0][:, idx] = 0
                self.hidden_states[1][:, idx] = 0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        device = obs.device

        # Initialize or validate hidden states
        if self.hidden_states is None or self.hidden_states[0].shape[1] != batch_size:
            self.hidden_states = (
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            )

        # Reshape observation
        obs = obs.view(batch_size, time_steps, feature_dim)
        obs_reshaped = obs.view(-1, feature_dim)

        # Split and process features
        market = obs_reshaped[:, :self.market_dim]
        position = obs_reshaped[:, self.market_dim:self.market_dim + self.position_dim]
        trade = obs_reshaped[:, self.market_dim + self.position_dim:]

        trade_out = self.trade_fc(trade)
        market_out = self.market_fc(market)
        position_out = self.position_fc(position)

        combined = torch.cat([market_out, trade_out, position_out], dim=-1)
        combined = combined.view(batch_size, time_steps, -1)

        # Process through LSTM with maintained hidden states
        lstm_out, (h_n, c_n) = self.lstm(combined, self.hidden_states)
        
        # Update hidden states
        self.hidden_states = (h_n.detach(), c_n.detach())

        # Take the last timestep's output
        output = self.final_linear(lstm_out[:, -1])

        return output

class VecAdapter(VecEnvWrapper):
    def __init__(self, venv: LitePool):
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)
        self.steps = 0
        self.action_env_ids = np.arange(self.venv.num_envs, dtype=np.int32)
        self.policy = None
        self.model = None

    def set_model(self, model):
        self.model = model

    def get_feature_extractor(self):
        return self.model.policy.features_extractor
        
    def reset(self):
        self.steps = 0
        try:
            obs = self.venv.reset(self.action_env_ids)[0]
            if obs is None or obs.size == 0:
                print("Warning: Received empty observation in reset")
                obs = np.zeros((self.num_envs, time_steps * feature_dim))
            
            # Reset feature extractor hidden states
            feature_extractor = self.get_feature_extractor()
            feature_extractor.reset_hidden_states()
            
            return obs
        except Exception as e:
            print(f"Error in reset: {e}")
            return np.zeros((self.num_envs, time_steps * feature_dim))

    def step_async(self, actions):
        self.last_actions = actions

    def step_wait(self):
        obs, rewards, terms, truncs, info_dict = self.venv.step(self.last_actions)
        dones = np.logical_or(terms, truncs)

        if (np.isnan(obs).any() or np.isinf(obs).any()):
            print("NaN in OBS...................")

        if (np.isnan(rewards).any() or np.isinf(rewards).any()):
            print("NaN in REWARDS...................")

        infos = []
        feature_extractor = self.get_feature_extractor()
        
        for i in range(self.num_envs):
            info = {}
            if "env_id" in info_dict:
                for key in info_dict.keys():
                    if isinstance(info_dict[key], np.ndarray):
                        info[key] = info_dict[key][i] if i < len(info_dict[key]) else 0

            if self.steps % 50 == 0 or dones[i]:
                print("id:{0}, steps:{1}, fees:{2:.8f}, balance:{3:.6f}, unreal:{4:.8f}, real:{5:.8f}, drawdown:{6:.8f}, leverage:{7:.4f}, count:{8}".format(
                    info.get("env_id", i), self.steps, info.get('fees', 0),
                    info.get('balance', 0) - info.get('fees', 0) + info.get('unrealized_pnl', 0),
                    info.get('unrealized_pnl', 0), info.get('realized_pnl', 0),
                    info.get('drawdown', 0), info.get('leverage', 0), info.get('trade_count', 0)
                ))

            if dones[i]:
                info["terminal_observation"] = obs[i]
                reset_obs = self.venv.reset(np.array([i]))[0]
                obs[i] = reset_obs[0] 
                feature_extractor.reset_hidden_states([i])
            infos.append(info)

        self.steps += 1
        return obs, rewards, dones, infos


num_envs = 64
env = litepool.make("RlTrader-v0", env_type="gymnasium",
    num_envs=num_envs, batch_size=num_envs,
    num_threads=num_envs, is_prod=False,
    is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL",
    tick_size=0.5, min_amount=10,
    maker_fee=-0.0001, taker_fee=0.0005,
    foldername="./train_files/",
    balance=1.0, start=36000*10, max=7201*30)

env.spec.id = 'RlTrader-v0'
venv = VecAdapter(env)
env = VecNormalize(venv, norm_obs=True, norm_reward=True)
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
        },
        "lstm_hidden_size": 64,
        "n_lstm_layers": 1,
    },
    env=env,
    learning_rate=2e-4,
    n_steps=256,          
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

venv.set_model(model)
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
