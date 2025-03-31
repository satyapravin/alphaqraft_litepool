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

# -----------------------
# Feature Extractor
# -----------------------
class TradingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = feature_dim - (18 + 6)

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

        self.output_dim = 64  # output passed to GRU

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        obs = obs.view(batch_size, time_steps, feature_dim)

        market = obs[:, :, :self.market_dim]
        position = obs[:, -1, self.market_dim:self.market_dim + self.position_dim]
        trade = obs[:, :, self.market_dim + self.position_dim:]

        trade_out = self.trade_fc(trade)
        market_out = self.market_fc(market)

        x = torch.cat([trade_out, market_out], dim=-1)
        return x  # [batch_size, time_steps, 64]

# -----------------------
# Custom Recurrent Policy
# -----------------------
class CustomRecurrentPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TradingFeatureExtractor,
            features_extractor_kwargs={"features_dim": 64},
        )

        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.action_net = nn.Linear(64, self.action_space.shape[0])
        self.value_net = nn.Linear(64, 1)

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        features = self.extract_features(obs)  # shape: (batch, time_steps, 64)

        # Unpack lstm_states
        hidden_state, mask = lstm_states  # hidden_state: (1, batch_size, hidden_dim)

        # Ensure episode_starts is float and reshape for broadcasting
        # Shape: (batch_size,) -> (1, batch_size, 1)
        episode_starts = episode_starts.to(features.device).float().reshape(1, -1, 1)

        # Reset hidden states where episode starts
        hidden_state = hidden_state * (1.0 - episode_starts)

        # Pass through GRU
        features, new_hidden = self.gru(features, hidden_state)

        # Use last time step
        x = features[:, -1, :]
        x = self.mlp(x)

        distribution = self._get_action_dist_from_latent(x)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(x)

        return actions, values, log_prob, (new_hidden, mask)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        features, _ = self.gru(features)
        x = features[:, -1, :]
        x = self.mlp(x)

        distribution = self._get_action_dist_from_latent(x)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = self.value_net(x)

        return value, log_prob, entropy

# -----------------------
# LitePool VecAdapter
# -----------------------
class VecAdapter(VecEnvWrapper):
    def __init__(self, venv: LitePool):
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)
        self.steps = 0
        self.action_env_ids = np.arange(self.venv.num_envs, dtype=np.int32)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.venv.send(self.actions, self.action_env_ids)

    def reset(self):
        self.steps = 0
        return self.venv.reset(self.action_env_ids)[0]

    def step_wait(self):
        obs, rewards, terms, truncs, info_dict = self.venv.recv()
        
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
                obs[i] = self.venv.reset(np.array([i]))[0]

            infos.append(info)

        self.steps += 1
        return obs, rewards, dones, infos

# -----------------------
# Create and Wrap Env
# -----------------------
env = litepool.make("RlTrader-v0", env_type="gymnasium",
    num_envs=32, batch_size=32,
    num_threads=32, is_prod=False,
    is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL",
    tick_size=0.5, min_amount=10,
    maker_fee=-0.00001, taker_fee=0.0005,
    foldername="./train_files/",
    balance=1.0, start=1, max=3601 * 10)

env.spec.id = 'RlTrader-v0'
env = VecAdapter(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecMonitor(env)

# -----------------------
# Create RecurrentPPO Model
# -----------------------
model = RecurrentPPO(
    policy=CustomRecurrentPolicy,
    env=env,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device=device,
    verbose=1,
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
    model.learn(total_timesteps=36005 * 64)
    model.save("recurrent_ppo_rltrader")
