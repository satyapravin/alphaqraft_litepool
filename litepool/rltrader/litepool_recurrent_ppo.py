from typing import Callable
import numpy as np
import time
import torch
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import copy

# Torch settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.optim import Adam
from tianshou.data import Collector, Batch, to_numpy, to_torch
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger

device = torch.device("cuda")

#-------------------------------------
# Make environment
#------------------------------------
num_of_envs = 64
stack_num = 10

import litepool
env = litepool.make(
    "RlTrader-v0", 
    env_type="gymnasium", 
    num_envs=num_of_envs, 
    batch_size=num_of_envs,
    num_threads=num_of_envs, 
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
    start=3601*20, 
    max=3600*10
)

env.spec.id = 'RlTrader-v0'
env_action_space = env.action_space

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

    def normalize_obs(self, obs):
        if not self.norm_obs:
            return obs

        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        batch_size = obs.shape[0]

        obs = obs.view(batch_size, self.time_steps, self.feature_dim)
        flat_obs = obs.view(-1, self.feature_dim)

        self.obs_rms.update(flat_obs)
        normed = (flat_obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)
        normed = torch.clamp(normed, -self.clip_obs, self.clip_obs)

        normed = normed.view(batch_size, self.time_steps, self.feature_dim)
        return normed.view(batch_size, -1)

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

class TradingRecurrentNet(nn.Module):
    def __init__(self, state_dim=2420, action_dim=4, hidden_dim=64, gru_hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers
        self.feature_dim = 242
        self.time_steps = 10
        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = 218
        self.action_dim = action_dim
        self.max_action = 1

        # Feature extractors
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

        # GRU for temporal features
        self.gru = nn.GRU(96, gru_hidden_dim, num_layers=num_layers, batch_first=True)

        # Shared features layer
        self.shared_features = nn.Sequential(
            nn.Linear(gru_hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Auxiliary heads
        self.move_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary prediction
        )

        self.pnl_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # PnL prediction
        )

    def forward(self, obs, state=None):
        obs = torch.as_tensor(obs, device=device)
        obs = obs.view(-1, self.time_steps, self.feature_dim)
        batch_size = obs.shape[0]

        # Process each timestep
        features = []
        for t in range(self.time_steps):
            # Split features
            market_state = obs[:, t, :self.market_dim]
            position_state = obs[:, t, self.market_dim:self.market_dim + self.position_dim]
            trade_state = obs[:, t, self.market_dim + self.position_dim:self.market_dim + self.position_dim + self.trade_dim]

            # Process features
            position_out = self.position_fc(position_state)
            trade_out = self.trade_fc(trade_state)
            market_out = self.market_fc(market_state)

            # Combine features
            combined = torch.cat([position_out, trade_out, market_out], dim=-1)
            features.append(combined)

        # Stack features across time
        features = torch.stack(features, dim=1)

        # Initialize GRU state if needed
        if state is None:
            state = torch.zeros(self.num_layers, batch_size, 
                              self.gru_hidden_dim, device=device)

        # GRU processing
        x, new_state = self.gru(features, state)
        x = x[:, -1, :]  # Take last timestep

        # Shared features
        shared_feat = self.shared_features(torch.cat([new_state[-1], x], dim=-1))

        # Get outputs
        actor_out = self.actor(shared_feat)
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        log_std = log_std.clamp(-20, 2)

        value = self.critic(shared_feat)
        
        move_pred = self.move_predictor(shared_feat)
        pnl_pred = self.pnl_predictor(shared_feat)

        return (mean, log_std), value, new_state, move_pred, pnl_pred

from tianshou.data import ReplayBuffer, Batch

class CustomCollector(Collector):
    def __init__(self, policy, env, buffer_size=20000, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.hidden_state = None
        self.buffer = ReplayBuffer(size=buffer_size)
        self.reset()
        
    def reset(self, reset_buffer=True):
        """Reset the collector and optionally the buffer."""
        if reset_buffer:
            self.buffer.reset()
        
        obs, _ = self.env.reset()
        self.obs = obs
        self.hidden_state = torch.zeros(
            2,  # num_layers 
            self.env_num,  # batch_size
            128,  # hidden_dim
            device=device
        )
        return obs
        
    def collect(self, n_step=0, n_episode=0, random=False, render=None):
        """Collect data with hidden state management."""
        # If both are specified, prioritize n_step
        if n_step > 0:
            n_episode = 0
        elif n_episode <= 0:
            n_episode = 1
            
        collected_episodes = 0
        collected_steps = 0
        
        while True:
            # Get action from policy
            with torch.no_grad():
                act, self.hidden_state = self.policy.forward(self.obs, self.hidden_state)
                
            # Execute action in environment
            obs_next, rew, terminated, truncated, info = self.env.step(act)
            done = np.logical_or(terminated, truncated)
            
            # Reset hidden state for done episodes
            if done.any():
                self.hidden_state[:, done, :] = 0.0
                collected_episodes += done.sum()
            
            collected_steps += 1
            
            # Store transition
            batch = Batch(
                obs=self.obs,
                act=act,
                rew=rew,
                done=done,
                obs_next=obs_next,
                info=info
            )
            
            self.buffer.add(batch)
            
            self.obs = obs_next
            
            if n_step > 0 and collected_steps >= n_step:
                break
            if n_episode > 0 and collected_episodes >= n_episode:
                break
                
        # Return statistics
        return {
            'n/ep': collected_episodes,
            'n/st': collected_steps,
            'buffer': self.buffer
        }

    @property
    def env_num(self):
        """Return the number of environments."""
        return self.env.num_envs if hasattr(self.env, 'num_envs') else 1

    def reset_buffer(self, keep_statistics=False):
        """Reset the buffer."""
        self.buffer.reset()

class TradingRecPPO(PPOPolicy):
    def __init__(
        self,
        actor: TradingRecurrentNet,
        critic: TradingRecurrentNet,
        optim: torch.optim.Optimizer,
        dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.distributions.Distribution],
        eps_clip: float = 0.2,
        aux_coef: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn=dist_fn, eps_clip=eps_clip, **kwargs)
        self.aux_coef = aux_coef

    def forward(self, batch, state=None, **kwargs):
        """Policy forward with state handling."""
        logits, _, hidden, _, _ = self.actor(batch, state)
        if isinstance(logits, tuple):
            mean, log_std = logits
            dist = self.dist_fn(mean, log_std)
        else:
            dist = self.dist_fn(logits)
        act = dist.sample()
        return act, hidden

    def learn(self, batch, batch_size=None, repeat=10, **kwargs):
        losses = []
        for step in range(repeat):
            with torch.no_grad():
                # Get old action probabilities and values
                obs_batch = to_torch(batch.obs, device=self.actor.device)
                (mean, log_std), value, state, move_pred, pnl_pred = self.actor(obs_batch)
                dist = self.dist_fn(mean, log_std)
                old_log_prob = dist.log_prob(batch.act)

            # Get new probabilities and values
            (mean, log_std), value, state, move_pred, pnl_pred = self.actor(obs_batch)
            dist = self.dist_fn(mean, log_std)
            log_prob = dist.log_prob(batch.act)

            # PPO losses
            ratio = (log_prob - old_log_prob).exp()
            surr1 = ratio * batch.adv
            surr2 = ratio.clamp(1 - self._eps_clip, 1 + self._eps_clip) * batch.adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(value.squeeze(-1), batch.returns)
            
            # Auxiliary losses
            mid_dev = torch.as_tensor(batch.info["mid_diff"], 
                                    dtype=torch.float32, 
                                    device=self.actor.device)
            binary_mid_dev = (mid_dev >= 0).float()
            
            move_loss = F.binary_cross_entropy_with_logits(
                move_pred.squeeze(-1), 
                binary_mid_dev[:, -1]
            )
            
            pnl_loss = F.mse_loss(
                pnl_pred.squeeze(-1), 
                batch.rew
            )

            loss = policy_loss + self._w_vf * value_loss + self.aux_coef * (move_loss + pnl_loss)

            self.optim.zero_grad()
            loss.backward()
            if self._grad_norm:
                nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self._grad_norm)
            self.optim.step()

            losses.append(loss.item())

        return {
            'loss': np.mean(losses),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'move_loss': move_loss.item(),
            'pnl_loss': pnl_loss.item()
        }

class MetricLogger:
    def __init__(self, print_interval=1000):
        self.print_interval = print_interval
        self.last_print_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def log(self, step_count, info, rew, policy):
        if step_count % self.print_interval == 0:
            print(f"\nStep: {step_count}")
            print("Env | Net_PnL   | R_PnL     | UR_PnL    | Fees      | Trades   | Drawdown | Leverage | Reward")
            print("-" * 80)
            
            for ii in info['env_id']:
                net_pnl = info['realized_pnl'][ii] + info['unrealized_pnl'][ii] - info['fees'][ii]
                print(f"{ii:3d} | {net_pnl:+7.6f} | {info['realized_pnl'][ii]:+6.6f} | "
                      f"{info['unrealized_pnl'][ii]:+6.6f} | {info['fees'][ii]:+6.6f} | "
                      f"{info['trade_count'][ii]} | {info['drawdown'][ii]:+8.6f} | "
                      f"{info['leverage'][ii]:+8.6f} | {rew[ii]:+8.6f}")
            
            print("-" * 80)


if __name__ == "__main__":
    # Setup directories
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Wrap environment
    env = VecNormalize(
        env,
        num_envs=num_of_envs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10000000.,
        clip_reward=1000.,
        gamma=0.99
    )

def build_dist_fn(mean, log_std):
    return Independent(Normal(mean, log_std.exp()), 1)

    # Initialize network and policy
    net = TradingRecurrentNet().to(device)
    policy = TradingRecPPO(
        actor=net,
        critic=net,
        optim=torch.optim.Adam(net.parameters(), lr=1e-4),
        dist_fn=build_dist_fn,  # Fixed this
        eps_clip=0.2,  # Added this
        discount_factor=0.99,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        max_batchsize=64,
        action_space=action_space,
        deterministic_eval=True,
        advantage_normalization=True,
        aux_coef=0.1
    )

    # Load existing checkpoint if available
    final_checkpoint_path = results_dir / "final_model.pth"
    start_epoch = 0

    if final_checkpoint_path.exists():
        print(f"Loading model from {final_checkpoint_path}")
        checkpoint = torch.load(final_checkpoint_path)
        
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.optim.load_state_dict(checkpoint['optim_state_dict'])
        
        env.obs_rms.mean = checkpoint['obs_rms_mean']
        env.obs_rms.var = checkpoint['obs_rms_var']
        env.obs_rms.count = checkpoint['obs_rms_count']
        env.ret_rms.mean = checkpoint['ret_rms_mean']
        env.ret_rms.var = checkpoint['ret_rms_var']
        env.ret_rms.count = checkpoint['ret_rms_count']
        
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    else:
        print(f"No existing checkpoint found at {final_checkpoint_path}")

    # Setup collector and logger
    collector = CustomCollector(
        policy, env,
        exploration_noise=True,
        buffer_size=2000
    )
    
    if final_checkpoint_path.exists():
        collector.hidden_state = checkpoint.get('hidden_state')
        
    logger = MetricLogger(print_interval=1000)
    collector.logger = logger

    # Training
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=collector,
        max_epoch=100,
        step_per_epoch=1000,
        repeat_per_collect=10,
        episode_per_collect=16,
        episode_per_test=0,
        batch_size=64,
        step_per_collect=2000,
        resume_from_log=True,
    )

    # Run training
    trainer.run()

    # Save final model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optim_state_dict': policy.optim.state_dict(),
        'obs_rms_mean': env.obs_rms.mean,
        'obs_rms_var': env.obs_rms.var,
        'obs_rms_count': env.obs_rms.count,
        'ret_rms_mean': env.ret_rms.mean,
        'ret_rms_var': env.ret_rms.var,
        'ret_rms_count': env.ret_rms.count,
        'epoch': trainer.epoch,
        'env_step': trainer.env_step,
        'gradient_step': trainer.gradient_step,
        'hidden_state': collector.hidden_state,
    }, final_checkpoint_path)

    print("\nTraining completed!")
    print(f"Final model saved to: {final_checkpoint_path}")
