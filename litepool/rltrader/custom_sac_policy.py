import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import copy
from tianshou.policy import SACPolicy
from torch.amp import autocast, GradScaler
from dataclasses import dataclass
from tianshou.data import Batch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def quantile_huber_loss(prediction, target, taus_predicted, taus_target, kappa=1.0):
    B, N = prediction.shape
    _, M = target.shape

    prediction = prediction.unsqueeze(2)
    target = target.unsqueeze(1)
    td_error = target - prediction

    huber = F.smooth_l1_loss(prediction.expand(-1, -1, M), 
                            target.expand(-1, N, -1), 
                            beta=kappa, 
                            reduction="none")

    taus_predicted = taus_predicted.unsqueeze(2)
    taus_target = taus_target.unsqueeze(1)
    
    weight_pred = torch.abs(taus_predicted - (td_error.detach() < 0).float())
    weight_target = torch.abs(taus_target - (td_error.detach() < 0).float())
    weight = (weight_pred + weight_target) / 2
    quantile_loss = weight * huber
    return quantile_loss.mean()


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


def compute_n_step_return(batch, gamma, critic1, critic2, actor, alpha, device, n_step=60, num_quantiles=32):
    rewards = batch.rew  # [batch_size, n_step], e.g., [64, 60]
    dones = batch.done   # [batch_size, n_step]
    obs_next = batch.obs_next  # [batch_size, n_step, obs_dim], e.g., [64, 60, 2420]
    batch_size = rewards.shape[0]

    assert rewards.ndim == 2 and rewards.shape[1] >= n_step, \
        f"Expected 2D rewards with at least {n_step} steps, got {rewards.shape}"

    discounted_rewards = torch.zeros(batch_size, device=device)
    for t in range(n_step):
        mask = 1 - dones[:, :t].any(dim=1).float()
        discounted_rewards += mask * (gamma ** t) * rewards[:, t]

    with torch.no_grad():
        next_state = obs_next[:, -1, :]
        next_loc, next_scale, *_ = actor(next_state)
        next_dist = Independent(Normal(next_loc, next_scale), 1)

        next_actions = torch.tanh(next_dist.rsample((num_quantiles,)))
        next_actions = next_actions.transpose(0, 1)

        next_log_prob = next_dist.log_prob(next_actions)
        next_log_prob = next_log_prob - torch.sum(torch.log(1 - next_actions.pow(2) + 1e-6), dim=-1)

        next_q1, _ = critic1(next_state.unsqueeze(1).expand(-1, num_quantiles, -1),
                            next_actions)
        next_q2, _ = critic2(next_state.unsqueeze(1).expand(-1, num_quantiles, -1),
                            next_actions)
        next_q = torch.min(next_q1, next_q2)
        target_q = next_q - alpha * next_log_prob

        not_done = 1 - dones[:, :n_step].any(dim=1).float()
        target_q = discounted_rewards.unsqueeze(1) + \
                  not_done.unsqueeze(1) * (gamma ** n_step) * target_q

    return target_q  # [batch_size, num_quantiles]


class CustomSACPolicy(SACPolicy):
    def __init__(
            self,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            device,
            action_space=None,
            tau=0.005,
            gamma=0.99,
            init_alpha=2.0,
            **kwargs
    ):
        super().__init__(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            action_space=action_space,
            tau=tau,
            gamma=gamma,
            alpha=init_alpha,
            **kwargs
        )

        self.device = device
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device),
                                      requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.critic1 = critic1
        self.critic2 = critic2
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim
        self.critic1_target = copy.deepcopy(critic1)
        self.critic2_target = copy.deepcopy(critic2)
        self.target_entropy = -np.prod(action_space.shape).item()

        self.scaler = GradScaler()

    @property
    def get_alpha(self):
        return self.log_alpha.exp().clamp(min=1e-5, max=10.0)

    def forward(self, batch: Batch, state=None, **kwargs):
        if hasattr(batch, 'obs'):
            obs = batch.obs
        else:
            obs = batch
        loc, scale, h, _ = self.actor(obs, state=state)
        dist = Independent(Normal(loc, scale), 1)
        act = dist.rsample()
        log_prob = dist.log_prob(act)
        act = torch.tanh(act)
        log_prob = log_prob - torch.sum(torch.log(1 - act.pow(2) + 1e-6), dim=-1)

        return Batch(act=act, state=h, dist=dist, log_prob=log_prob)

    def update_hidden_states(self, obs_next, act, state_h1, state_h2):
        with torch.no_grad():
            obs_next = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
            act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            _, new_state_h1 = self.critic1(obs_next, act, state_h=state_h1)
            _, new_state_h2 = self.critic2(obs_next, act, state_h=state_h2)

        return new_state_h1, new_state_h2

    def process_fn(self, batch: Batch, buffer, indices: np.ndarray) -> Batch:
        """Override process_fn to use custom quantile-based n-step return."""
        batch = self._compute_nstep_return(batch, buffer, indices)
        return batch

    def _compute_nstep_return(self, batch: Batch, buffer, indices: np.ndarray) -> Batch:
        """Custom n-step return computation for quantiles."""
        target_q = compute_n_step_return(
            batch, self.gamma, self.critic1_target, self.critic2_target,
            self.actor, self.get_alpha.detach(), self.device, n_step=60,
            num_quantiles=self.critic1.num_quantiles
        )
        # Since learn expects q_target directly, assign it to batch
        batch.q_target = target_q  # [batch_size, num_quantiles]
        return batch

    def learn(self, batch: Batch, state_h1=None, state_h2=None, **kwargs):
        start_time = time.time()

        for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
            val = getattr(batch, key)
            if isinstance(val, np.ndarray):
                setattr(batch, key, torch.as_tensor(val, dtype=torch.float32, device=self.device))
            elif isinstance(val, torch.Tensor) and val.device != self.device:
                setattr(batch, key, val.to(self.device))

        print(f"learn: batch.rew shape: {batch.rew.shape}, sample rewards: {batch.rew[0, :5]}")

        batch_size = batch.obs.shape[0]

        with autocast(device_type='cuda'):
            # Actor forward pass
            loc, scale, _, predicted_pnl = self.actor(batch.obs)
            dist = Independent(Normal(loc, scale), 1)
            act = dist.rsample()
            log_prob = dist.log_prob(act)
            act_tanh = torch.tanh(act)
            log_prob = log_prob - torch.sum(torch.log(1 - act_tanh.pow(2) + 1e-6), dim=-1)

            # Alpha loss
            alpha_loss = -(self.get_alpha * (log_prob + self.target_entropy).detach()).mean()

            # Sample taus for critics
            taus1 = torch.rand(batch_size, self.critic1.num_quantiles, device=self.device)
            taus2 = torch.rand(batch_size, self.critic2.num_quantiles, device=self.device)
            taus_target = torch.rand(batch_size, self.critic1.num_quantiles, device=self.device)

            # Current quantile predictions
            current_q1, _ = self.critic1(batch.obs, batch.act, taus1)
            current_q2, _ = self.critic2(batch.obs, batch.act, taus2)

            # Use precomputed quantile target from process_fn
            target_q = batch.q_target  # [batch_size, num_quantiles]
            print(f"target_q shape: {target_q.shape}")
            print(f"current_q1 shape: {current_q1.shape}")
            print(f"current_q2 shape: {current_q2.shape}")

            # Critic loss with quantiles
            critic_loss = (
                quantile_huber_loss(current_q1, target_q, taus1, taus_target, kappa=5.0) +
                quantile_huber_loss(current_q2, target_q, taus2, taus_target, kappa=5.0)
            )

            # Actor loss
            q1_new, _ = self.critic1(batch.obs, act_tanh)
            q2_new, _ = self.critic2(batch.obs, act_tanh)
            q_new = torch.min(q1_new, q2_new).mean(dim=1)
            actor_loss = (self.get_alpha.detach() * log_prob - q_new).mean()

            # PnL prediction loss
            pnl_target = batch.rew.sum(dim=1)
            pnl_loss = F.mse_loss(predicted_pnl.squeeze(-1), pnl_target)

            # Total loss
            total_loss = actor_loss + critic_loss + 0.1 * pnl_loss

        # Optimize
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.alpha_optim.zero_grad()

        self.scaler.scale(critic_loss).backward(retain_graph=True)
        self.scaler.scale(actor_loss + 0.1 * pnl_loss).backward(retain_graph=True)
        self.scaler.scale(alpha_loss).backward()

        self.scaler.unscale_(self.critic1_optim)
        self.scaler.unscale_(self.critic2_optim)
        self.scaler.unscale_(self.actor_optim)
        self.scaler.unscale_(self.alpha_optim)

        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)

        self.scaler.step(self.critic1_optim)
        self.scaler.step(self.critic2_optim)
        self.scaler.step(self.actor_optim)
        self.scaler.step(self.alpha_optim)
        self.scaler.update()

        self.sync_weight()

        print("\nDetailed Training Stats:")
        print(f"Actor Loss: {actor_loss.item():.6f}")
        print(f"P/L Prediction Loss: {pnl_loss.item():.6f}")
        print(f"Critic Loss: {critic_loss.item():.6f}")
        print(f"Alpha Loss: {alpha_loss.item():.6f}")
        print(f"Alpha: {self.get_alpha.item():.6f}")
        print(f"Total Loss: {total_loss.item():.6f}")
        print(f"Mean Q1: {current_q1.mean().item():.6f}")
        print(f"Mean Q2: {current_q2.mean().item():.6f}")
        print(f"Mean Target Q: {target_q.mean().item():.6f}")
        print(f"Mean Reward: {batch.rew.mean().item():.6f}")
        print(f"Log Prob Mean: {log_prob.mean().item():.6f}")
        print("-" * 50)

        return SACSummary(
            loss=total_loss.item(),
            loss_actor=actor_loss.item(),
            loss_critic=critic_loss.item(),
            loss_alpha=alpha_loss.item(),
            alpha=self.get_alpha.item(),
            train_time=time.time() - start_time
        )
