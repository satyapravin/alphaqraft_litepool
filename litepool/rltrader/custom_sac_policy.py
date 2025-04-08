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


def compute_n_step_return(batch, gamma, critic1, critic2, actor, alpha, device, n_step=60, num_quantiles=32, chunk_size=8):
    # Input shapes verification
    assert batch.rew.shape == (batch_size := len(batch), n_step), "Rewards shape mismatch"
    assert batch.done.shape == (batch_size, n_step), "Dones shape mismatch"
    assert batch.obs_next.dim() == 4, "obs_next should be 4D [B,T,stack,obs_dim]"

    # 1. Compute cumulative rewards
    cumulative_rewards = torch.zeros(batch_size, n_step, device=device)
    discount_factors = gamma ** torch.arange(n_step, device=device)

    for t in range(n_step):
        mask = 1 - batch.done[:, :t+1].any(dim=1).float()
        cumulative_rewards[:, t] = mask * (batch.rew[:, t:] * discount_factors[:n_step-t]).sum(dim=1)

    # 2. Process in chunks to save memory
    target_q = torch.zeros(batch_size, n_step, num_quantiles, device=device)

    for b_start in range(0, batch_size, chunk_size):
        b_end = min(b_start + chunk_size, batch_size)
        chunk_obs = batch.obs_next[b_start:b_end, :, -1, :]  # [chunk, T, obs_dim]

        with torch.no_grad():
            # Process each timestep in the chunk
            for t in range(n_step):
                # Actor forward
                loc, scale, *_ = actor(chunk_obs[:, t])
                dist = Independent(Normal(loc, scale), 1)
                actions = torch.tanh(dist.rsample((num_quantiles,)))  # [num_quantiles, chunk, act_dim]
                actions = actions.transpose(0, 1)  # [chunk, num_quantiles, act_dim]

                # Critic evaluation with proper quantile handling
                obs_expanded = chunk_obs[:, t].unsqueeze(1).expand(-1, num_quantiles, -1)  # [chunk, num_quantiles, obs_dim]
                taus = torch.rand(len(obs_expanded), num_quantiles, device=device)  # [chunk, num_quantiles]

                # Process through critics
                q1, _ = critic1(
                    obs_expanded.reshape(-1, obs_expanded.shape[-1]),
                    actions.reshape(-1, actions.shape[-1]),
                    taus.reshape(-1, 1)
                )
                q2, _ = critic2(
                    obs_expanded.reshape(-1, obs_expanded.shape[-1]),
                    actions.reshape(-1, actions.shape[-1]),
                    taus.reshape(-1, 1)
                )

                q1 = q1.view(len(obs_expanded), num_quantiles)
                q2 = q2.view(len(obs_expanded), num_quantiles)
                target_q[b_start:b_end, t] = torch.min(q1, q2) - alpha * dist.log_prob(actions).diagonal(dim1=-2, dim2=-1)

    # 3. Compute final targets with proper broadcasting
    not_done = 1 - batch.done.any(dim=1).float().view(batch_size, 1, 1)
    final_targets = cumulative_rewards.unsqueeze(2) + not_done * (gamma ** n_step) * target_q

    return final_targets.reshape(batch_size * n_step, num_quantiles)

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
        for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
            val = getattr(batch, key)
            if isinstance(val, np.ndarray):
                setattr(batch, key, torch.as_tensor(val, dtype=torch.float32, device=self.device))
            elif isinstance(val, torch.Tensor) and val.device != self.device:
                setattr(batch, key, val.to(self.device))

        batch = self._compute_nstep_return(batch, buffer, indices)
        return batch

    def _compute_nstep_return(self, batch: Batch, buffer, indices: np.ndarray) -> Batch:
        target_q = compute_n_step_return(
            batch, self.gamma, self.critic1_target, self.critic2_target,
            self.actor, self.get_alpha.detach(), self.device, n_step=60,
            num_quantiles=32
        )
        batch.q_target = target_q
        return batch


    def learn(self, batch: Batch, state_h1=None, state_h2=None, **kwargs):
        start_time = time.time()
        batch_size = batch.obs.shape[0]
        n_step = batch.obs.shape[1]
        num_quantiles = 32
        chunk_size = 8  # Adjust based on your GPU memory

        # Process in chunks to save memory
        def process_chunk(chunk_obs, chunk_act):
            with autocast(device_type='cuda'):
                # Actor forward
                loc, scale, _, predicted_pnl = self.actor(chunk_obs)
                dist = Independent(Normal(loc, scale), 1)
                act = dist.rsample()
                log_prob = dist.log_prob(act)
                act_tanh = torch.tanh(act)
                log_prob = log_prob - torch.sum(torch.log(1 - act_tanh.pow(2) + 1e-6), dim=-1)

                # Critic forward
                taus1 = torch.rand(chunk_obs.size(0), num_quantiles, device=self.device)
                taus2 = torch.rand(chunk_obs.size(0), num_quantiles, device=self.device)
                
                obs_flat = chunk_obs.unsqueeze(1).expand(-1, num_quantiles, -1).reshape(-1, chunk_obs.shape[-1])
                act_flat = chunk_act.unsqueeze(1).expand(-1, num_quantiles, -1).reshape(-1, chunk_act.shape[-1])
                
                current_q1, _ = self.critic1(obs_flat, act_flat, taus1.reshape(-1, 1))
                current_q2, _ = self.critic2(obs_flat, act_flat, taus2.reshape(-1, 1))
                
                return {
                    'log_prob': log_prob,
                    'act_tanh': act_tanh,
                    'current_q1': current_q1.view(-1, num_quantiles),
                    'current_q2': current_q2.view(-1, num_quantiles),
                    'predicted_pnl': predicted_pnl
                }

        # Initialize losses
        alpha_loss = 0
        actor_loss = 0
        critic_loss = 0
        pnl_loss = 0
        
        # Process in chunks
        num_chunks = (batch_size * n_step) // chunk_size + 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, batch_size * n_step)
            if start >= end:
                continue
                
            # Prepare chunk data
            chunk_obs = batch.obs.reshape(batch_size * n_step, *batch.obs.shape[2:])[start:end, -1, :]
            chunk_act = batch.act.reshape(batch_size * n_step, -1)[start:end]
            chunk_target = batch.q_target[start:end]
            chunk_taus_target = torch.rand(end-start, num_quantiles, device=self.device)
            
            # Process chunk
            chunk_results = process_chunk(chunk_obs, chunk_act)
            
            # Compute losses for this chunk
            alpha_loss += -(self.get_alpha * (chunk_results['log_prob'] + self.target_entropy).detach()).mean()
            
            chunk_critic_loss = (
                quantile_huber_loss(chunk_results['current_q1'], chunk_target, 
                                  torch.rand(end-start, num_quantiles, device=self.device),
                                  chunk_taus_target, kappa=5.0) +
                quantile_huber_loss(chunk_results['current_q2'], chunk_target,
                                  torch.rand(end-start, num_quantiles, device=self.device),
                                  chunk_taus_target, kappa=5.0)
            )
            critic_loss += chunk_critic_loss
            
            q1_new, _ = self.critic1(chunk_obs, chunk_results['act_tanh'])
            q2_new, _ = self.critic2(chunk_obs, chunk_results['act_tanh'])
            q_new = torch.min(q1_new, q2_new).mean(dim=1)
            actor_loss += (self.get_alpha.detach() * chunk_results['log_prob'] - q_new).mean()
            
            pnl_target = batch.rew.reshape(batch_size, n_step).sum(dim=1)
            pnl_target = pnl_target.unsqueeze(1).expand(-1, n_step).reshape(-1)[start:end]
            pnl_loss += F.mse_loss(chunk_results['predicted_pnl'].squeeze(-1), pnl_target)

        # Average losses across chunks
        alpha_loss /= num_chunks
        actor_loss /= num_chunks
        critic_loss /= num_chunks
        pnl_loss /= num_chunks
        
        total_loss = actor_loss + critic_loss + 0.1 * pnl_loss

        # Optimization (same as before)
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.alpha_optim.zero_grad()

        self.scaler.scale(total_loss).backward()

        # Gradient clipping and updates (same as before)
        # ...
        
        return SACSummary(
            loss=total_loss.item(),
            loss_actor=actor_loss.item(),
            loss_critic=critic_loss.item(),
            loss_alpha=alpha_loss.item(),
            alpha=self.get_alpha.item(),
            train_time=time.time() - start_time
        )
