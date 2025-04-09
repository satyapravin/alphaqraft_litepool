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

def compute_n_step_return(batch, gamma, critic1, critic2, actor, alpha, device, n_step=300, num_quantiles=32, chunk_size=16):
    batch_size = len(batch)
    rewards = batch.rew.squeeze(-1)  # [64, 300]
    dones = batch.done.squeeze(-1)  # [64, 300]
    
    assert rewards.shape == (batch_size, n_step), "Rewards shape mismatch"
    assert dones.shape == (batch_size, n_step), "Dones shape mismatch"
    assert batch.obs_next.shape == (batch_size, n_step, 2420), "obs_next should be [B, seq_len, obs_dim]"

    # 1. Compute cumulative rewards (vectorized)
    discount_factors = gamma ** torch.arange(n_step, device=device)  # [300]
    mask = 1 - dones.cumsum(dim=1).clamp(0, 1).float()  # [64, 300]
    cumulative_rewards = torch.zeros(batch_size, n_step, device=device)  # [64, 300]
    for t in range(n_step):
        cumulative_rewards[:, t] = (rewards[:, t:] * discount_factors[:n_step-t] * mask[:, t:]).sum(dim=1)

    # 2. Compute target Q-values with chunking
    target_q = torch.zeros(batch_size, n_step, num_quantiles, device=device)  # [64, 300, 32]
    
    # Process in chunks to reduce memory usage
    num_chunks = (batch_size + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, batch_size)
        chunk_batch_size = end_idx - start_idx
        
        # Initialize hidden states for this chunk
        actor_h = torch.zeros(actor.num_layers, chunk_batch_size, actor.gru_hidden_dim, device=device)
        critic1_h = torch.zeros(critic1.num_layers, chunk_batch_size, critic1.gru_hidden_dim, device=device)
        critic2_h = torch.zeros(critic2.num_layers, chunk_batch_size, critic2.gru_hidden_dim, device=device)

        with torch.no_grad():
            for t in range(n_step):
                # Get current chunk data
                obs_chunk = batch.obs_next[start_idx:end_idx, t:t+1]
                
                # Actor forward pass with hidden state
                loc, scale, actor_h, _ = actor(obs_chunk, state=actor_h)
                loc = loc.squeeze(1)
                scale = scale.squeeze(1)
                dist = Independent(Normal(loc, scale), 1)
                actions = torch.tanh(dist.rsample())
                log_prob = dist.log_prob(actions)

                # Critic forward passes with hidden states
                taus = torch.rand(chunk_batch_size, num_quantiles, device=device)
                q1, critic1_h = critic1(obs_chunk.squeeze(1), actions, taus, state_h=critic1_h)
                q2, critic2_h = critic2(obs_chunk.squeeze(1), actions, taus, state_h=critic2_h)

                q = torch.min(q1, q2)
                target_q[start_idx:end_idx, t] = q - alpha * log_prob.unsqueeze(-1)

    # 3. Compute final targets
    not_done = 1 - dones.any(dim=1).float().view(batch_size, 1, 1)
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
            num_quantiles=32,
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
        self.num_quantiles = num_quantiles
        self.critic1 = critic1
        self.critic2 = critic2
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim
        self.critic1_target = copy.deepcopy(critic1)
        self.critic2_target = copy.deepcopy(critic2)
        self.target_entropy = -np.prod(action_space.shape).item()

        self.scaler = GradScaler()
        self.chunk_size = 16  # Reduced chunk size for memory efficiency

    @property
    def get_alpha(self):
        return self.log_alpha.exp().clamp(min=1e-5, max=10.0)

    def forward(self, batch: Batch, state=None, **kwargs):
        if hasattr(batch, 'obs'):
            obs = batch.obs
        else:
            obs = batch
        
        # State shape handling
        if state is not None:
            if state.dim() == 3 and state.shape[0] == obs.shape[0]:  # [B, L, H]
                state = state.transpose(0, 1).contiguous()  # -> [L, B, H] for GRU
        
        loc, scale, new_state, _ = self.actor(obs, state=state)
        dist = Independent(Normal(loc, scale), 1)
        act = dist.rsample()
        log_prob = dist.log_prob(act)
        act = torch.tanh(act)
        log_prob = log_prob - torch.sum(torch.log(1 - act.pow(2) + 1e-6), dim=-1)

        # Return state in collector-friendly shape [B, L, H]
        return Batch(act=act, state=new_state, dist=dist, log_prob=log_prob)

    def update_hidden_states(self, obs_next, act, state_h1, state_h2):
        with torch.no_grad():
            obs_next = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)
            act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            _, new_state_h1 = self.critic1(obs_next, act, state_h=state_h1)
            _, new_state_h2 = self.critic2(obs_next, act, state_h=state_h2)

        return new_state_h1, new_state_h2

    def process_fn(self, batch: Batch, buffer, indices: np.ndarray) -> Batch:
        if not batch:
            return batch

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
            self.actor, self.get_alpha.detach(), self.device, n_step=300,
            num_quantiles=32, chunk_size=self.chunk_size
        )
        batch.q_target = target_q
        return batch

    def learn(self, batch: Batch, state_h1=None, state_h2=None, **kwargs):
        start_time = time.time()
        batch_size = batch.obs.shape[0]
        n_step = batch.obs.shape[1]  # Should be 60 (stack_num)
        num_quantiles = self.num_quantiles  # Use the policy's num_quantiles (32)

        # Reshape batch for processing
        flat_obs = batch.obs.view(batch_size * n_step, -1)  # [B*N, obs_dim]
        flat_act = batch.act.view(batch_size * n_step, -1)  # [B*N, action_dim]
        flat_target = batch.q_target  # [B*N, num_quantiles]

        # Initialize losses
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_alpha_loss = 0.0
        total_pnl_loss = 0.0

        # Process in chunks of observations, keeping full quantile dimension
        num_chunks = (batch_size * n_step + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, batch_size * n_step)

            chunk_obs = flat_obs[start_idx:end_idx]
            chunk_act = flat_act[start_idx:end_idx]
            chunk_target = flat_target[start_idx:end_idx]  # [chunk_size, 32]

            taus = torch.rand(end_idx - start_idx, num_quantiles, device=self.device)
            single_tau = torch.rand(end_idx - start_idx, 1, device=self.device)

            with autocast(device_type='cuda'):
                # Actor forward pass
                loc, scale, _, predicted_pnl = self.actor(chunk_obs)
                dist = Independent(Normal(loc, scale), 1)
                actions = torch.tanh(dist.rsample())
                log_prob = dist.log_prob(actions) - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)

                # Current Q estimates
                current_q1, _ = self.critic1(chunk_obs, chunk_act, taus)
                current_q2, _ = self.critic2(chunk_obs, chunk_act, taus)

                # New Q estimates for actor loss (use single tau)
                new_q1, _ = self.critic1(chunk_obs, actions, single_tau)
                new_q2, _ = self.critic2(chunk_obs, actions, single_tau)
                new_q1 = new_q1.mean(dim=1, keepdim=True)  # Average over quantiles
                new_q2 = new_q2.mean(dim=1, keepdim=True)
                new_q = torch.min(new_q1, new_q2)

            # Loss calculations for this chunk
            alpha_loss = -(self.get_alpha * (log_prob + self.target_entropy).detach()).mean()

            critic_loss = (
                quantile_huber_loss(current_q1, chunk_target,
                                  torch.rand(end_idx - start_idx, num_quantiles, device=self.device),
                                  taus, kappa=5.0) +
                quantile_huber_loss(current_q2, chunk_target,
                                  torch.rand(end_idx - start_idx, num_quantiles, device=self.device),
                                  taus, kappa=5.0)
            )

            actor_loss = (self.get_alpha.detach() * log_prob - new_q.mean(dim=1)).mean()

            # PnL prediction loss (auxiliary task)
            pnl_target = batch.rew.view(batch_size, n_step).sum(dim=1)  # [batch_size]
            pnl_target = pnl_target.unsqueeze(1).expand(-1, n_step).reshape(-1)[start_idx:end_idx]
            pnl_loss = F.mse_loss(predicted_pnl.squeeze(-1), pnl_target)

            # Accumulate losses
            chunk_loss = actor_loss + critic_loss + 0.1 * pnl_loss
            total_actor_loss += actor_loss.item() * (end_idx - start_idx)
            total_critic_loss += critic_loss.item() * (end_idx - start_idx)
            total_alpha_loss += alpha_loss.item() * (end_idx - start_idx)
            total_pnl_loss += pnl_loss.item() * (end_idx - start_idx)

            # Backpropagation for this chunk
            self.critic1_optim.zero_grad()
            self.critic2_optim.zero_grad()
            self.actor_optim.zero_grad()
            self.alpha_optim.zero_grad()

            self.scaler.scale(chunk_loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

            # Update parameters
            self.scaler.step(self.critic1_optim)
            self.scaler.step(self.critic2_optim)
            self.scaler.step(self.actor_optim)
            self.scaler.step(self.alpha_optim)
            self.scaler.update()

        # Update target networks
        self.sync_weight()

        # Calculate average losses
        total_samples = batch_size * n_step
        avg_actor_loss = total_actor_loss / total_samples
        avg_critic_loss = total_critic_loss / total_samples
        avg_alpha_loss = total_alpha_loss / total_samples
        avg_pnl_loss = total_pnl_loss / total_samples
        avg_total_loss = avg_actor_loss + avg_critic_loss + 0.1 * avg_pnl_loss

        mean_reward = batch.rew.mean().item()

        # Print training statistics
        print("\n=== Training Statistics ===")
        print(f"1. Policy Loss: {avg_actor_loss:.4f}")
        print(f"2. Critic Loss: {avg_critic_loss:.4f}")
        print(f"3. Alpha Loss: {avg_alpha_loss:.4f}")
        print(f"4. PnL Loss: {avg_pnl_loss:.4f}")
        print(f"5. Total Loss: {avg_total_loss:.4f}")
        print(f"6. Alpha Value: {self.get_alpha.item():.4f}")
        print(f"7. Mean Reward: {mean_reward:.4f}")
        print("="*30 + "\n", flush=True)

        return SACSummary(
            loss=avg_total_loss,
            loss_actor=avg_actor_loss,
            loss_critic=avg_critic_loss,
            loss_alpha=avg_alpha_loss,
            alpha=self.get_alpha.item(),
            train_time=time.time() - start_time
        )
