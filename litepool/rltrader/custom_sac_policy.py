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

def compute_n_step_return(batch, gamma, critic1, critic2, actor, alpha, device, n_step=300, num_quantiles=32):
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

    # 2. Compute target Q-values with sequential hidden state propagation
    target_q = torch.zeros(batch_size, n_step, num_quantiles, device=device)  # [64, 300, 32]

    # Initialize hidden states for the full batch
    actor_h = torch.zeros(actor.num_layers, batch_size, actor.gru_hidden_dim, device=device)  # [2, 64, 128]
    critic1_h = torch.zeros(critic1.num_layers, batch_size, critic1.gru_hidden_dim, device=device)  # [2, 64, 128]
    critic2_h = torch.zeros(critic2.num_layers, batch_size, critic2.gru_hidden_dim, device=device)  # [2, 64, 128]

    with torch.no_grad():
        for t in range(n_step):
            # Actor forward pass with hidden state
            loc, scale, actor_h, _ = actor(batch.obs_next[:, t:t+1], state=actor_h)  # [64, 1, 3], [2, 64, 128]
            loc = loc.squeeze(1)  # [64, 3]
            scale = scale.squeeze(1)  # [64, 3]
            dist = Independent(Normal(loc, scale), 1)
            actions = torch.tanh(dist.rsample())  # [64, 3]
            log_prob = dist.log_prob(actions)  # [64]

            # Critic forward passes with hidden states
            taus = torch.rand(batch_size, num_quantiles, device=device)  # [64, 32]
            q1, critic1_h = critic1(batch.obs_next[:, t, :], actions, taus, state_h=critic1_h)  # [64, 32], [2, 64, 128]
            q2, critic2_h = critic2(batch.obs_next[:, t, :], actions, taus, state_h=critic2_h)  # [64, 32], [2, 64, 128]

            q = torch.min(q1, q2)  # [64, 32]
            target_q[:, t] = q - alpha * log_prob.unsqueeze(-1)  # [64, 32]

    # 3. Compute final targets
    not_done = 1 - dones.any(dim=1).float().view(batch_size, 1, 1)  # [64, 1, 1]
    final_targets = cumulative_rewards.unsqueeze(2) + not_done * (gamma ** n_step) * target_q  # [64, 300, 32]

    return final_targets.reshape(batch_size * n_step, num_quantiles)  # [19200, 32]

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
            num_quantiles=32
        )
        batch.q_target = target_q
        return batch


    def learn(self, batch: Batch, state_h1=None, state_h2=None, **kwargs):
        start_time = time.time()
        batch_size = batch.obs.shape[0]
        n_step = batch.obs.shape[1]  # Should be 60 (stack_num)
        num_quantiles = 32
        chunk_size = 8  # Adjust based on GPU memory

        # Initialize losses
        alpha_loss = torch.tensor(0.0, device=self.device)
        actor_loss = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)
        pnl_loss = torch.tensor(0.0, device=self.device)

        # Process in chunks to save memory
        num_chunks = (batch_size * n_step + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, batch_size * n_step)

            # Prepare chunk data
            chunk_obs = batch.obs.view(batch_size * n_step, -1)[start:end]  # [chunk, 2420]
            chunk_act = batch.act.view(batch_size * n_step, -1)[start:end]  # [chunk, action_dim]
            chunk_target = batch.q_target[start:end]  # [chunk, num_quantiles]
            chunk_taus = torch.rand(end - start, num_quantiles, device=self.device)

            # Reshape observations: [chunk, 2420] -> [chunk, 10, 242]
            chunk_obs = chunk_obs.view(-1, 10, 242)

            # Actor forward pass
            with autocast(device_type='cuda'):
                loc, scale, _, predicted_pnl = self.actor(chunk_obs)
                dist = Independent(Normal(loc, scale), 1)
                actions = torch.tanh(dist.rsample())  # [chunk, action_dim]
                log_prob = dist.log_prob(actions) - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)

                # Critic forward passes
                obs_expanded = chunk_obs[:, -1, :].unsqueeze(1).expand(-1, num_quantiles, -1).reshape(-1, 242)
                act_expanded = actions.unsqueeze(1).expand(-1, num_quantiles, -1).reshape(-1, actions.shape[-1])

                # Current Q estimates
                current_q1, _ = self.critic1(obs_expanded, act_expanded, chunk_taus.reshape(-1, 1))
                current_q2, _ = self.critic2(obs_expanded, act_expanded, chunk_taus.reshape(-1, 1))
                current_q1 = current_q1.view(-1, num_quantiles)
                current_q2 = current_q2.view(-1, num_quantiles)

                # New Q estimates for actor loss
                new_q1, _ = self.critic1(chunk_obs[:, -1, :], actions)
                new_q2, _ = self.critic2(chunk_obs[:, -1, :], actions)
                new_q = torch.min(new_q1, new_q2)

            # Loss calculations
            alpha_loss += -(self.get_alpha * (log_prob + self.target_entropy).detach()).mean()

            # Quantile Huber loss for critics
            critic_loss += (
                quantile_huber_loss(current_q1, chunk_target,
                                  torch.rand(end-start, num_quantiles, device=self.device),
                                  chunk_taus, kappa=5.0) +
                quantile_huber_loss(current_q2, chunk_target,
                                  torch.rand(end-start, num_quantiles, device=self.device),
                                  chunk_taus, kappa=5.0)
            )

            actor_loss += (self.get_alpha.detach() * log_prob - new_q).mean()

            # PnL prediction loss (auxiliary task)
            pnl_target = batch.rew.view(batch_size, n_step).sum(dim=1)  # [batch_size]
            pnl_target = pnl_target.unsqueeze(1).expand(-1, n_step).reshape(-1)[start:end]  # [chunk]
            pnl_loss += F.mse_loss(predicted_pnl.squeeze(-1), pnl_target)

        # Average losses across chunks
        alpha_loss /= num_chunks
        actor_loss /= num_chunks
        critic_loss /= num_chunks
        pnl_loss /= num_chunks

        # Total loss
        total_loss = actor_loss + critic_loss + 0.1 * pnl_loss  # pnl_loss is auxiliary

        # Backpropagation
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.alpha_optim.zero_grad()

        self.scaler.scale(total_loss).backward()

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
        
        mean_reward = batch.rew.mean().item()
        # Print training statistics
        print("\n=== Training Statistics ===")
        print(f"1. Policy Loss: {actor_loss.item():.4f}")
        print(f"2. Critic Loss: {critic_loss.item():.4f}")
        print(f"3. Alpha Loss: {alpha_loss.item():.4f}")
        print(f"4. PnL Loss: {pnl_loss.item():.4f}")
        print(f"5. Total Loss: {total_loss.item():.4f}")
        print(f"6. Alpha Value: {self.get_alpha.item():.4f}")
        print(f"7. Mean Reward: {mean_reward:.4f}")
        print("="*30 + "\n")

        return SACSummary(
            loss=total_loss.item(),
            loss_actor=actor_loss.item(),
            loss_critic=critic_loss.item(),
            loss_alpha=alpha_loss.item(),
            alpha=self.get_alpha.item(),
            train_time=time.time() - start_time
        )
