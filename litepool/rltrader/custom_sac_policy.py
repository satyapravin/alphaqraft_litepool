import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import copy
from tianshou.policy import SACPolicy
from dataclasses import dataclass
from tianshou.data import Batch
from itertools import chain
from tqdm import tqdm

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

def compute_n_step_return(batch, gamma, critic1, critic2, actor, alpha, device, n_step=60, num_quantiles=32, chunk_size=16):
    batch_size = len(batch)
    rewards = batch.rew.squeeze(-1)
    dones = batch.done.squeeze(-1)
    
    actor_state = batch.get('state', None)
    critic1_state = batch.get('state_h1', None)
    critic2_state = batch.get('state_h2', None)
    
    discount_factors = gamma ** torch.arange(n_step, device=device)
    mask = 1 - dones.cumsum(dim=1).clamp(0, 1).float()
    cumulative_rewards = (rewards * discount_factors.unsqueeze(0) * mask).sum(dim=1)
    
    with torch.no_grad():
        obs_next = batch.obs_next
        flat_obs = obs_next.reshape(batch_size * n_step, -1)
        
        actor_state = actor_state.transpose(0, 1).reshape(actor.num_layers, -1, actor.gru_hidden_dim).contiguous()
        loc, scale, _, _ = actor(flat_obs, state=actor_state)
        dist = Independent(Normal(loc, scale), 1)
        raw_actions = dist.rsample()
        actions = torch.tanh(raw_actions)
        raw_log_prob = dist.log_prob(raw_actions)
        correction = torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)
        log_prob = raw_log_prob - correction
        
        actions = actions.reshape(batch_size, n_step, -1)
        log_prob = log_prob.reshape(batch_size, n_step)
        
        flat_actions = actions.reshape(batch_size * n_step, -1)
        taus = torch.rand(batch_size * n_step, num_quantiles, device=device)
        
        critic1_state = critic1_state.transpose(0, 1).reshape(critic1.num_layers, -1, critic1.gru_hidden_dim).contiguous()
        critic2_state = critic2_state.transpose(0, 1).reshape(critic2.num_layers, -1, critic2.gru_hidden_dim).contiguous()
            
        q1, _ = critic1(flat_obs, flat_actions, taus, state_h=critic1_state)
        q2, _ = critic2(flat_obs, flat_actions, taus, state_h=critic2_state)
        q = torch.min(q1, q2).reshape(batch_size, n_step, num_quantiles)
        
        target_q = q - alpha * log_prob.unsqueeze(-1)
    
    not_done = 1 - dones.any(dim=1).float().view(batch_size, 1, 1)
    final_targets = cumulative_rewards.unsqueeze(1).unsqueeze(2) + not_done * (gamma ** n_step) * target_q
    
    print("\n=== compute_n_step_return Debug ===")
    print(f"Actor Loc Mean: {loc.mean().item():.4f}, Min: {loc.min().item():.4f}, Max: {loc.max().item():.4f}")
    print(f"Actor Scale Mean: {scale.mean().item():.4f}, Min: {scale.min().item():.4f}, Max: {scale.max().item():.4f}")
    print(f"Raw Log Prob Mean: {raw_log_prob.mean().item():.4f}, Min: {raw_log_prob.min().item():.4f}, Max: {raw_log_prob.max().item():.4f}")
    print(f"Correction Mean: {correction.mean().item():.4f}, Min: {correction.min().item():.4f}, Max: {correction.max().item():.4f}")
    print(f"Log Prob Mean: {log_prob.mean().item():.4f}, Min: {log_prob.min().item():.4f}, Max: {log_prob.max().item():.4f}")
    print(f"Target Q Mean: {target_q.mean().item():.4f}, Min: {target_q.min().item():.4f}, Max: {target_q.max().item():.4f}")
    print(f"Final Targets Mean: {final_targets.mean().item():.4f}, Min: {final_targets.min().item():.4f}, Max: {final_targets.max().item():.4f}")
    print("="*35 + "\n")
    
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
        self.target_entropy = -np.prod(action_space.shape).item() * 0.5

    @property
    def get_alpha(self):
        return self.log_alpha.exp().clamp(min=1e-5)

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
            self.actor, self.get_alpha.detach(), self.device, n_step=60,
            num_quantiles=32
        )
        batch.q_target = target_q
        return batch

    def learn(self, batch: Batch, **kwargs):
        start_time = time.time()
        batch_size = batch.obs.shape[0]
        n_step = batch.obs.shape[1]

        # Get and reshape hidden states
        actor_state = batch.get('state', None).reshape(-1, *batch.state.shape[2:]).transpose(0, 1).contiguous()
        critic1_state = batch.get('state_h1', None).reshape(-1, *batch.state_h1.shape[2:]).transpose(0, 1).contiguous()
        critic2_state = batch.get('state_h2', None).reshape(-1, *batch.state_h2.shape[2:]).transpose(0, 1).contiguous()

        # Reshape batch
        flat_obs = batch.obs.view(batch_size * n_step, -1)
        flat_act = batch.act.view(batch_size * n_step, -1)
        flat_target = batch.q_target

         
        # Actor forward pass with stabilization
        loc, scale, new_actor_state, predicted_pnl = self.actor(flat_obs, state=actor_state)

        dist = Independent(Normal(loc, scale), 1)
        actions = torch.tanh(dist.rsample()).clamp(-0.999, 0.999)
        log_prob = (dist.log_prob(actions) -
                    torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)).clamp(-20, 20)

        # Critic forward passes with stabilization
        taus = torch.rand(batch_size * n_step, self.num_quantiles, device=self.device)
        current_q1, new_critic1_state = self.critic1(flat_obs, flat_act, taus, state_h=critic1_state)
        current_q2, new_critic2_state = self.critic2(flat_obs, flat_act, taus, state_h=critic2_state)

        print("\n=== Q-Value Statistics ===")
        print(f"Q1 Mean: {current_q1.mean().item():.4f}, Min: {current_q1.min().item():.4f}, Max: {current_q1.max().item():.4f}")
        print(f"Q2 Mean: {current_q2.mean().item():.4f}, Min: {current_q2.min().item():.4f}, Max: {current_q2.max().item():.4f}")
        print(f"Target (q_target) Mean: {batch.q_target.mean().item():.4f}, Min: {batch.q_target.min().item():.4f}, Max: {batch.q_target.max().item():.4f}")
        print("="*30 + "\n")

        # Actor Q-values with multiple quantiles (improved over single tau)
        taus_actor = torch.rand(batch_size * n_step, self.num_quantiles, device=self.device)
        new_q1, _ = self.critic1(flat_obs, actions, taus_actor, state_h=critic1_state)
        new_q2, _ = self.critic2(flat_obs, actions, taus_actor, state_h=critic2_state)
        new_q_dist = torch.min(new_q1, new_q2)  # [batch_size * n_step, num_quantiles]
        new_q = new_q_dist.mean(dim=1)  # Average over quantiles for stability

        # Stabilized losses
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        critic_loss = (quantile_huber_loss(current_q1, flat_target, taus, taus, kappa=1.0) +
                       quantile_huber_loss(current_q2, flat_target, taus, taus, kappa=1.0))
        actor_loss = (self.get_alpha.detach() * log_prob - new_q).mean()

        # PnL loss with stabilization
        pnl_target = batch.rew.view(batch_size, n_step).sum(dim=1)
        pnl_target = pnl_target.unsqueeze(1).expand(-1, n_step).reshape(-1)
        pnl_loss = F.mse_loss(predicted_pnl.squeeze(-1), pnl_target)


        total_loss = actor_loss + critic_loss + 0.1 * pnl_loss

        # Backpropagation with safety checks
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.alpha_optim.zero_grad()

        total_loss.backward()
        alpha_loss.backward()

        self.critic1_optim.step()
        self.critic2_optim.step()
        self.actor_optim.step()
        self.alpha_optim.step()

        # Update target networks
        self.sync_weight()

        # Calculate losses
        total_samples = batch_size * n_step
        avg_actor_loss = actor_loss.item()
        avg_critic_loss = critic_loss.item()
        avg_alpha_loss = alpha_loss.item()
        avg_pnl_loss = pnl_loss.item()
        avg_total_loss = total_loss.item()
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

        # Return training statistics
        return SACSummary(
            loss=avg_total_loss,
            loss_actor=avg_actor_loss,
            loss_critic=avg_critic_loss,
            loss_alpha=avg_alpha_loss,
            alpha=self.get_alpha.item(),
            train_time=time.time() - start_time
        )
