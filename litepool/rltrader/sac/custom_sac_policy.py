import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import copy
from tianshou.policy import SACPolicy
from tianshou.data import Batch
from dataclasses import dataclass

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

class RudderGRU(nn.Module):
    def __init__(self, input_dim=2420, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, hidden_state=None):
        out, hidden_state = self.gru(obs_seq, hidden_state)
        cum_return_pred = self.fc(out).squeeze(-1)
        return cum_return_pred, hidden_state

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

        # Add RUDDER model
        self.rudder = RudderGRU(input_dim=2420, hidden_dim=128, num_layers=2).to(self.device)
        self.rudder_optim = torch.optim.Adam(self.rudder.parameters(), lr=1e-4)
        self.rudder_loss_fn = nn.MSELoss()

    @property
    def get_alpha(self):
        return self.log_alpha.exp().clamp(min=1e-5)

    def forward(self, batch: Batch, state=None, **kwargs):
        obs = batch.obs if hasattr(batch, 'obs') else batch
        if state is not None and state.dim() == 3 and state.shape[0] == obs.shape[0]:
            state = state.transpose(0, 1).contiguous()

        loc, scale, new_state, _ = self.actor(obs, state=state)
        dist = Independent(Normal(loc, scale), 1)
        act = torch.tanh(dist.rsample())
        log_prob = dist.log_prob(act) - torch.sum(torch.log(1 - act.pow(2) + 1e-6), dim=-1)

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

        with torch.no_grad():
            obs_seq = batch.obs
            cum_return_pred, _ = self.rudder(obs_seq)

            first_step = cum_return_pred[:, 0:1]
            later_steps = cum_return_pred[:, 1:] - cum_return_pred[:, :-1]
            redistributed_rew = torch.cat([first_step, later_steps], dim=1)
            batch.rew = redistributed_rew.unsqueeze(-1)
        return batch

    def learn(self, batch: Batch, **kwargs):
        start_time = time.time()
        batch_size = batch.obs.shape[0]
        n_step = batch.obs.shape[1]

        # --- RUDDER training ---
        with torch.no_grad():
            mc_return = batch.rew.squeeze(-1).sum(dim=1)

        obs_seq = batch.obs.detach()
        pred_seq, _ = self.rudder(obs_seq)
        pred_final = pred_seq[:, -1]

        rudder_loss = self.rudder_loss_fn(pred_final, mc_return)

        self.rudder_optim.zero_grad()
        rudder_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.rudder.parameters(), 1.0)
        self.rudder_optim.step()

        # --- Redistribute reward ---
        with torch.no_grad():
            first_step = pred_seq[:, 0:1]
            later_steps = pred_seq[:, 1:] - pred_seq[:, :-1]
            redistributed_rew = torch.cat([first_step, later_steps], dim=1)
            batch.rew = redistributed_rew.unsqueeze(-1)

        # --- Actor-Critic training ---
        actor_state = batch.get('state', None).reshape(-1, *batch.state.shape[2:]).transpose(0, 1).contiguous()
        critic1_state = batch.get('state_h1', None).reshape(-1, *batch.state_h1.shape[2:]).transpose(0, 1).contiguous()
        critic2_state = batch.get('state_h2', None).reshape(-1, *batch.state_h2.shape[2:]).transpose(0, 1).contiguous()

        flat_obs = batch.obs.view(batch_size * n_step, -1)
        flat_act = batch.act.view(batch_size * n_step, -1)

        loc, scale, new_actor_state, predicted_pnl = self.actor(flat_obs, state=actor_state)
        dist = Independent(Normal(loc, scale), 1)
        actions = torch.tanh(dist.rsample()).clamp(-0.999, 0.999)
        log_prob = (dist.log_prob(actions) -
                    torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)).clamp(-20, 20)

        taus = torch.rand(batch_size * n_step, self.num_quantiles, device=self.device)
        current_q1, new_critic1_state = self.critic1(flat_obs, flat_act, taus, state_h=critic1_state)
        current_q2, new_critic2_state = self.critic2(flat_obs, flat_act, taus, state_h=critic2_state)

        taus_actor = torch.rand(batch_size * n_step, self.num_quantiles, device=self.device)
        new_q1, _ = self.critic1(flat_obs, actions, taus_actor, state_h=critic1_state)
        new_q2, _ = self.critic2(flat_obs, actions, taus_actor, state_h=critic2_state)
        new_q = torch.min(new_q1, new_q2).mean(dim=1)

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        reward = batch.rew.reshape(-1)  
        target = reward.unsqueeze(-1).expand(-1, self.num_quantiles)
        critic_loss = (quantile_huber_loss(current_q1, target, taus, taus, kappa=1.0) +
                       quantile_huber_loss(current_q2, target, taus, taus, kappa=1.0))
        actor_loss = (self.get_alpha.detach() * log_prob - new_q).mean()
        pnl_target = batch.rew.view(batch_size, n_step).sum(dim=1).unsqueeze(1).expand(-1, n_step).reshape(-1)
        pnl_loss = F.mse_loss(predicted_pnl.squeeze(-1), pnl_target)

        total_loss = actor_loss + critic_loss + 0.1 * pnl_loss + 0.5 * rudder_loss

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.alpha_optim.zero_grad()

        total_loss.backward()
        alpha_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)

        self.critic1_optim.step()
        self.critic2_optim.step()
        self.actor_optim.step()
        self.alpha_optim.step()

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

        return SACSummary(
            loss=total_loss.item(),
            loss_actor=actor_loss.item(),
            loss_critic=critic_loss.item(),
            loss_alpha=alpha_loss.item(),
            alpha=self.get_alpha.item(),
            train_time=time.time() - start_time
        )
