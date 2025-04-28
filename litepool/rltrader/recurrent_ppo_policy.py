import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

class RecurrentPPOPolicy:
    def __init__(self, model, action_space, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.device = model.device
        self.action_space = action_space

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, obs, state=None):
        mean, std, value, new_state = self.model(obs, state)
        dist = Independent(Normal(mean, std), 1)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1), new_state

    def compute_gae(self, rewards, values, dones):
        rewards = rewards.cpu().numpy()
        values = values.cpu().numpy()
        dones = dones.cpu().numpy()

        advantages = []
        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * (1 - dones[step]) * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        return advantages, returns

    def learn(self, batch):
        obs = batch.obs.to(self.device)
        act = batch.act.to(self.device)
        old_log_prob = batch.log_prob.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)

        mean, std, value, _ = self.model(obs)
        dist = Independent(Normal(mean, std), 1)
        log_prob = dist.log_prob(act)
        entropy = dist.entropy().mean()

        ratio = (log_prob - old_log_prob).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = F.mse_loss(value.squeeze(-1), returns)

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }
