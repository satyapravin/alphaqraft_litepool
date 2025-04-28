import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentPPOPolicy(nn.Module):
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    ):
        super().__init__()

        self.model = model  # <-- Your custom RecurrentActorCritic
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def forward(self, obs, hidden_state):
        """Forward single step for collector."""
        return self.model(obs, hidden_state)

    def forward_train(self, obs_seq):
        """Forward full sequence for training."""
        return self.model.forward_sequence(obs_seq)

    def learn(self, minibatch):
        """Update policy using PPO clipped loss."""
        obs = minibatch.obs
        act = minibatch.act
        old_logp = minibatch.log_prob
        adv = minibatch.adv
        ret = minibatch.ret

        logits, values = self.forward_train(obs)
        dist = torch.distributions.Categorical(logits=logits)
        new_logp = dist.log_prob(act)
        entropy = dist.entropy()

        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, ret)

        entropy_loss = -entropy.mean()

        loss = actor_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    def compute_gae(self, rewards, values, dones):
        """Compute GAE."""
        rollout_len, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(num_envs, device=rewards.device)

        for t in reversed(range(rollout_len)):
            if t == rollout_len - 1:
                next_value = torch.zeros(num_envs, device=rewards.device)
                next_done = torch.ones(num_envs, device=rewards.device)
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1.0 - next_done) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * last_advantage

        returns = advantages + values
        return advantages, returns
