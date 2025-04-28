import torch
import torch.nn.functional as F
import numpy as np

class RecurrentPPOPolicy:
    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def forward(self, obs, hidden_state=None):
        dist, value, new_hidden_state = self.model.forward(obs, hidden_state)

        raw_action = dist.rsample()  # reparameterized sample
        action = torch.tanh(raw_action)  # squash

        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1)

        return action, log_prob, value, new_hidden_state

    def forward_train(self, obs_seq):
        dist, value = self.model.forward_sequence(obs_seq)
        return dist, value

    def learn(self, minibatch):
        obs = minibatch['obs']
        act = minibatch['act']
        old_logp = minibatch['logp']
        adv = minibatch['adv']
        ret = minibatch['ret']

        dist, values = self.forward_train(obs)

        # Invert tanh to get pre-squashed action
        raw_action = torch.atanh(act.clamp(-0.999, 0.999))
        new_logp = dist.log_prob(raw_action).sum(-1)
        new_logp -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1)

        entropy = dist.entropy().sum(-1).mean()

        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.flatten(), ret.flatten())
        loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy.item()
        }

    def init_hidden_state(self, batch_size=1):
        """Initialize the RNN hidden state for a batch of environments."""
        return self.model.init_hidden_state(batch_size)
