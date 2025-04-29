import torch
import torch.nn as nn
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

    def forward_train(self, obs_seq, state=None):
        dist, value, new_state = self.model.forward_sequence(obs_seq, state)
        return dist, value, new_state

    def learn(self, minibatch):
        obs = minibatch['obs']
        act = minibatch['act']
        old_logp = minibatch['logp']
        val = minibatch['val']
        adv = minibatch['adv']
        ret = minibatch['ret']
        state = minibatch['state']

        self.model.train()
        dist, values, _ = self.model.forward_sequence(obs, state)
        logp = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1).mean()

        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, ret)

        # === KL Regularization from Bayesian Layers ===
        kl_loss = self.model.kl_loss()  # Blitz handles all BayesianLinear layers

        # Tune this coefficient
        kl_coef = 1e-4

        # === Total Loss with KL ===
        total_loss = (
            policy_loss +
            self.vf_coef * value_loss -
            self.ent_coef * entropy +
            kl_coef * kl_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "actor_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy.item(),
            "kl_loss": kl_loss
        }

    def init_hidden_state(self, batch_size=1):
        """Initialize the RNN hidden state for a batch of environments."""
        return self.model.init_hidden_state(batch_size)


