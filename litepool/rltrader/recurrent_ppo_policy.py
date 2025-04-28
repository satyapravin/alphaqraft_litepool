import torch
import torch.nn.functional as F

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
        logits, values = self.model.forward(obs, hidden_state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values, hidden_state

    def forward_train(self, obs_seq):
        logits, values = self.model.forward_sequence(obs_seq)
        return logits, values

    def learn(self, minibatch):
        obs = minibatch['obs']
        act = minibatch['act']
        old_logp = minibatch['logp']
        adv = minibatch['adv']
        ret = minibatch['ret']

        logits, values = self.forward_train(obs)

        dist = torch.distributions.Categorical(logits=logits)
        new_logp = dist.log_prob(act)
        entropy = dist.entropy().mean()

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
