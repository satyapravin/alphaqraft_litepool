import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecurrentPPOPolicy:
    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, 
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, target_kl=1, policy_kl_coef=0.1):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.bayesian_kl_coef = 10  # For Bayesian layer KL divergence
        self.policy_kl_coef = policy_kl_coef  # For policy KL divergence (old vs. new)

    def init_hidden_state(self, batch_size=1):
        """Initialize the RNN hidden state for a batch of environments."""
        return self.model.init_hidden_state(batch_size)

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

    def compute_policy_kl(self, dist_new, raw_act, old_logp):
        """
        Compute KL divergence between new and old policy distributions.
        Args:
            dist_new: New policy distribution (Normal) from current model.
            raw_act: Raw actions (before tanh) used to compute old_logp.
            old_logp: Log probabilities from the old policy.
        Returns:
            kl_div: Mean KL divergence (scalar).
        """
        # Compute log probabilities under new distribution
        logp_new = dist_new.log_prob(raw_act).sum(-1)
        logp_new -= (2 * (np.log(2) - raw_act - F.softplus(-2 * raw_act))).sum(dim=-1)
        
        # Approximate KL divergence: E[log(p_old) - log(p_new)]
        kl_div = (old_logp - logp_new).mean()
        return kl_div

    def learn(self, minibatch):
        obs = minibatch['obs']
        act = minibatch['act']  # Squashed actions (tanh applied)
        old_logp = minibatch['logp']
        val = minibatch['val']
        adv = minibatch['adv']
        ret = minibatch['ret']
        state = minibatch['state']

        self.model.train()
        dist, values, _ = self.model.forward_sequence(obs, state)

        # Compute raw actions from squashed actions
        raw_act = torch.atanh(torch.clamp(act, -0.999999, 0.999999))  # Avoid numerical issues at boundaries
        logp = dist.log_prob(raw_act).sum(-1)
        action_std = dist.stddev.mean().item()

        # Apply tanh correction
        logp -= (2 * (np.log(2) - raw_act - F.softplus(-2 * raw_act))).sum(dim=-1)

        entropy = dist.entropy().sum(-1).mean()

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Clipped surrogate objective (trust region)
        ratio = torch.exp(logp - old_logp)
        ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, ret)

        # Bayesian KL divergence (from Bayesian layers)
        bayesian_kl_loss = self.model.nn_kl_divergence() / (obs.shape[0] * obs.shape[1])

        # Policy KL divergence (old vs. new policy)
        policy_kl_loss = self.compute_policy_kl(dist, raw_act, old_logp)

        # Variance penalty
        var_penalty = 0.01 * torch.mean(dist.stddev**2)

        # Adaptive KL coefficients
        current_bayesian_kl = bayesian_kl_loss.item()
        if current_bayesian_kl > 2 * self.target_kl:
            self.bayesian_kl_coef *= 1.5
        elif current_bayesian_kl < self.target_kl / 2:
            self.bayesian_kl_coef /= 1.5
        self.bayesian_kl_coef = min(self.bayesian_kl_coef, 1)

        current_policy_kl = policy_kl_loss.item()
        if current_policy_kl > 2 * self.target_kl:
            self.policy_kl_coef *= 1.5
        elif current_policy_kl < self.target_kl / 2:
            self.policy_kl_coef /= 1.5
        self.policy_kl_coef = max(min(self.policy_kl_coef, 1.0), 2.0)

        # Total loss
        total_loss = (
            policy_loss +
            self.vf_coef * value_loss -
            self.ent_coef * entropy +
            self.bayesian_kl_coef * bayesian_kl_loss +
            self.policy_kl_coef * policy_kl_loss +
            var_penalty
        )

        # Early stopping if policy KL is too high
        if current_policy_kl > 4 * self.target_kl:
            print(f"Early stopping: Policy KL ({current_policy_kl:.6f}) exceeds threshold ({4 * self.target_kl:.6f})")
            return {
                "loss": total_loss.item(),
                "actor_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy_loss": entropy.item(),
                "bayesian_kl_loss": bayesian_kl_loss.item(),
                "policy_kl_loss": policy_kl_loss.item(),
                "bayesian_kl_coef": self.bayesian_kl_coef,
                "policy_kl_coef": self.policy_kl_coef,
                "action_std": action_std,
                "early_stop": True
            }

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "actor_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy.item(),
            "bayesian_kl_loss": bayesian_kl_loss.item(),
            "policy_kl_loss": policy_kl_loss.item(),
            "bayesian_kl_coef": self.bayesian_kl_coef,
            "policy_kl_coef": self.policy_kl_coef,
            "action_std": action_std,
            "early_stop": False
        }
