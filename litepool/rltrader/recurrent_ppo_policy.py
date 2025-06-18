import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecurrentPPOPolicy:
    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, 
                 vf_coef=0.5, ent_coef=0.1, max_grad_norm=0.5, target_kl=1, policy_kl_coef=0.1):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.policy_kl_coef = policy_kl_coef  # For policy KL divergence (old vs. new)

    def init_hidden_state(self, batch_size=1):
        """Initialize the RNN hidden state for a batch of environments."""
        return self.model.init_hidden_state(batch_size)

    def forward(self, obs, hidden_state=None):
        dist, value, entropy, new_hidden_state = self.model.forward(obs, hidden_state)
    
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
    
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= 2*(np.log(2) - raw_action - F.softplus(-2*raw_action)).sum(dim=-1)
    
        return action, log_prob, value, entropy, new_hidden_state


    def forward_train(self, obs_seq, state=None):
        dist, value, entropy, new_state = self.model.forward_sequence(obs_seq, state)
        return dist, value, entropy, new_state

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
        # ------------------------------------------------------------------
        # 1A  – check parameters before any forward pass
        for n, p in self.model.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"✗ NaN in parameter {n}")
                raise RuntimeError("model parameters corrupt")
        # ------------------------------------------------------------------

        obs = minibatch['obs']
        act = minibatch['act']
        old_logp = minibatch['logp']
        val = minibatch['val']
        adv = minibatch['adv']
        ret = minibatch['ret']
        state = minibatch['state']
        ret = torch.clamp(ret, -10.0, 10.0)

        # ------------------------------------------------------------------
        # 1B – check hidden state coming from the collector
        for i, h in enumerate(state):
            if not torch.isfinite(h).all():
                print(f"✗ NaN in hidden_state[{i}] incoming from collector")
                raise RuntimeError("hidden state corrupt")
        # ------------------------------------------------------------------
        self.model.train()
        dist, values, entropy, _ = self.model.forward_sequence(obs, state)
        entropy_loss = entropy.mean()

        # Compute raw actions
        raw_act = torch.atanh(torch.clamp(act, -0.999999, 0.999999))
        logp = dist.log_prob(raw_act).sum(-1)
        action_std = dist.stddev.mean().item()

        # Apply tanh correction
        logp -= (2 * (np.log(2) - raw_act - F.softplus(-2 * raw_act))).sum(dim=-1)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Clipped surrogate objective
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # replace the old value_loss line
        v_pred = values
        v_clip = val + (v_pred - val).clamp(-self.clip_eps, +self.clip_eps)
        value_loss = 0.5 * torch.max(          # element-wise
            (v_pred - ret).pow(2),
            (v_clip - ret).pow(2)
        ).mean()
    
        # Policy KL divergence
        policy_kl_loss = self.compute_policy_kl(dist, raw_act, old_logp)


        current_policy_kl = policy_kl_loss.item()
        if current_policy_kl > 2 * self.target_kl:
            self.policy_kl_coef *= 1.5
        elif current_policy_kl < self.target_kl / 2:
            self.policy_kl_coef /= 1.5
        self.policy_kl_coef = max(1e-3, min(self.policy_kl_coef, 2.0))

        # Total loss
        total_loss = (
            policy_loss +
            self.vf_coef * value_loss -
            self.ent_coef * entropy_loss +
            self.policy_kl_coef * policy_kl_loss
        )

        # Early stopping if policy KL is too high
        if current_policy_kl > 8 * self.target_kl:
            print(f"Early stopping: Policy KL ({current_policy_kl:.6f}) exceeds threshold ({4 * self.target_kl:.6f})")
            return {
                "loss": total_loss.item(),
                 "actor_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "policy_kl_loss": policy_kl_loss.item(),
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
            "entropy_loss": entropy_loss.item(),
            "policy_kl_loss": policy_kl_loss.item(),
            "policy_kl_coef": self.policy_kl_coef,
            "action_std": action_std,
            "early_stop": False
        }
