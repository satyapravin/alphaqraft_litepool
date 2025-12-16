"""
Simple MLP Actor-Critic with separate heads:
- Binary head for requote decision (Bernoulli)
- Continuous heads for quote parameters (Normal)
No recurrence needed since signals already capture temporal dynamics.

Simplified 4-action space:
- spread: [-1, 1] symmetric spread control
- size: [-1, 1] order size control
- skew: [-1, 1] asymmetry for inventory management
- requote: binary decision to update quotes
"""
import torch
import torch.nn as nn


class SimpleActorCritic(nn.Module):
    def __init__(self, obs_dim=16, action_dim=4, hidden_dim=64):
        super().__init__()
        assert action_dim == 4, "Expected 4 actions: 3 quote params (spread, size, skew) + 1 requote"
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate heads for quote parameters (continuous) and requote decision (binary)
        self.quote_mean = nn.Linear(hidden_dim, 3)  # 3 continuous: spread, size, skew
        self.quote_log_std = nn.Parameter(torch.zeros(3))  # Learnable std for quote params
        
        self.requote_logit = nn.Linear(hidden_dim, 1)  # Binary requote decision
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        # Quote output layer - use slightly larger gain for exploration
        nn.init.orthogonal_(self.quote_mean.weight, gain=0.1)
        # Requote output layer - strongly bias AGAINST requoting to let orders persist longer
        nn.init.orthogonal_(self.requote_logit.weight, gain=0.01)
        nn.init.constant_(self.requote_logit.bias, -3.0)  # ~5% requote probability initially (sigmoid(-3) ≈ 0.05)
        # Critic output layer
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim] observation tensor
        Returns:
            quote_dist: Normal distribution for quote parameters [batch_size, 3]
            requote_dist: Bernoulli distribution for requote decision [batch_size]
            value: Value estimate [batch_size]
        """
        # Check for NaN in inputs
        if torch.isnan(obs).any():
            # Replace NaN with 0
            obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)
        
        features = self.shared(obs)
        
        # Check for NaN in features and replace
        if torch.isnan(features).any():
            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        
        # Quote parameters: continuous actions (Normal distribution)
        quote_mean = self.quote_mean(features)
        # Clamp quote_mean to prevent extreme values
        quote_mean = torch.clamp(quote_mean, -10.0, 10.0)
        # Replace any remaining NaN with 0
        quote_mean = torch.where(torch.isnan(quote_mean), torch.zeros_like(quote_mean), quote_mean)
        
        # Clamp log_std to prevent entropy explosion: [-2, 0.5] -> std in [0.14, 1.65]
        clamped_log_std = torch.clamp(self.quote_log_std, -2.0, 0.5)
        quote_std = clamped_log_std.exp().expand_as(quote_mean)
        # Ensure std is positive and not NaN
        quote_std = torch.clamp(quote_std, min=1e-6)
        quote_std = torch.where(torch.isnan(quote_std), torch.ones_like(quote_std) * 0.1, quote_std)
        
        quote_dist = torch.distributions.Normal(quote_mean, quote_std)
        
        # Requote decision: binary action (Bernoulli distribution)
        requote_logit = self.requote_logit(features).squeeze(-1)
        # Clamp to prevent extreme values
        requote_logit = torch.clamp(requote_logit, -10.0, 10.0)
        requote_logit = torch.where(torch.isnan(requote_logit), torch.zeros_like(requote_logit), requote_logit)
        requote_dist = torch.distributions.Bernoulli(logits=requote_logit)
        
        # Value
        value = self.critic(features).squeeze(-1)
        # Clamp value to prevent extreme values
        value = torch.clamp(value, -100.0, 100.0)
        value = torch.where(torch.isnan(value), torch.zeros_like(value), value)
        
        return quote_dist, requote_dist, value
    
    def get_action(self, obs, deterministic=False):
        """
        Get action for environment interaction.
        
        Args:
            obs: observation (numpy or tensor)
            deterministic: if True, return mean/probability > 0.5 for requote
        Returns:
            action: numpy array [4] - [spread, size, skew, requote]
            log_prob: log probability of action
            value: value estimate
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            quote_dist, requote_dist, value = self.forward(obs)
            
            if deterministic:
                quote_action = quote_dist.mean
                requote_action = (requote_dist.probs > 0.5).float()
            else:
                quote_action = quote_dist.sample()
                requote_action = requote_dist.sample()
            
            # Clamp quote parameters to [-1, 1]
            quote_action = torch.clamp(quote_action, -1.0, 1.0)
            
            # Convert requote from {0, 1} to {-1, 1} for action space
            requote_action = requote_action * 2.0 - 1.0  # {0,1} -> {-1,1}
            
            # Combine actions: [spread, size, skew, requote]
            action = torch.cat([quote_action, requote_action.unsqueeze(-1)], dim=-1)
            
            # Compute log probabilities
            quote_log_prob = quote_dist.log_prob(quote_action).sum(-1)
            requote_log_prob = requote_dist.log_prob((requote_action + 1.0) / 2.0)
            log_prob = quote_log_prob + requote_log_prob
        
        return action.squeeze(0).numpy(), log_prob.item(), value.item()
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: [batch_size, obs_dim]
            actions: [batch_size, 4] - [spread, size, skew, requote]
        Returns:
            log_probs: [batch_size]
            values: [batch_size]
            entropy: [batch_size]
        """
        quote_dist, requote_dist, values = self.forward(obs)
        
        # Split actions: first 3 are quote params, last is requote
        quote_actions = actions[:, :3]
        requote_actions = actions[:, 3]
        
        # Convert requote from {-1, 1} back to {0, 1} for Bernoulli
        requote_binary = (requote_actions + 1.0) / 2.0
        
        # Compute log probabilities
        quote_log_probs = quote_dist.log_prob(quote_actions).sum(-1)
        requote_log_probs = requote_dist.log_prob(requote_binary)
        log_probs = quote_log_probs + requote_log_probs
        
        # Compute entropy (sum of both distributions)
        quote_entropy = quote_dist.entropy().sum(-1)
        requote_entropy = requote_dist.entropy()
        entropy = quote_entropy + requote_entropy
        
        return log_probs, values, entropy
