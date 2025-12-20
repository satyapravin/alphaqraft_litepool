"""
LSTM Actor-Critic for temporal market making.

Architecture:
- MLP feature extractor → LSTM for temporal patterns → Actor/Critic heads
- Captures temporal dependencies in market dynamics

4-action space:
- bid_spread: [-1, 1] -> exponential mapping to spread multiplier
- ask_spread: [-1, 1] -> exponential mapping to spread multiplier  
- target_inventory: [-1, 1] desired inventory level (skew computed automatically)
- requote: binary decision to update quotes (>0 = requote)
"""
import numpy as np
import torch
import torch.nn as nn


class SimpleActorCritic(nn.Module):
    def __init__(self, obs_dim=30, action_dim=4, hidden_dim=128, lstm_hidden=64):
        super().__init__()
        assert action_dim == 4, "Expected 4 actions: 3 continuous (bid_spread, ask_spread, target_inventory) + 1 binary (requote)"
        
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # Feature extractor (MLP)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        # Combined features (MLP output + LSTM output)
        combined_dim = hidden_dim + lstm_hidden
        
        # Actor heads
        self.quote_mean = nn.Linear(combined_dim, 3)  # 3 continuous: bid_spread, ask_spread, target_inventory
        self.quote_log_std = nn.Parameter(torch.zeros(3))
        self.requote_logit = nn.Linear(combined_dim, 1)
        
        # Critic head
        self.critic_hidden = nn.Linear(combined_dim, hidden_dim // 2)
        self.critic = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        # Output layers - small init
        nn.init.orthogonal_(self.quote_mean.weight, gain=0.01)
        nn.init.zeros_(self.quote_mean.bias)
        nn.init.orthogonal_(self.requote_logit.weight, gain=0.01)
        nn.init.constant_(self.requote_logit.bias, -3.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def _get_hidden(self, batch_size, device):
        """Get or create hidden states for batch."""
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)
    
    def forward(self, obs, hidden=None):
        """
        Args:
            obs: [batch_size, obs_dim] or [batch_size, seq_len, obs_dim]
            hidden: Optional LSTM hidden state tuple
        Returns:
            quote_dist, requote_dist, value, new_hidden
        """
        # Handle NaN
        if torch.isnan(obs).any():
            obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)
        
        # Ensure 3D for LSTM: [batch, seq_len, features]
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [batch, 1, obs_dim]
        
        batch_size = obs.shape[0]
        device = obs.device
        
        # Feature extraction
        features = self.feature_extractor(obs)  # [batch, seq, hidden]
        
        # LSTM
        if hidden is None:
            hidden = self._get_hidden(batch_size, device)
        
        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep [batch, lstm_hidden]
        
        # Combine MLP features with LSTM output
        mlp_features = features[:, -1, :]  # [batch, hidden]
        combined = torch.cat([mlp_features, lstm_out], dim=-1)  # [batch, hidden + lstm_hidden]
        
        # Handle NaN in combined features
        if torch.isnan(combined).any():
            combined = torch.where(torch.isnan(combined), torch.zeros_like(combined), combined)
        
        # Actor: quote parameters
        quote_mean = self.quote_mean(combined)
        quote_mean = torch.clamp(quote_mean, -10.0, 10.0)
        quote_mean = torch.where(torch.isnan(quote_mean), torch.zeros_like(quote_mean), quote_mean)
        
        clamped_log_std = torch.clamp(self.quote_log_std, -2.0, 0.5)
        quote_std = clamped_log_std.exp().expand_as(quote_mean)
        quote_std = torch.clamp(quote_std, min=1e-6)
        
        quote_dist = torch.distributions.Normal(quote_mean, quote_std)
        
        # Actor: requote decision
        requote_logit = self.requote_logit(combined).squeeze(-1)
        requote_logit = torch.clamp(requote_logit, -10.0, 10.0)
        requote_logit = torch.where(torch.isnan(requote_logit), torch.zeros_like(requote_logit), requote_logit)
        requote_dist = torch.distributions.Bernoulli(logits=requote_logit)
        
        # Critic
        critic_features = torch.relu(self.critic_hidden(combined))
        value = self.critic(critic_features).squeeze(-1)
        # Clamp values to match returns range [-200, 200] to allow learning actual scale
        # With reward_scale=1.0, returns are naturally small ([-3.5, 0.5])
        # Clamp to [-200, 200] to allow learning while preventing explosion
        value = torch.clamp(value, -200.0, 200.0)
        value = torch.where(torch.isnan(value), torch.zeros_like(value), value)
        
        return quote_dist, requote_dist, value, new_hidden
    
    def get_action(self, obs, deterministic=False, hidden=None):
        """
        Get action for environment interaction.
        
        Args:
            obs: observation (numpy or tensor)
            deterministic: if True, return mean/threshold
            hidden: LSTM hidden state
        Returns:
            action, log_prob, value, new_hidden
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            quote_dist, requote_dist, value, new_hidden = self.forward(obs, hidden)
            
            if deterministic:
                quote_action = quote_dist.mean
                requote_action = (requote_dist.probs > 0.5).float()
            else:
                quote_action = quote_dist.sample()
                requote_action = requote_dist.sample()
            
            quote_action = torch.clamp(quote_action, -1.0, 1.0)
            requote_action = requote_action * 2.0 - 1.0  # {0,1} -> {-1,1}
            
            actions = torch.cat([quote_action, requote_action.unsqueeze(-1)], dim=-1)
            
            quote_log_probs = quote_dist.log_prob(quote_action).sum(-1)
            requote_log_probs = requote_dist.log_prob((requote_action + 1.0) / 2.0)
            log_probs = quote_log_probs + requote_log_probs
        
        return actions.numpy(), log_probs.numpy(), value.numpy(), new_hidden
    
    def evaluate_actions(self, obs, actions, hidden=None):
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: [batch_size, obs_dim]
            actions: [batch_size, 4]
            hidden: Optional LSTM hidden state tuple (h, c)
        Returns:
            log_probs, values, entropy
        Note: For training, we use stateless evaluation (hidden=None) to avoid
        backpropagating through the entire episode sequence. The LSTM state
        is only maintained during rollout collection.
        """
        # Use fresh hidden states for training (stateless)
        # This is standard practice - gradients don't flow through rollout sequences
        import time
        eval_start = time.perf_counter()
        quote_dist, requote_dist, values, _ = self.forward(obs, hidden=None)
        eval_time = time.perf_counter() - eval_start
        # Log if evaluation is slow (>50ms for minibatch)
        if eval_time > 0.05:
            print(f"[Model Evaluate] batch_size={obs.shape[0]}, eval_time={eval_time*1000:.2f}ms")
        
        # Split actions
        quote_actions = actions[:, :3]
        requote_actions = actions[:, 3]
        
        # Convert requote from {-1, 1} to {0, 1}
        requote_binary = (requote_actions + 1.0) / 2.0
        
        # Log probabilities
        quote_log_probs = quote_dist.log_prob(quote_actions).sum(-1)
        requote_log_probs = requote_dist.log_prob(requote_binary)
        log_probs = quote_log_probs + requote_log_probs
        
        # Entropy
        quote_entropy = quote_dist.entropy().sum(-1)
        requote_entropy = requote_dist.entropy()
        entropy = quote_entropy + requote_entropy
        
        return log_probs, values, entropy
