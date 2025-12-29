# Copyright 2024 Alphaqraft
#
# Market Making Agent for Hierarchical RL
# 
# This agent learns optimal quoting based on market signals.
# Uses shared encoder + attention to focus on relevant signals.
# Reward: Spread Capture + Fee Rebates (learns execution quality)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from shared_encoder import SharedEncoder, AttentionModule


class MMAgent(nn.Module):
    """
    Market Making Agent - Execution quality optimization
    
    Learns optimal bid/ask spreads using shared encoder + attention.
    Attention mechanism allows focusing on microstructure signals.
    Updates every step (500ms).
    
    Architecture: Shared Encoder + Attention + LSTM + Actor/Critic
    
    Input observations (40 dims): All observations (shared with Inventory agent)
    
    Output (2 actions):
        - bid_spread: [0, 1] → multiplies base_spread_bps to get actual bid spread
        - ask_spread: [0, 1] → multiplies base_spread_bps to get actual ask spread
        
    Note: Requote is handled automatically by the environment (smart requote).
    """
    
    def __init__(
        self,
        obs_dim: int = 40,  # Full observation dimension
        shared_encoder: Optional[SharedEncoder] = None,
        hidden_dim: int = 128,
        lstm_hidden: int = 64,
        attention_heads: int = 4,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # Shared encoder (created externally and passed in)
        if shared_encoder is None:
            self.shared_encoder = SharedEncoder(obs_dim, hidden_dim)
            self.owns_encoder = True
        else:
            self.shared_encoder = shared_encoder
            self.owns_encoder = False
        
        # Attention mechanism for MM agent (focuses on microstructure signals)
        self.attention = AttentionModule(hidden_dim, attention_heads)
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        # Actor head: spread actions [0, 1] using Beta distribution
        # Output mean (via sigmoid) and concentration parameter (via softplus)
        self.spread_mean = nn.Linear(lstm_hidden, 2)  # bid_spread, ask_spread mean
        self.spread_concentration = nn.Linear(lstm_hidden, 2)  # concentration parameter for Beta
        
        # Critic head
        self.critic = nn.Linear(lstm_hidden, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights: gain ~1.0 for hidden layers and spread_mean, 0.01 for critic."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Critic: use small gain for stability
                if 'critic' in name:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                # spread_mean and spread_concentration: use standard gain to allow exploration
                elif 'spread_mean' in name or 'spread_concentration' in name:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                elif 'attention' in name or 'lstm' in name:
                    # Attention and LSTM: use standard gain
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    # Hidden layers: use standard gain
                    nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    if 'spread_mean' in name:
                        # Initialize bias to small random values to break symmetry and encourage exploration
                        # Small values (std=0.1) will map to roughly [0.4, 0.6] after sigmoid
                        nn.init.normal_(m.bias, mean=0.0, std=0.1)
                    elif 'spread_concentration' in name:
                        # Initialize concentration bias to encourage moderate exploration
                        # Positive bias -> higher concentration -> less exploration initially
                        # We want moderate exploration, so initialize to small positive value
                        nn.init.constant_(m.bias, 1.0)  # After softplus, this gives ~1.31 concentration
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                # LSTM weights: use standard gain for hidden layers
                for param_name, param in m.named_parameters():
                    if 'weight' in param_name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            obs: Full observations [batch, seq, obs_dim] or [batch, obs_dim]
            hidden: LSTM hidden state tuple (h, c)
            
        Returns:
            spread_mean: Mean of spread actions [batch, 2]
            spread_concentration: Concentration for Beta distribution [batch, 2]
            value: State value [batch, 1]
            hidden: Updated LSTM hidden state
        """
        # Handle 2D input (add sequence dim)
        add_seq_dim = obs.ndim == 2
        if add_seq_dim:
            obs = obs.unsqueeze(1)
        
        batch_size, seq_len, _ = obs.shape
        
        # Ensure observations are completely fresh to avoid in-place modification errors
        # This is critical when the same encoder is used in both rollout and update phases
        obs = obs.clone()
        
        # Shared encoder processes all observations
        encoded = self.shared_encoder(obs)  # [batch, seq, hidden_dim]
        
        # Apply attention (allows focusing on relevant signals)
        # Don't clone here - let attention handle it internally if needed
        attended = self.attention(encoded)  # [batch, seq, hidden_dim]
        
        # LSTM for temporal patterns
        if hidden is None:
            hidden = self._init_hidden(batch_size, obs.device)
        else:
            # Detach and clone hidden states to prevent in-place modifications by LSTM
            # This ensures they're completely independent of any computation graph
            h, c = hidden
            hidden = (h.detach().clone(), c.detach().clone())
        
        lstm_out, hidden = self.lstm(attended, hidden)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        # Actor output: spread actions [0, 1] - multiplies base_spread_bps
        # Mean and concentration for Beta distribution
        spread_mean = torch.sigmoid(self.spread_mean(last_out))  # [0, 1]
        # Concentration parameter: use softplus to ensure positive, minimum 1.0 for numerical stability
        spread_concentration = F.softplus(self.spread_concentration(last_out)) + 1.0
        
        # Critic
        value = self.critic(last_out)
        
        return spread_mean, spread_concentration, value, hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)
    
    def get_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Sample action for execution using Beta distribution (naturally bounded [0, 1]).
        
        Args:
            obs: Full observations [batch, obs_dim]
            hidden: LSTM hidden state
            deterministic: If True, return mean (no exploration)
            temperature: Scale concentration (0.0 = deterministic, 1.0 = full exploration)
            
        Returns:
            action: Spread actions [batch, 2] (bid_spread, ask_spread)
            log_prob: Log probability [batch, 1]
            value: State value [batch, 1]
            hidden: Updated hidden state
        """
        spread_mean, spread_concentration, value, hidden = self.forward(obs, hidden)
        
        if temperature == 0.0:
            # Fully deterministic: use means (regardless of deterministic flag)
            action = spread_mean
            log_prob = torch.zeros(spread_mean.shape[0], 1, device=spread_mean.device)
            return action, log_prob, value, hidden
        
        # Beta distribution: alpha = mean * concentration, beta = (1 - mean) * concentration
        # Temperature scales concentration: lower temperature = higher concentration = less exploration
        effective_concentration = spread_concentration / (temperature + 1e-8)
        alpha = spread_mean * effective_concentration
        beta = (1.0 - spread_mean) * effective_concentration
        
        # Ensure minimum values for numerical stability
        alpha = torch.clamp(alpha, min=0.1)
        beta = torch.clamp(beta, min=0.1)
        
        # Create Beta distribution and sample
        spread_dist = torch.distributions.Beta(alpha, beta)
        action = spread_dist.sample()
        log_prob = spread_dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value, hidden
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions (for PPO update).
        
        Args:
            obs: Full observations [batch, obs_dim]
            actions: Spread actions taken [batch, 2] (must be in [0, 1])
            hidden: LSTM hidden state
            
        Returns:
            log_prob: Log probability [batch, 1]
            entropy: Action entropy [batch, 1]
            value: State value [batch, 1]
        """
        spread_mean, spread_concentration, value, _ = self.forward(obs, hidden)
        
        # Beta distribution: alpha = mean * concentration, beta = (1 - mean) * concentration
        alpha = spread_mean * spread_concentration
        beta = (1.0 - spread_mean) * spread_concentration
        
        # Ensure minimum values for numerical stability
        alpha = torch.clamp(alpha, min=0.1)
        beta = torch.clamp(beta, min=0.1)
        
        # Create Beta distribution and evaluate actions
        spread_dist = torch.distributions.Beta(alpha, beta)
        log_prob = spread_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = spread_dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value
    
# MM agent now uses all 40 observation dimensions via shared encoder + attention
# No need for observation extraction - both agents see everything  # 18
