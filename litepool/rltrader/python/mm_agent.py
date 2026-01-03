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
from typing import Tuple, Optional
from shared_encoder import SharedEncoder, AttentionModule


class MMAgent(nn.Module):
    """
    Market Making Agent - Execution quality optimization
    
    Learns optimal bid/ask spreads using shared encoder + attention.
    Attention mechanism allows focusing on microstructure signals.
    Updates every step (500ms).
    
    Architecture: Shared Encoder + Attention + LSTM + Actor/Critic
    
    Input observations (42 dims): All observations (shared with Inventory agent)
    
    Output (3 actions):
        - bid_spread: [0, 1] → multiplier on base spread for bid side
        - ask_spread: [0, 1] → multiplier on base spread for ask side
        - base_spread_bps: [0.5, 3] → learned base spread width in basis points
    """
    
    def __init__(
        self,
        obs_dim: int = 42,  # Full observation dimension
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
        
        # Actor head: 3 actions use Beta distribution [0, 1]
        # - bid_spread, ask_spread: spread multipliers [0, 1]
        # - base_spread_bps: base spread in bps [0.5, 3] (scaled from [0, 1])
        # Output mean (via sigmoid) and concentration parameter (via softplus)
        self.action_mean = nn.Linear(lstm_hidden, 3)  # bid_spread, ask_spread, base_spread_bps
        self.action_concentration = nn.Linear(lstm_hidden, 3)  # concentration for Beta
        
        # Critic head
        self.critic = nn.Linear(lstm_hidden, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights: gain ~1.0 for hidden layers and action heads, 0.01 for critic."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Critic: use small gain for stability
                if 'critic' in name:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                # action_mean and action_concentration: use standard gain to allow exploration
                elif 'action_mean' in name or 'action_concentration' in name:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                elif 'attention' in name or 'lstm' in name:
                    # Attention and LSTM: use standard gain
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    # Hidden layers: use standard gain
                    nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    if 'action_mean' in name:
                        # Initialize: [bid_spread, ask_spread, base_spread_bps]
                        # Spreads: small random for exploration (~0.4-0.6 after sigmoid)
                        # base_spread_bps: bias toward 0.5 (1 bps after scaling)
                        nn.init.normal_(m.bias, mean=0.0, std=0.1)
                    elif 'action_concentration' in name:
                        # Initialize concentration to get UNIFORM distribution (entropy = 0)
                        # For mean = 0.5, we need concentration = 2.0 to get α=1, β=1 (uniform)
                        # softplus(0.54) + 1.0 ≈ 2.0
                        nn.init.constant_(m.bias, 0.54)
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
            action_mean: Mean of 3 actions [batch, 3] (bid_spread, ask_spread, base_spread_bps)
            action_concentration: Concentration for Beta distribution [batch, 3]
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
        attended = self.attention(encoded)  # [batch, seq, hidden_dim]
        
        # LSTM for temporal patterns
        if hidden is None:
            hidden = self._init_hidden(batch_size, obs.device)
        else:
            # Detach and clone hidden states to prevent in-place modifications by LSTM
            h, c = hidden
            hidden = (h.detach().clone(), c.detach().clone())
        
        lstm_out, hidden = self.lstm(attended, hidden)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        # Actor output: 3 actions [0, 1] using Beta distribution
        # [bid_spread, ask_spread, base_spread_bps (scaled)]
        action_mean = torch.sigmoid(self.action_mean(last_out))  # [0, 1]
        # Concentration parameter: clamp to [1.0, 2.0] for non-negative entropy
        action_concentration = torch.clamp(F.softplus(self.action_concentration(last_out)) + 1.0, min=1.0, max=2.0)
        
        # Critic
        value = self.critic(last_out)
        
        return action_mean, action_concentration, value, hidden
    
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
            action: Actions [batch, 3] (bid_spread, ask_spread, base_spread_bps)
            log_prob: Log probability [batch, 1]
            value: State value [batch, 1]
            hidden: Updated hidden state
        """
        action_mean, action_concentration, value, hidden = self.forward(obs, hidden)
        
        if temperature == 0.0 or deterministic:
            # Fully deterministic: use means directly
            # Scale base_spread_bps from [0, 1] to [0.5, 3]
            action = action_mean.clone()
            action[:, 2] = action[:, 2] * 2.5 + 0.5  # base_spread_bps: [0, 1] -> [0.5, 3]
            log_prob = torch.zeros(action_mean.shape[0], 1, device=action_mean.device)
            return action, log_prob, value, hidden
        
        # Beta distribution for 3 actions
        effective_concentration = action_concentration / (temperature + 1e-8)
        alpha = action_mean * effective_concentration
        beta_param = (1.0 - action_mean) * effective_concentration
        alpha = torch.clamp(alpha, min=0.1)
        beta_param = torch.clamp(beta_param, min=0.1)
        
        action_dist = torch.distributions.Beta(alpha, beta_param)
        action_raw = action_dist.sample()  # [0, 1] for all actions
        log_prob = action_dist.log_prob(action_raw).sum(dim=-1, keepdim=True)
        
        # Scale base_spread_bps from [0, 1] to [0.5, 3]
        # Jacobian correction: log(2.5) for the scaling (affine transform y = 2.5x + 0.5)
        action = action_raw.clone()
        action[:, 2] = action_raw[:, 2] * 2.5 + 0.5
        log_prob = log_prob - torch.log(torch.tensor(2.5, device=log_prob.device))
        
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
            actions: Actions taken [batch, 3] (bid_spread, ask_spread, base_spread_bps)
            hidden: LSTM hidden state
            
        Returns:
            log_prob: Log probability [batch, 1]
            entropy: Action entropy [batch, 1]
            value: State value [batch, 1]
        """
        action_mean, action_concentration, value, _ = self.forward(obs, hidden)
        
        # Beta distribution for 3 actions
        alpha = action_mean * action_concentration
        beta_param = (1.0 - action_mean) * action_concentration
        alpha = torch.clamp(alpha, min=0.1)
        beta_param = torch.clamp(beta_param, min=0.1)
        
        # Unscale base_spread_bps from [0.5, 3] back to [0, 1] for Beta distribution
        actions_unscaled = actions.clone()
        actions_unscaled[:, 2] = (actions[:, 2] - 0.5) / 2.5
        
        # Clamp actions to valid Beta range
        actions_clamped = torch.clamp(actions_unscaled, min=1e-6, max=1.0 - 1e-6)
        
        action_dist = torch.distributions.Beta(alpha, beta_param)
        log_prob = action_dist.log_prob(actions_clamped).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        # Jacobian correction for base_spread_bps scaling (y = 2.5x + 0.5)
        log_prob = log_prob - torch.log(torch.tensor(2.5, device=log_prob.device))
        
        return log_prob, entropy, value
    
# NOTE: MM agent uses all 42 observation dimensions via shared encoder + attention
# The attention mechanism learns to focus on microstructure signals dynamically
# No need for observation extraction - both agents see everything

