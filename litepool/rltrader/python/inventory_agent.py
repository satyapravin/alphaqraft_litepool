# Copyright 2024 Alphaqraft
#
# Inventory Agent for Hierarchical RL Market Making
# 
# This agent learns WHAT position to hold (strategic, slow decisions)
# Uses shared encoder + attention to focus on relevant signals
# Reward: Unrealized P&L changes (learns market direction)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
from shared_encoder import SharedEncoder, AttentionModule


class InventoryAgent(nn.Module):
    """
    Inventory Agent - Strategic position sizing
    
    Decides target_inventory based on market signals using shared encoder + attention.
    Attention mechanism allows focusing on flow, position, and volatility signals.
    Learns from unrealized P&L (market direction).
    
    Architecture: Shared Encoder + Attention + LSTM + Actor/Critic
    
    Input observations (40 dims): All observations (shared with MM agent)
    
    Output:
        - target_inventory: [-0.1, 0.1] (target leverage)
        - risk_aversion: [0, 1] (risk aversion parameter γ for A-S model)
    """
    
    def __init__(
        self,
        obs_dim: int = 40,  # Full observation dimension
        shared_encoder: Optional[SharedEncoder] = None,
        hidden_dim: int = 128,
        lstm_hidden: int = 32,
        target_range: float = 0.1,  # Max leverage target
        attention_heads: int = 4,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.target_range = target_range
        self.lstm_hidden = lstm_hidden
        
        # Shared encoder (created externally and passed in)
        if shared_encoder is None:
            self.shared_encoder = SharedEncoder(obs_dim, hidden_dim)
            self.owns_encoder = True
        else:
            self.shared_encoder = shared_encoder
            self.owns_encoder = False
        
        # AMM-specific encoder for inventory agent (emphasizes AMM flow signals)
        # AMM signals are at indices [13..16]: net_flow, flow_imbalance, inventory_delta, cumulative_flow
        self.amm_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # Process 4 AMM signals
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Attention mechanism for Inventory agent (focuses on flow, position, volatility)
        # Input is concatenated shared + AMM features, so hidden_dim * 1.5
        self.attention = AttentionModule(hidden_dim + hidden_dim // 2, attention_heads)
        
        # LSTM for temporal patterns (helps reduce flipping by learning sequences)
        # Input size is now hidden_dim + hidden_dim//2 (shared + AMM features)
        self.lstm = nn.LSTM(
            input_size=hidden_dim + hidden_dim // 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        # Actor head: outputs mean and concentration for Beta distribution
        # Both outputs use Beta distribution (naturally bounded [0, 1])
        # target_inventory will be transformed to [-target_range, target_range]
        # risk_aversion stays in [0, 1]
        self.target_inv_mean = nn.Linear(lstm_hidden, 1)  # Mean for target_inventory Beta
        self.target_inv_concentration = nn.Linear(lstm_hidden, 1)  # Concentration for target_inventory Beta
        self.risk_aversion_mean = nn.Linear(lstm_hidden, 1)  # Mean for risk_aversion Beta
        self.risk_aversion_concentration = nn.Linear(lstm_hidden, 1)  # Concentration for risk_aversion Beta
        
        # Critic head: outputs value estimate
        self.critic = nn.Linear(lstm_hidden, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights: gain ~1.0 for hidden layers, 1.0 for output layers."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Output layers: use standard gain for Beta distribution
                if 'target_inv_mean' in name:
                    # Target inventory: initialize to encourage exploration around 0
                    # Use small gain to start near 0.5 (neutral), but allow learning
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # sigmoid(0) = 0.5 (neutral position)
                elif 'risk_aversion_mean' in name:
                    # Risk aversion: initialize to encourage exploration
                    # Use small gain to start near 0.5, but allow learning
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # sigmoid(0) = 0.5 (moderate risk)
                elif 'target_inv_concentration' in name or 'risk_aversion_concentration' in name:
                    # Lower initial concentration = higher variance = more exploration
                    # Start with lower concentration to encourage exploration early
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        # softplus(x) + 1.0, want initial concentration ~1.5 (lower = more exploration)
                        # softplus(0.4) ≈ 0.5, so softplus(0.4) + 1.0 ≈ 1.5
                        nn.init.constant_(m.bias, 0.4)
                elif 'amm_encoder' in name:
                    # AMM encoder: standard initialization
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'critic' in name:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # Hidden layers: use standard gain
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                # Initialize LSTM weights with standard gain
                for name_param, param in m.named_parameters():
                    if 'weight_ih' in name_param:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'weight_hh' in name_param:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name_param:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1 (helps with gradient flow)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
    
    def forward(
        self, 
        obs: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs: Full observations [batch, obs_dim] or [batch, seq_len, obs_dim]
            lstm_hidden: LSTM hidden state (h, c) tuple, or None for initial state
            
        Returns:
            target_inv_mean: Beta mean for target_inventory [batch, 1] or [batch, seq_len, 1] (in [0, 1])
            target_inv_concentration: Beta concentration [batch, 1] or [batch, seq_len, 1]
            risk_aversion_mean: Beta mean for risk_aversion [batch, 1] or [batch, seq_len, 1] (in [0, 1])
            risk_aversion_concentration: Beta concentration [batch, 1] or [batch, seq_len, 1]
            value: State value estimate [batch, 1] or [batch, seq_len, 1]
            lstm_hidden: Updated LSTM hidden state (h, c) tuple
        """
        # Add sequence dimension if needed (for single timestep)
        add_seq_dim = obs.dim() == 2
        if add_seq_dim:
            obs = obs.unsqueeze(1)  # [batch, 1, obs_dim]
        
        # Ensure observations are completely fresh to avoid in-place modification errors
        # This is critical when the same encoder is used in both rollout and update phases
        obs = obs.clone()
        
        # Shared encoder processes all observations
        encoded = self.shared_encoder(obs)  # [batch, seq_len, hidden_dim]
        
        # Extract AMM signals [13..16]: net_flow, flow_imbalance, inventory_delta, cumulative_flow
        amm_signals = obs[..., 13:17]  # [batch, seq_len, 4] or [batch, 4]
        
        # Process AMM signals separately (inventory agent should rely primarily on AMM)
        amm_features = self.amm_encoder(amm_signals)  # [batch, seq_len, hidden_dim//2] or [batch, hidden_dim//2]
        
        # Concatenate shared encoder output with AMM features
        # This gives inventory agent direct access to processed AMM signals
        combined_features = torch.cat([encoded, amm_features], dim=-1)  # [batch, seq_len, hidden_dim + hidden_dim//2]
        
        # Apply attention (allows focusing on relevant signals, including AMM)
        # Don't clone here - let attention handle it internally if needed
        attended = self.attention(combined_features)  # [batch, seq_len, hidden_dim + hidden_dim//2]
        
        # LSTM forward pass
        if lstm_hidden is None:
            # Initialize hidden state if not provided
            batch_size = attended.shape[0]
            lstm_hidden = self._init_hidden(batch_size, attended.device)
        else:
            # Detach and clone hidden states to prevent in-place modifications by LSTM
            # This ensures they're completely independent of any computation graph
            h, c = lstm_hidden
            lstm_hidden = (h.detach().clone(), c.detach().clone())
        
        lstm_out, lstm_hidden = self.lstm(attended, lstm_hidden)  # [batch, seq_len, lstm_hidden]
        
        # Remove sequence dimension if we added it
        if add_seq_dim:
            lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]
        
        # Actor: Beta distribution parameters
        # Mean via sigmoid (bounded [0, 1])
        target_inv_mean = torch.sigmoid(self.target_inv_mean(lstm_out))  # [batch, 1] or [batch, seq_len, 1]
        risk_aversion_mean = torch.sigmoid(self.risk_aversion_mean(lstm_out))  # [batch, 1] or [batch, seq_len, 1]
        
        # Concentration via softplus + 1.0 (ensures >= 1.0 for numerical stability)
        target_inv_concentration = F.softplus(self.target_inv_concentration(lstm_out)) + 1.0
        risk_aversion_concentration = F.softplus(self.risk_aversion_concentration(lstm_out)) + 1.0
        
        # Critic: value estimate
        value = self.critic(lstm_out)
        
        return target_inv_mean, target_inv_concentration, risk_aversion_mean, risk_aversion_concentration, value, lstm_hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample action for execution using Beta distribution.
        
        Args:
            obs: Observations [batch, obs_dim]
            lstm_hidden: LSTM hidden state (h, c) tuple, or None for initial state
            deterministic: If True, return mean (no exploration)
            temperature: Scale exploration noise (0.0 = deterministic, 1.0 = full exploration)
            
        Returns:
            action: [target_inventory, risk_aversion] [batch, 2]
                - target_inventory: [-target_range, target_range] (transformed from Beta [0, 1])
                - risk_aversion: [0, 1] (direct from Beta)
            log_prob: Log probability of action [batch, 1]
            value: State value [batch, 1]
            lstm_hidden: Updated LSTM hidden state (h, c) tuple
        """
        target_inv_mean, target_inv_concentration, risk_aversion_mean, risk_aversion_concentration, value, lstm_hidden = self.forward(obs, lstm_hidden)
        
        if temperature == 0.0 or deterministic:
            # Fully deterministic: use means
            # Transform target_inventory: [0, 1] -> [-target_range, target_range]
            target_inv_beta = target_inv_mean
            risk_aversion_beta = risk_aversion_mean
            # Deterministic: log_prob is 0 (no randomness)
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device)
        else:
            # Beta distribution: alpha = mean * concentration, beta = (1 - mean) * concentration
            # Scale concentration by temperature (higher temperature = more exploration)
            effective_target_concentration = target_inv_concentration / (temperature + 1e-8)
            effective_risk_concentration = risk_aversion_concentration / (temperature + 1e-8)
            
            target_alpha = target_inv_mean * effective_target_concentration
            target_beta = (1.0 - target_inv_mean) * effective_target_concentration
            
            risk_alpha = risk_aversion_mean * effective_risk_concentration
            risk_beta = (1.0 - risk_aversion_mean) * effective_risk_concentration
            
            # Ensure minimum values for numerical stability
            target_alpha = torch.clamp(target_alpha, min=0.1)
            target_beta = torch.clamp(target_beta, min=0.1)
            risk_alpha = torch.clamp(risk_alpha, min=0.1)
            risk_beta = torch.clamp(risk_beta, min=0.1)
            
            # Create Beta distributions and sample
            target_dist = torch.distributions.Beta(target_alpha, target_beta)
            risk_dist = torch.distributions.Beta(risk_alpha, risk_beta)
            
            target_inv_beta = target_dist.sample()
            risk_aversion_beta = risk_dist.sample()
            
            # Compute log probabilities
            target_log_prob = target_dist.log_prob(target_inv_beta)
            risk_log_prob = risk_dist.log_prob(risk_aversion_beta)
            log_prob = (target_log_prob + risk_log_prob).sum(dim=-1, keepdim=True)
        
        # Transform target_inventory from [0, 1] to [-target_range, target_range]
        # y = (x - 0.5) * 2 * target_range
        # Jacobian: |dy/dx| = 2 * target_range
        # log_prob_y = log_prob_x - log(2 * target_range)
        target_inv = (target_inv_beta - 0.5) * 2.0 * self.target_range
        risk_aversion = risk_aversion_beta  # Already in [0, 1]
        
        # Adjust log_prob for transformation (Jacobian correction)
        if not (temperature == 0.0 or deterministic):
            log_prob = log_prob - torch.log(torch.tensor(2.0 * self.target_range, device=obs.device))
        
        action = torch.cat([target_inv, risk_aversion], dim=-1)
        
        return action, log_prob, value, lstm_hidden
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions (for PPO update).
        
        Args:
            obs: Observations [batch, obs_dim] or [batch, seq_len, obs_dim]
            actions: Actions taken [batch, 2] or [batch, seq_len, 2] - [target_inventory, risk_aversion]
                - target_inventory: [-target_range, target_range]
                - risk_aversion: [0, 1]
            lstm_hidden: LSTM hidden state (h, c) tuple, or None for initial state
            
        Returns:
            log_prob: Log probability [batch, 1] or [batch, seq_len, 1]
            entropy: Action entropy [batch, 1] or [batch, seq_len, 1]
            value: State value [batch, 1] or [batch, seq_len, 1]
        """
        target_inv_mean, target_inv_concentration, risk_aversion_mean, risk_aversion_concentration, value, _ = self.forward(obs, lstm_hidden)
        
        # Transform target_inventory back to [0, 1] for Beta distribution
        # y = (x - 0.5) * 2 * target_range
        # x = (y / (2 * target_range)) + 0.5
        target_inv_beta = (actions[:, 0:1] / (2.0 * self.target_range)) + 0.5
        risk_aversion_beta = actions[:, 1:2]  # Already in [0, 1]
        
        # Clamp to valid Beta range [0, 1] (shouldn't be needed, but safety check)
        target_inv_beta = torch.clamp(target_inv_beta, min=1e-6, max=1.0 - 1e-6)
        risk_aversion_beta = torch.clamp(risk_aversion_beta, min=1e-6, max=1.0 - 1e-6)
        
        # Beta distribution: alpha = mean * concentration, beta = (1 - mean) * concentration
        target_alpha = target_inv_mean * target_inv_concentration
        target_beta = (1.0 - target_inv_mean) * target_inv_concentration
        
        risk_alpha = risk_aversion_mean * risk_aversion_concentration
        risk_beta = (1.0 - risk_aversion_mean) * risk_aversion_concentration
        
        # Ensure minimum values for numerical stability
        target_alpha = torch.clamp(target_alpha, min=0.1)
        target_beta = torch.clamp(target_beta, min=0.1)
        risk_alpha = torch.clamp(risk_alpha, min=0.1)
        risk_beta = torch.clamp(risk_beta, min=0.1)
        
        # Create Beta distributions and evaluate
        target_dist = torch.distributions.Beta(target_alpha, target_beta)
        risk_dist = torch.distributions.Beta(risk_alpha, risk_beta)
        
        target_log_prob = target_dist.log_prob(target_inv_beta)
        risk_log_prob = risk_dist.log_prob(risk_aversion_beta)
        
        # Adjust log_prob for transformation (Jacobian correction)
        # y = (x - 0.5) * 2 * target_range, so |dy/dx| = 2 * target_range
        log_prob = (target_log_prob + risk_log_prob).sum(dim=-1, keepdim=True) - torch.log(torch.tensor(2.0 * self.target_range, device=obs.device))
        
        # Entropy (no transformation needed, entropy is invariant under linear transformations)
        target_entropy = target_dist.entropy()
        risk_entropy = risk_dist.entropy()
        entropy = (target_entropy + risk_entropy).sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value
        """
        Extract inventory-relevant observations from full observation.
        
        Full observation (40 dims):
            [0-12]: Market signals (13)
            [13-16]: AMM flow signals (4)
            [17-24]: Trade signals (8)
            [25-35]: Agent state (11)
            [36]: Previous actual bid/ask spread (1)
            [37]: Bid distance from mid (1)
            [38]: Ask distance from mid (1)
            [39]: Mid price change (1)
            
        Inventory obs (12 dims):
            - AMM: net_flow[13], cumulative_flow[16] (only stable signals, removed noisy flow_imbalance[14] and inventory_delta[15])
            - Position: leverage[25], deviation[30], target_ema[32], entry_distance[33]
            - Quote spreads: bid_distance[37], ask_distance[38], total_spread[36], mid_change[39]
            - Trade: volume_imbalance[19], intensity[20], buy_pressure[22], sell_pressure[23]
        """
        # Select relevant indices (removed noisy AMM signals: flow_imbalance[14] and inventory_delta[15])
        indices = [
            13, 16,          # AMM signals (2): net_flow, cumulative_flow (stable signals only)
            25,              # current_leverage (1)
            30,              # deviation_from_target (1)
            32,              # target_inventory_ema (1)
            33,              # entry_price_distance (1)
            37,              # bid_distance_from_mid (1)
            38,              # ask_distance_from_mid (1)
            36,              # total_spread (previous actual spread) (1)
            39,              # mid_price_change (1)
            19, 20, 22, 23,  # trade signals (4)
        ]
        
        if full_obs.ndim == 1:
            return full_obs[indices]
        else:
            return full_obs[:, indices]


# Observation indices for inventory agent (from full 40-dim obs)
# Removed noisy AMM signals: flow_imbalance[14] and inventory_delta[15]
INVENTORY_OBS_INDICES = [
    13, 16,          # AMM: net_flow, cumulative_flow (stable signals only)
    25,              # current_leverage
    30,              # deviation_from_target
    32,              # target_inventory_ema (what agent asked for)
    33,              # entry_price_distance (how far from entry)
    37,              # bid_distance_from_mid (actual bid spread)
    38,              # ask_distance_from_mid (actual ask spread)
    36,              # total_spread (previous actual bid/ask spread)
    39,              # mid_change (mid price change)
    19, 20, 22, 23,  # trade: volume_imbalance, intensity, buy_pressure, sell_pressure
]
INVENTORY_OBS_DIM = len(INVENTORY_OBS_INDICES)  # 12 (removed noisy AMM signals)

