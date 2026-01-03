# Copyright 2024 Alphaqraft
#
# Inventory Agent for Hierarchical RL Market Making
# 
# This agent learns WHAT position to hold (strategic, slow decisions)
# Uses FOCUSED observations (AMM flow + position state) to avoid signal dilution
# Reward: Unrealized P&L delta (position direction) + fee rebates

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from shared_encoder import SharedEncoder  # For interface compatibility only


class InventoryAgent(nn.Module):
    """
    Inventory Agent - Strategic position sizing
    
    Decides target_inventory based on FOCUSED observations:
    - AMM flow signals (indices 13-16): directional market flow
    - Position state (indices 25-33): current leverage, P&L, deviation, entry distance
    - Price trend (index 39): dual EMA crossover trend signal
    - P&L momentum (index 40): rolling net P&L trend (winning/losing streak)
    - P&L volatility (index 41): stability of P&L stream
    
    DOES NOT use shared encoder - uses only 16 focused features to avoid dilution.
    
    Architecture: Focused Encoder + LSTM + Actor/Critic
    
    Input observations (42 dims): Full obs, but only uses indices [13:17] + [25:34] + [39:42]
    
    Output:
        - target_inventory: [-target_range, +target_range] (target leverage, default ±1.0)
        - risk_aversion: [0, 1] (risk aversion parameter γ for A-S model)
    """
    
    # Observation indices for focused features
    AMM_INDICES = slice(13, 17)  # net_flow, flow_imbalance, inventory_delta, cumulative_flow
    POSITION_INDICES = slice(25, 34)  # Position state (9 signals):
    # [25] leverage, [26] position_value, [27] inventory_pnl, [28] realized_pnl, [29] spread_capture,
    # [30] deviation_from_target, [31] unrealized_pnl_pct, [32] target_inventory_ema, [33] entry_price_distance
    # Additional trend/momentum signals extracted separately:
    PRICE_TREND_INDEX = 39       # Price trend signal (fast EMA vs slow EMA crossover)
    ROLLING_PNL_INDEX = 40       # Rolling P&L momentum (winning/losing streak)
    PNL_VOLATILITY_INDEX = 41    # P&L volatility (sqrt of EMA of squared deltas)
    FOCUSED_DIM = 16  # 4 AMM + 9 position + 3 extra (price_trend, rolling_pnl, pnl_vol)
    
    def __init__(
        self,
        obs_dim: int = 42,  # Full observation dimension (for interface compatibility)
        shared_encoder: Optional[SharedEncoder] = None,  # Not used, kept for interface compatibility
        hidden_dim: int = 128,
        lstm_hidden: int = 32,
        target_range: float = 1.0,  # Max leverage target
        attention_heads: int = 4,  # Not used, kept for interface compatibility
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.target_range = target_range
        self.lstm_hidden = lstm_hidden
        
        # Store but don't use shared encoder (for interface compatibility with HierarchicalPolicy)
        self.shared_encoder = shared_encoder
        self.owns_encoder = False
        
        # Focused encoder: processes only AMM + position signals (9 dims -> hidden_dim)
        # This prevents dilution of critical directional signals
        self.focused_encoder = nn.Sequential(
            nn.Linear(self.FOCUSED_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
        )
        
        # LSTM for temporal patterns (helps reduce flipping by learning sequences)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
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
                    # Target inventory: initialize to encourage SYMMETRIC exploration
                    # Use higher gain (1.0) so different observations map to different directions
                    # This prevents the agent from collapsing to always-long or always-short
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # sigmoid(0) = 0.5 (neutral center)
                elif 'risk_aversion_mean' in name:
                    # Risk aversion: initialize to encourage exploration
                    # Use small gain to start near 0.5, but allow learning
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # sigmoid(0) = 0.5 (moderate risk)
                elif 'target_inv_concentration' in name or 'risk_aversion_concentration' in name:
                    # Initialize concentration to get UNIFORM distribution (entropy = 0)
                    # For mean = 0.5, we need concentration = 2.0 to get α=1, β=1 (uniform)
                    # softplus(0.54) + 1.0 ≈ 1.0 + 1.0 = 2.0
                    nn.init.orthogonal_(m.weight, gain=0.3)  # Low gain to keep concentration stable
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.54)  # softplus(0.54) + 1.0 ≈ 2.0 → uniform
                elif 'focused_encoder' in name:
                    # Focused encoder: standard initialization
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
        Forward pass using FOCUSED observations (AMM + position only).
        
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
        
        # Extract FOCUSED observations: AMM signals + position state + trend/momentum/volatility signals
        # This prevents dilution of critical directional signals
        amm_signals = obs[..., self.AMM_INDICES]  # [batch, seq_len, 4]
        position_state = obs[..., self.POSITION_INDICES]  # [batch, seq_len, 9]
        price_trend = obs[..., self.PRICE_TREND_INDEX:self.PRICE_TREND_INDEX+1]  # [batch, seq_len, 1]
        rolling_pnl = obs[..., self.ROLLING_PNL_INDEX:self.ROLLING_PNL_INDEX+1]  # [batch, seq_len, 1]
        pnl_volatility = obs[..., self.PNL_VOLATILITY_INDEX:self.PNL_VOLATILITY_INDEX+1]  # [batch, seq_len, 1]
        focused_obs = torch.cat([amm_signals, position_state, price_trend, rolling_pnl, pnl_volatility], dim=-1)  # [batch, seq_len, 16]
        
        # Process focused observations (no shared encoder dilution)
        encoded = self.focused_encoder(focused_obs)  # [batch, seq_len, hidden_dim]
        
        # LSTM forward pass
        if lstm_hidden is None:
            # Initialize hidden state if not provided
            batch_size = encoded.shape[0]
            lstm_hidden = self._init_hidden(batch_size, encoded.device)
        else:
            # Detach and clone hidden states to prevent in-place modifications by LSTM
            # This ensures they're completely independent of any computation graph
            h, c = lstm_hidden
            lstm_hidden = (h.detach().clone(), c.detach().clone())
        
        lstm_out, lstm_hidden = self.lstm(encoded, lstm_hidden)  # [batch, seq_len, lstm_hidden]
        
        # Remove sequence dimension if we added it
        if add_seq_dim:
            lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]
        
        # Actor: Beta distribution parameters
        # Mean via sigmoid (bounded [0, 1])
        target_inv_mean = torch.sigmoid(self.target_inv_mean(lstm_out))  # [batch, 1] or [batch, seq_len, 1]
        risk_aversion_mean = torch.sigmoid(self.risk_aversion_mean(lstm_out))  # [batch, 1] or [batch, seq_len, 1]
        
        # Concentration via softplus + 1.0, clamped to [1.0, 2.0]
        # Beta(mean*c, (1-mean)*c) has non-negative entropy when c <= 2
        # concentration=1 gives U-shaped distribution, concentration=2 gives uniform-like
        target_inv_concentration = torch.clamp(F.softplus(self.target_inv_concentration(lstm_out)) + 1.0, min=1.0, max=2.0)
        risk_aversion_concentration = torch.clamp(F.softplus(self.risk_aversion_concentration(lstm_out)) + 1.0, min=1.0, max=2.0)
        
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
        # Transform risk_aversion from [0, 1] to [0, 0.1] for A-S model
        # Lower gamma = more aggressive inventory management
        risk_aversion = risk_aversion_beta * 0.1
        
        # Adjust log_prob for transformation (Jacobian correction)
        # target_inv: y = (x - 0.5) * 2 * target_range, |dy/dx| = 2 * target_range
        # risk_aversion: y = x * 0.1, |dy/dx| = 0.1
        # Total Jacobian = 2 * target_range * 0.1 = 0.2 * target_range
        if not (temperature == 0.0 or deterministic):
            log_prob = log_prob - torch.log(torch.tensor(0.2 * self.target_range, device=obs.device))
        
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
        # Transform risk_aversion back from [0, 0.1] to [0, 1] for Beta distribution
        risk_aversion_beta = actions[:, 1:2] / 0.1
        
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
        # target_inv: y = (x - 0.5) * 2 * target_range, |dy/dx| = 2 * target_range
        # risk_aversion: y = x * 0.1, |dy/dx| = 0.1
        # Total Jacobian = 2 * target_range * 0.1 = 0.2 * target_range
        log_prob = (target_log_prob + risk_log_prob).sum(dim=-1, keepdim=True) - torch.log(torch.tensor(0.2 * self.target_range, device=obs.device))
        
        # Entropy: use RAW Beta entropy without Jacobian correction
        # The Jacobian correction is a constant offset that doesn't reflect exploration
        # For PPO entropy bonus, we care about the SHAPE of the distribution, not the absolute value
        # Beta entropy CAN be negative for concentrated distributions - that's fine
        target_entropy = target_dist.entropy()
        risk_entropy = risk_dist.entropy()
        # Sum entropies without Jacobian - this reflects actual exploration behavior
        entropy = (target_entropy + risk_entropy).sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value
    

# NOTE: Inventory agent now uses FOCUSED observations (16 dims) instead of shared encoder
# - AMM signals (indices 13-16): net_flow, flow_imbalance, inventory_delta, cumulative_flow
# - Position state (indices 25-33): leverage, position_value, inventory_pnl, realized_pnl, 
#   spread_capture, deviation_from_target, unrealized_pnl_pct, target_inventory_ema, entry_price_distance
# - Price trend (index 39): dual EMA crossover (fast vs slow) - positive = uptrend, negative = downtrend
# - Rolling P&L momentum (index 40): winning/losing streak indicator (EMA of net P&L deltas, weighted avg)
# - P&L volatility (index 41): stability of returns (sqrt of EMA of squared deltas)
# This gives the agent direct visibility into P&L, trends, and return stability

