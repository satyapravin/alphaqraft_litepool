# Copyright 2024 Alphaqraft
#
# Inventory Agent for Hierarchical RL Market Making
# 
# This agent learns WHAT position to hold (strategic, slow decisions)
# Based on AMM flow signals, volatility, and market regime
# Reward: Unrealized P&L changes (learns market direction)

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional


class InventoryAgent(nn.Module):
    """
    Inventory Agent - Strategic position sizing
    
    Decides target_inventory based on market flow signals.
    Updates every N steps (e.g., 100 steps = 10 seconds).
    Learns from unrealized P&L (market direction).
    
    Architecture: Small MLP [64, 32] - runs infrequently
    
    Input observations (12 dims):
        - AMM signals: net_flow, cumulative_flow (2 stable signals only, removed noisy flow_imbalance and inventory_delta)
        - Position state: leverage, deviation_from_target, target_ema, entry_distance (4)
        - Quote spreads: actual_bid_spread, actual_ask_spread, total_spread, mid_change (4)
        - Trade signals: volume_imbalance, trade_intensity, buy_pressure, sell_pressure (4)
    
    Output:
        - target_inventory: [-0.1, 0.1] (target leverage)
        - risk_aversion: [0, 1] (risk aversion parameter γ for A-S model)
    """
    
    def __init__(
        self,
        obs_dim: int = 12,
        hidden_dims: Tuple[int, ...] = (64, 32),
        lstm_hidden: int = 32,
        target_range: float = 0.1,  # Max leverage target
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.target_range = target_range
        self.lstm_hidden = lstm_hidden
        
        # Build MLP encoder layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # LSTM for temporal patterns (helps reduce flipping by learning sequences)
        self.lstm = nn.LSTM(
            input_size=prev_dim,  # Last hidden_dim from encoder
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        # Actor head: outputs mean for target_inventory and risk_aversion
        self.actor_mean = nn.Linear(lstm_hidden, 2)  # [target_inventory, risk_aversion]
        self.actor_log_std = nn.Parameter(torch.zeros(2))  # Separate log_std for each output
        
        # Critic head: outputs value estimate
        self.critic = nn.Linear(lstm_hidden, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights: gain ~1.0 for hidden layers, 0.01 for output layers."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Output layers (actor_mean, actor_log_std, critic): use small gain
                if 'actor_mean' in name or 'critic' in name:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    # Hidden layers: use standard gain
                    nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    if 'actor_mean' in name:
                        # Initialize bias to output 0 for target_inventory (first output)
                        # For tanh(0.3 * (Wx + b)) to be 0, we want b[0] ≈ 0
                        # Since weights are small (gain=0.01), b[0] = 0 should give near-zero output
                        # b[1] for risk_aversion: sigmoid(0) = 0.5, so keep at 0
                        nn.init.zeros_(m.bias)
                    else:
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
            obs: Inventory observations [batch, obs_dim] or [batch, seq_len, obs_dim]
            lstm_hidden: LSTM hidden state (h, c) tuple, or None for initial state
            
        Returns:
            action_mean: [target_inventory, risk_aversion] [batch, 2] or [batch, seq_len, 2]
            value: State value estimate [batch, 1] or [batch, seq_len, 1]
            lstm_hidden: Updated LSTM hidden state (h, c) tuple
        """
        # Encode observations
        features = self.encoder(obs)  # [batch, hidden_dim] or [batch, seq_len, hidden_dim]
        
        # Add sequence dimension if needed (for single timestep)
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, 1, hidden_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # LSTM forward pass
        if lstm_hidden is None:
            # Initialize hidden state if not provided
            batch_size = features.shape[0]
            lstm_hidden = self._init_hidden(batch_size, features.device)
        
        lstm_out, lstm_hidden = self.lstm(features, lstm_hidden)  # [batch, seq_len, lstm_hidden]
        
        # Remove sequence dimension if we added it
        if squeeze_output:
            lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]
        
        # Actor: outputs [target_inventory, risk_aversion]
        # Use smaller scaling (0.3 instead of 0.5) to prevent saturation and encourage learning from 0
        # With small weights (gain=0.01) and bias=0, initial output should be near 0
        raw_mean = self.actor_mean(lstm_out) * 0.3
        raw_mean = torch.tanh(raw_mean)
        
        # target_inventory: [-target_range, target_range]
        # Initial output will be near 0 (neutral position), then learn from there
        target_inv_mean = raw_mean[:, 0:1] * self.target_range
        
        # risk_aversion: [0, 1] using sigmoid (already in correct range)
        risk_aversion_mean = torch.sigmoid(raw_mean[:, 1:2])
        
        action_mean = torch.cat([target_inv_mean, risk_aversion_mean], dim=-1)
        
        # Critic: value estimate
        value = self.critic(lstm_out)
        
        return action_mean, value, lstm_hidden
    
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
        Sample action for execution.
        
        Args:
            obs: Observations [batch, obs_dim]
            lstm_hidden: LSTM hidden state (h, c) tuple, or None for initial state
            deterministic: If True, return mean (no exploration)
            temperature: Scale exploration noise (0.0 = deterministic, 1.0 = full exploration)
            
        Returns:
            action: [target_inventory, risk_aversion] [batch, 2]
            log_prob: Log probability of action [batch, 1]
            value: State value [batch, 1]
            lstm_hidden: Updated LSTM hidden state (h, c) tuple
        """
        action_mean, value, lstm_hidden = self.forward(obs, lstm_hidden)
        
        if temperature == 0.0:
            # Fully deterministic: use means (regardless of deterministic flag)
            return action_mean, torch.zeros_like(action_mean), value, lstm_hidden
        
        # Sample from Gaussian with temperature scaling
        std = torch.exp(self.actor_log_std).expand_as(action_mean) * temperature
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # Sum log probs for both outputs
        
        # Clamp actions to valid ranges
        target_inv = torch.clamp(action[:, 0:1], -self.target_range, self.target_range)
        risk_aversion = torch.clamp(action[:, 1:2], 0.0, 1.0)
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
            lstm_hidden: LSTM hidden state (h, c) tuple, or None for initial state
            
        Returns:
            log_prob: Log probability [batch, 1] or [batch, seq_len, 1]
            entropy: Action entropy [batch, 1] or [batch, seq_len, 1]
            value: State value [batch, 1] or [batch, seq_len, 1]
        """
        action_mean, value, _ = self.forward(obs, lstm_hidden)
        
        std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)  # Sum log probs for both outputs
        entropy = dist.entropy().sum(dim=-1, keepdim=True)  # Sum entropy for both outputs
        
        return log_prob, entropy, value
    
    @staticmethod
    def extract_obs(full_obs: np.ndarray) -> np.ndarray:
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

