# Copyright 2024 Alphaqraft
#
# Market Making Agent for Hierarchical RL
# 
# This agent learns optimal quoting based on market MICROSTRUCTURE only.
# Does NOT see target_inventory - inventory skew comes from Inventory Agent.
# Reward: Spread Capture + Fee Rebates (learns execution quality)

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MMAgent(nn.Module):
    """
    Market Making Agent - Microstructure-based execution
    
    Learns optimal bid/ask spreads based on orderbook microstructure.
    Does NOT see target_inventory - that skew is handled by Inventory Agent.
    Updates every step (500ms).
    
    Architecture: MLP [128, 64] + LSTM(64) - runs every step
    
    Input observations (18 dims):
        - Market microstructure signals (13): spread, depth, flow, volatility
        - Time since last fill [34] (1)
        - Quote mid distance [35] (1)
        - Previous actual bid/ask spread [36] (1)
        - Bid distance from mid [37] (1)
        - Ask distance from mid [38] (1)
    
    Output (2 actions):
        - bid_spread: [-1, 1] → controls bid quote width
        - ask_spread: [-1, 1] → controls ask quote width
        
    Note: Requote is handled automatically by the environment (smart requote).
    """
    
    def __init__(
        self,
        market_obs_dim: int = 18,  # Market signals (13) + time_since_fill (1) + quote_distance (1) + previous_spread (1) + bid_distance (1) + ask_distance (1)
        hidden_dim: int = 128,
        lstm_hidden: int = 64,
    ):
        super().__init__()
        
        self.market_obs_dim = market_obs_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # Encoder for observations (market microstructure only)
        self.encoder = nn.Sequential(
            nn.Linear(market_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        # Actor head: spread actions (continuous [-1, 1])
        self.spread_mean = nn.Linear(lstm_hidden, 2)  # bid_spread, ask_spread
        self.spread_log_std = nn.Parameter(torch.zeros(2))
        
        # Critic head
        self.critic = nn.Linear(lstm_hidden, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights: gain ~1.0 for hidden layers, 0.01 for output layers."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Output layers (spread_mean, critic): use small gain
                if 'spread_mean' in name or 'critic' in name:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    # Hidden layers: use standard gain
                    nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
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
        market_obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            market_obs: Market observations [batch, seq, market_obs_dim] or [batch, market_obs_dim]
            hidden: LSTM hidden state tuple (h, c)
            
        Returns:
            spread_mean: Mean of spread actions [batch, 2]
            value: State value [batch, 1]
            hidden: Updated LSTM hidden state
        """
        # Handle 2D input (add sequence dim)
        add_seq_dim = market_obs.ndim == 2
        if add_seq_dim:
            market_obs = market_obs.unsqueeze(1)
        
        batch_size, seq_len, _ = market_obs.shape
        
        # Encode market observations only (no target - inventory skew comes from Inv Agent)
        features = self.encoder(market_obs)  # [batch, seq, hidden//2]
        
        # LSTM
        if hidden is None:
            hidden = self._init_hidden(batch_size, market_obs.device)
        
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        # Actor output: spread actions
        spread_mean = torch.tanh(self.spread_mean(last_out))  # [-1, 1]
        
        # Critic
        value = self.critic(last_out)
        
        return spread_mean, value, hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)
    
    def get_action(
        self,
        market_obs: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Sample action for execution.
        
        Args:
            market_obs: Market observations [batch, market_obs_dim]
            hidden: LSTM hidden state
            deterministic: If True, return mean (no exploration)
            temperature: Scale exploration noise (0.0 = deterministic, 1.0 = full exploration)
            
        Returns:
            action: Spread actions [batch, 2] (bid_spread, ask_spread)
            log_prob: Log probability [batch, 1]
            value: State value [batch, 1]
            hidden: Updated hidden state
        """
        spread_mean, value, hidden = self.forward(market_obs, hidden)
        
        if temperature == 0.0:
            # Fully deterministic: use means (regardless of deterministic flag)
            action = spread_mean
            log_prob = torch.zeros(spread_mean.shape[0], 1, device=spread_mean.device)
            return action, log_prob, value, hidden
        
        # Sample spreads from Gaussian with temperature scaling
        # Clamp log_std to prevent explosion: [-5, 2] → std range [0.0067, 7.39]
        # Max clamp prevents std from exploding (which makes policy completely random)
        log_std_clamped = torch.clamp(self.spread_log_std, min=-5.0, max=2.0)
        spread_std = torch.exp(log_std_clamped).expand_as(spread_mean) * temperature
        spread_dist = torch.distributions.Normal(spread_mean, spread_std)
        action = spread_dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = spread_dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value, hidden
    
    def evaluate_actions(
        self,
        market_obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions (for PPO update).
        
        Args:
            market_obs: Market observations [batch, market_obs_dim]
            actions: Spread actions taken [batch, 2]
            hidden: LSTM hidden state
            
        Returns:
            log_prob: Log probability [batch, 1]
            entropy: Action entropy [batch, 1]
            value: State value [batch, 1]
        """
        spread_mean, value, _ = self.forward(market_obs, hidden)
        
        # Clamp log_std to prevent explosion: [-5, 2] → std range [0.0067, 7.39]
        # Max clamp prevents std from exploding (which makes policy completely random)
        log_std_clamped = torch.clamp(self.spread_log_std, min=-5.0, max=2.0)
        spread_std = torch.exp(log_std_clamped).expand_as(spread_mean)
        spread_dist = torch.distributions.Normal(spread_mean, spread_std)
        log_prob = spread_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = spread_dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value
    
    @staticmethod
    def extract_market_obs(full_obs: np.ndarray) -> np.ndarray:
        """
        Extract market observations for MM agent.
        
        Full observation (40 dims):
            [0-12]: Market signals (13)
            [13-16]: AMM flow signals (4)
            [17-24]: Trade signals (8)
            [25-35]: Agent state (11)
            [36]: Previous actual bid/ask spread (1)
            [37]: Bid distance from mid (1)
            [38]: Ask distance from mid (1)
            
        MM obs (18 dims):
            - Market signals [0-12] (13)
            - time_since_last_fill [34] (1)
            - quote_mid_distance [35] (1)
            - previous_actual_spread [36] (1)
            - bid_distance_from_mid [37] (1)
            - ask_distance_from_mid [38] (1)
        """
        indices = list(range(13)) + [34, 35, 36, 37, 38]
        if full_obs.ndim == 1:
            return full_obs[indices]
        else:
            return full_obs[:, indices]


# MM observation indices (market + execution signals)
MM_OBS_INDICES = list(range(13)) + [34, 35, 36, 37, 38]  # 13 market + time_since_fill + quote_distance + previous_spread + bid_distance + ask_distance
MARKET_OBS_DIM = len(MM_OBS_INDICES)  # 18
