# Copyright 2024 Alphaqraft
#
# Market Making Agent for Hierarchical RL
# 
# This agent learns HOW to execute toward a target (tactical, fast decisions)
# Takes target_inventory from Inventory Agent as input
# Reward: Realized P&L + Spread Capture (learns execution)

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MMAgent(nn.Module):
    """
    Market Making Agent - Tactical execution
    
    Executes toward the target_inventory set by Inventory Agent.
    Updates every step (100ms).
    Learns from realized P&L and spread capture (execution quality).
    
    Architecture: MLP [128, 64] + LSTM(64) - runs every step
    
    Input observations (16 dims):
        - Market microstructure signals (13)
        - Time since last fill [34] (1)
        - Quote mid distance [35] (1)
        - target_inventory from Inventory Agent (1)
    
    Output (3 actions):
        - bid_spread: [-1, 1] → controls bid quote aggressiveness
        - ask_spread: [-1, 1] → controls ask quote aggressiveness  
        - requote: probability of requoting orders
    """
    
    def __init__(
        self,
        market_obs_dim: int = 15,  # Market signals (13) + time_since_fill (1) + quote_distance (1)
        target_dim: int = 1,       # Target from inventory agent
        hidden_dim: int = 128,
        lstm_hidden: int = 64,
    ):
        super().__init__()
        
        self.market_obs_dim = market_obs_dim
        self.target_dim = target_dim
        self.total_obs_dim = market_obs_dim + target_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # Encoder for observations
        self.encoder = nn.Sequential(
            nn.Linear(self.total_obs_dim, hidden_dim),
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
        
        # Actor heads
        # Spread actions: continuous [-1, 1]
        self.spread_mean = nn.Linear(lstm_hidden, 2)  # bid_spread, ask_spread
        self.spread_log_std = nn.Parameter(torch.zeros(2))
        
        # Requote action: Bernoulli probability
        self.requote_logit = nn.Linear(lstm_hidden, 1)
        
        # Critic head
        self.critic = nn.Linear(lstm_hidden, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.01)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        market_obs: torch.Tensor,
        target: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            market_obs: Market observations [batch, seq, market_obs_dim] or [batch, market_obs_dim]
            target: Target inventory [batch, seq, 1] or [batch, 1]
            hidden: LSTM hidden state tuple (h, c)
            
        Returns:
            spread_mean: Mean of spread actions [batch, 2]
            requote_logit: Logit for requote probability [batch, 1]
            value: State value [batch, 1]
            hidden: Updated LSTM hidden state
        """
        # Handle 2D input (add sequence dim)
        add_seq_dim = market_obs.ndim == 2
        if add_seq_dim:
            market_obs = market_obs.unsqueeze(1)
            target = target.unsqueeze(1)
        
        batch_size, seq_len, _ = market_obs.shape
        
        # Concatenate market obs with target
        obs = torch.cat([market_obs, target], dim=-1)
        
        # Encode
        features = self.encoder(obs)  # [batch, seq, hidden//2]
        
        # LSTM
        if hidden is None:
            hidden = self._init_hidden(batch_size, market_obs.device)
        
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        # Actor outputs
        spread_mean = torch.tanh(self.spread_mean(last_out))  # [-1, 1]
        requote_logit = self.requote_logit(last_out)
        
        # Critic
        value = self.critic(last_out)
        
        return spread_mean, requote_logit, value, hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)
    
    def get_action(
        self,
        market_obs: torch.Tensor,
        target: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Sample action for execution.
        
        Args:
            market_obs: Market observations [batch, market_obs_dim]
            target: Target inventory [batch, 1]
            hidden: LSTM hidden state
            deterministic: If True, return mean (no exploration)
            
        Returns:
            action: Combined action [batch, 3] (bid_spread, ask_spread, requote)
            log_prob: Log probability [batch, 1]
            value: State value [batch, 1]
            hidden: Updated hidden state
        """
        spread_mean, requote_logit, value, hidden = self.forward(
            market_obs, target, hidden
        )
        
        if deterministic:
            # Use means
            spread_action = spread_mean
            requote_action = (torch.sigmoid(requote_logit) > 0.5).float()
            action = torch.cat([spread_action, requote_action], dim=-1)
            log_prob = torch.zeros(spread_mean.shape[0], 1, device=spread_mean.device)
            return action, log_prob, value, hidden
        
        # Sample spreads from Gaussian
        spread_std = torch.exp(self.spread_log_std).expand_as(spread_mean)
        spread_dist = torch.distributions.Normal(spread_mean, spread_std)
        spread_action = spread_dist.sample()
        spread_action = torch.clamp(spread_action, -1, 1)
        spread_log_prob = spread_dist.log_prob(spread_action).sum(dim=-1, keepdim=True)
        
        # Sample requote from Bernoulli
        requote_prob = torch.sigmoid(requote_logit)
        requote_dist = torch.distributions.Bernoulli(requote_prob)
        requote_action = requote_dist.sample()
        requote_log_prob = requote_dist.log_prob(requote_action)
        
        # Combine
        action = torch.cat([spread_action, requote_action], dim=-1)
        log_prob = spread_log_prob + requote_log_prob
        
        return action, log_prob, value, hidden
    
    def evaluate_actions(
        self,
        market_obs: torch.Tensor,
        target: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions (for PPO update).
        
        Args:
            market_obs: Market observations [batch, market_obs_dim]
            target: Target inventory [batch, 1]
            actions: Actions taken [batch, 3]
            hidden: LSTM hidden state
            
        Returns:
            log_prob: Log probability [batch, 1]
            entropy: Action entropy [batch, 1]
            value: State value [batch, 1]
        """
        spread_mean, requote_logit, value, _ = self.forward(
            market_obs, target, hidden
        )
        
        # Evaluate spreads
        spread_std = torch.exp(self.spread_log_std).expand_as(spread_mean)
        spread_dist = torch.distributions.Normal(spread_mean, spread_std)
        spread_actions = actions[:, :2]
        spread_log_prob = spread_dist.log_prob(spread_actions).sum(dim=-1, keepdim=True)
        spread_entropy = spread_dist.entropy().sum(dim=-1, keepdim=True)
        
        # Evaluate requote
        requote_prob = torch.sigmoid(requote_logit)
        requote_dist = torch.distributions.Bernoulli(requote_prob)
        requote_action = actions[:, 2:3]
        requote_log_prob = requote_dist.log_prob(requote_action)
        requote_entropy = requote_dist.entropy()
        
        # Combine
        log_prob = spread_log_prob + requote_log_prob
        entropy = spread_entropy + requote_entropy
        
        return log_prob, entropy, value
    
    @staticmethod
    def extract_market_obs(full_obs: np.ndarray) -> np.ndarray:
        """
        Extract market observations for MM agent.
        
        Full observation (36 dims):
            [0-12]: Market signals (13)
            [13-16]: AMM flow signals (4)
            [17-24]: Trade signals (8)
            [25-35]: Agent state (11)
            
        MM obs (15 dims):
            - Market signals [0-12] (13)
            - time_since_last_fill [34] (1)
            - quote_mid_distance [35] (1)
        """
        indices = list(range(13)) + [34, 35]
        if full_obs.ndim == 1:
            return full_obs[indices]
        else:
            return full_obs[:, indices]


# MM observation indices (market + execution signals)
MM_OBS_INDICES = list(range(13)) + [34, 35]  # 13 market + time_since_fill + quote_distance
MARKET_OBS_DIM = len(MM_OBS_INDICES)  # 15

