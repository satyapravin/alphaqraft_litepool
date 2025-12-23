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
from typing import Tuple, Dict, Any


class InventoryAgent(nn.Module):
    """
    Inventory Agent - Strategic position sizing
    
    Decides target_inventory based on market flow signals.
    Updates every N steps (e.g., 100 steps = 10 seconds).
    Learns from unrealized P&L (market direction).
    
    Architecture: Small MLP [64, 32] - runs infrequently
    
    Input observations (12 dims):
        - AMM signals: net_flow, flow_imbalance, inventory_delta, cumulative_flow (4)
        - Volatility: realized_vol (1)
        - Position state: current_leverage, deviation_from_target (2)
        - Price momentum: mid_diff, spread (2)
        - Trade signals: volume_imbalance, trade_intensity, buy_pressure (3)
    
    Output:
        - target_inventory: [-0.1, 0.1] (single continuous action)
    """
    
    def __init__(
        self,
        obs_dim: int = 12,
        hidden_dims: Tuple[int, ...] = (64, 32),
        target_range: float = 0.1,  # Max leverage target
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.target_range = target_range
        
        # Build MLP layers
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
        
        # Actor head: outputs mean and log_std for target_inventory
        self.actor_mean = nn.Linear(prev_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        
        # Critic head: outputs value estimate
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Inventory observations [batch, obs_dim]
            
        Returns:
            action_mean: Target inventory mean [batch, 1]
            value: State value estimate [batch, 1]
        """
        features = self.encoder(obs)
        
        # Actor: mean of target_inventory, scaled to [-target_range, target_range]
        action_mean = torch.tanh(self.actor_mean(features)) * self.target_range
        
        # Critic: value estimate
        value = self.critic(features)
        
        return action_mean, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action for execution.
        
        Args:
            obs: Observations [batch, obs_dim]
            deterministic: If True, return mean (no exploration)
            
        Returns:
            action: Sampled target_inventory [batch, 1]
            log_prob: Log probability of action [batch, 1]
            value: State value [batch, 1]
        """
        action_mean, value = self.forward(obs)
        
        if deterministic:
            return action_mean, torch.zeros_like(action_mean), value
        
        # Sample from Gaussian
        std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Clamp action to valid range
        action = torch.clamp(action, -self.target_range, self.target_range)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions (for PPO update).
        
        Args:
            obs: Observations [batch, obs_dim]
            actions: Actions taken [batch, 1]
            
        Returns:
            log_prob: Log probability [batch, 1]
            entropy: Action entropy [batch, 1]
            value: State value [batch, 1]
        """
        action_mean, value = self.forward(obs)
        
        std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy, value
    
    @staticmethod
    def extract_obs(full_obs: np.ndarray) -> np.ndarray:
        """
        Extract inventory-relevant observations from full observation.
        
        Full observation (32 dims):
            [0-12]: Market signals
            [13-16]: AMM flow signals  
            [17-24]: Trade signals
            [25-31]: Agent state
            
        Inventory obs (12 dims):
            - AMM: net_flow[13], flow_imbalance[14], inventory_delta[15], cumulative_flow[16]
            - Vol: We need to add this or use a proxy
            - Position: leverage[25], deviation[30]
            - Price: mid_diff from market signals
            - Trade: volume_imbalance[19], trade_intensity[20], buy_pressure[22]
        """
        # Select relevant indices
        indices = [
            13, 14, 15, 16,  # AMM signals (4)
            25,              # current_leverage (1)
            30,              # deviation_from_target (1)
            0,               # spread (proxy for vol) (1)
            1,               # mid_price change (1)
            19, 20, 22, 23,  # trade signals (4)
        ]
        
        if full_obs.ndim == 1:
            return full_obs[indices]
        else:
            return full_obs[:, indices]


# Observation indices for inventory agent (from full 32-dim obs)
INVENTORY_OBS_INDICES = [
    13, 14, 15, 16,  # AMM: net_flow, flow_imbalance, inventory_delta, cumulative_flow
    25,              # current_leverage
    30,              # deviation_from_target
    0,               # spread (vol proxy)
    1,               # mid change
    19, 20, 22, 23,  # trade: volume_imbalance, intensity, buy_pressure, sell_pressure
]
INVENTORY_OBS_DIM = len(INVENTORY_OBS_INDICES)  # 12

