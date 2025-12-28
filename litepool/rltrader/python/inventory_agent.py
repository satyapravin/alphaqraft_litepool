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
        - AMM signals: net_flow, cumulative_flow (2 stable signals only, removed noisy flow_imbalance and inventory_delta)
        - Position state: leverage, deviation_from_target, target_ema, entry_distance (4)
        - Quote spreads: actual_bid_spread, actual_ask_spread, total_spread, mid_change (4)
        - Trade signals: volume_imbalance, trade_intensity, buy_pressure, sell_pressure (4)
    
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
        # Scale actor_mean output by 0.5 before tanh to prevent saturation and encourage smoother actions
        # This allows the network to output intermediate values more easily
        raw_mean = self.actor_mean(features) * 0.5
        action_mean = torch.tanh(raw_mean) * self.target_range
        
        # Critic: value estimate
        value = self.critic(features)
        
        return action_mean, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action for execution.
        
        Args:
            obs: Observations [batch, obs_dim]
            deterministic: If True, return mean (no exploration)
            temperature: Scale exploration noise (0.0 = deterministic, 1.0 = full exploration)
            
        Returns:
            action: Sampled target_inventory [batch, 1]
            log_prob: Log probability of action [batch, 1]
            value: State value [batch, 1]
        """
        action_mean, value = self.forward(obs)
        
        if temperature == 0.0:
            # Fully deterministic: use means (regardless of deterministic flag)
            return action_mean, torch.zeros_like(action_mean), value
        
        # Sample from Gaussian with temperature scaling
        # Clamp log_std to prevent explosion: [-5, 1.5] → std range [0.0067, 4.48]
        # Max clamp prevents std from exploding (which makes policy completely random)
        log_std_clamped = torch.clamp(self.actor_log_std, min=-5.0, max=1.5)
        std = torch.exp(log_std_clamped).expand_as(action_mean) * temperature
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
        
        # Clamp log_std to prevent explosion: [-5, 1.5] → std range [0.0067, 4.48]
        # Max clamp prevents std from exploding (which makes policy completely random)
        log_std_clamped = torch.clamp(self.actor_log_std, min=-5.0, max=1.5)
        std = torch.exp(log_std_clamped).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
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

