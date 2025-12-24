# Copyright 2024 Alphaqraft
#
# Hierarchical Policy for Two-Agent Market Making
#
# Coordinates:
# - Inventory Agent (slow, strategic): decides target_inventory every N steps
# - MM Agent (fast, tactical): executes toward target every step

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any

from inventory_agent import InventoryAgent, INVENTORY_OBS_INDICES, INVENTORY_OBS_DIM
from mm_agent import MMAgent, MM_OBS_INDICES, MARKET_OBS_DIM


class HierarchicalPolicy(nn.Module):
    """
    Hierarchical Policy combining Inventory and MM agents.
    
    Inventory Agent:
        - Updates every `inventory_update_freq` steps (e.g., 100)
        - Learns from unrealized P&L
        - Outputs: target_inventory ∈ [-0.1, 0.1]
        
    MM Agent:
        - Updates every step
        - Learns from realized P&L + spread capture
        - Outputs: bid_spread, ask_spread, requote
    
    Combined action sent to environment: [bid_spread, ask_spread, target_inventory, requote]
    """
    
    def __init__(
        self,
        inventory_update_freq: int = 100,
        inventory_hidden: Tuple[int, ...] = (64, 32),
        mm_hidden: int = 128,
        mm_lstm: int = 64,
        target_range: float = 0.1,
        device: str = 'cpu',
    ):
        super().__init__()
        
        self.inventory_update_freq = inventory_update_freq
        self.target_range = target_range
        self.device = device
        
        # Create agents
        self.inventory_agent = InventoryAgent(
            obs_dim=INVENTORY_OBS_DIM,
            hidden_dims=inventory_hidden,
            target_range=target_range,
        )
        
        self.mm_agent = MMAgent(
            market_obs_dim=MARKET_OBS_DIM,
            target_dim=1,
            hidden_dim=mm_hidden,
            lstm_hidden=mm_lstm,
        )
        
        # State tracking (per environment)
        self.current_targets: Optional[torch.Tensor] = None
        self.step_counters: Optional[np.ndarray] = None
        self.mm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Move to device
        self.to(device)
    
    def reset(self, num_envs: int):
        """Reset state for new episodes."""
        self.current_targets = torch.zeros(num_envs, 1, device=self.device)
        self.step_counters = np.zeros(num_envs, dtype=np.int64)
        self.mm_hidden = None
    
    def reset_env(self, env_id: int):
        """Reset state for a specific environment (after episode end)."""
        if self.current_targets is not None:
            self.current_targets[env_id] = 0.0
        if self.step_counters is not None:
            self.step_counters[env_id] = 0
        # Note: LSTM hidden is shared, reset handled by batch dimension
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass for both agents.
        
        Args:
            obs: Full observations [batch, 32]
            deterministic: If True, use mean actions (no exploration)
            
        Returns:
            action: Combined action [batch, 4] for environment
            log_probs: Dict with 'inventory' and 'mm' log probs
            values: Dict with 'inventory' and 'mm' values
        """
        batch_size = obs.shape[0]
        
        # Initialize if needed
        if self.current_targets is None or self.current_targets.shape[0] != batch_size:
            self.reset(batch_size)
        
        # Extract observations for each agent
        inv_obs = self._extract_inventory_obs(obs)
        market_obs = self._extract_market_obs(obs)
        
        # Inventory agent: update target if it's time
        update_inventory = self._should_update_inventory()
        
        inv_action, inv_log_prob, inv_value = self.inventory_agent.get_action(
            inv_obs, deterministic=deterministic
        )
        
        # Update targets for envs that need updating
        if update_inventory.any():
            self.current_targets[update_inventory] = inv_action[update_inventory]
        
        # Increment step counters
        self.step_counters += 1
        
        # MM agent: act every step using current target
        mm_action, mm_log_prob, mm_value, self.mm_hidden = self.mm_agent.get_action(
            market_obs, 
            self.current_targets,
            self.mm_hidden,
            deterministic=deterministic,
        )
        
        # Combine into environment action format: [bid_spread, ask_spread, target, requote]
        action = torch.cat([
            mm_action[:, 0:1],  # bid_spread
            mm_action[:, 1:2],  # ask_spread
            self.current_targets,  # target_inventory
            mm_action[:, 2:3],  # requote
        ], dim=-1)
        
        log_probs = {
            'inventory': inv_log_prob,
            'mm': mm_log_prob,
        }
        
        values = {
            'inventory': inv_value,
            'mm': mm_value,
        }
        
        info = {
            'updated_inventory': update_inventory,
            'current_targets': self.current_targets.clone(),
        }
        
        return action, log_probs, values, info
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get action for environment (numpy interface).
        
        Args:
            obs: Full observations [batch, 32]
            deterministic: If True, use mean actions
            
        Returns:
            action: Combined action [batch, 4]
            info: Additional info (log_probs, values, etc.)
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action, log_probs, values, info = self.forward(obs_tensor, deterministic)
        
        return action.cpu().numpy(), {
            'log_prob_inv': log_probs['inventory'].cpu().numpy(),
            'log_prob_mm': log_probs['mm'].cpu().numpy(),
            'value_inv': values['inventory'].cpu().numpy(),
            'value_mm': values['mm'].cpu().numpy(),
            'updated_inventory': info['updated_inventory'],
            'targets': info['current_targets'].cpu().numpy(),
        }
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        inventory_actions: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Evaluate log probabilities for PPO update.
        
        Args:
            obs: Full observations [batch, 32]
            actions: MM actions [batch, 3] (bid_spread, ask_spread, requote)
            inventory_actions: Inventory actions [batch, 1]
            
        Returns:
            Dict with 'inventory' and 'mm' evaluation results
        """
        inv_obs = self._extract_inventory_obs(obs)
        market_obs = self._extract_market_obs(obs)
        
        # Evaluate inventory agent
        inv_log_prob, inv_entropy, inv_value = self.inventory_agent.evaluate_actions(
            inv_obs, inventory_actions
        )
        
        # Evaluate MM agent (use inventory actions as target)
        mm_log_prob, mm_entropy, mm_value = self.mm_agent.evaluate_actions(
            market_obs, inventory_actions, actions
        )
        
        return {
            'inventory': (inv_log_prob, inv_entropy, inv_value),
            'mm': (mm_log_prob, mm_entropy, mm_value),
        }
    
    def _should_update_inventory(self) -> np.ndarray:
        """Check which environments should update inventory target."""
        return (self.step_counters % self.inventory_update_freq) == 0
    
    def _extract_inventory_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract inventory-relevant observations."""
        return obs[:, INVENTORY_OBS_INDICES]
    
    def _extract_market_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract market observations (first 13 dims)."""
        return obs[:, MM_OBS_INDICES]
    
    def get_inventory_reward(
        self,
        prev_unrealized: np.ndarray,
        curr_unrealized: np.ndarray,
        initial_balance: float,
    ) -> np.ndarray:
        """
        Compute inventory agent reward from unrealized P&L change.
        
        Only computed when inventory is updated (every N steps).
        """
        delta = (curr_unrealized - prev_unrealized) / initial_balance
        return delta
    
    def get_mm_reward(
        self,
        realized_pnl_delta: np.ndarray,
        spread_capture_delta: np.ndarray,
        fee_delta: np.ndarray,
        initial_balance: float,
    ) -> np.ndarray:
        """
        Compute MM agent reward from execution quality.
        
        Computed every step.
        """
        # Normalize
        realized = realized_pnl_delta / initial_balance
        spread_capture = spread_capture_delta / initial_balance
        fees = fee_delta / initial_balance
        
        # Simple sum (equal weights, can tune later)
        return realized + spread_capture + fees
    
    def save(self, path: str):
        """Save both agents."""
        torch.save({
            'inventory_agent': self.inventory_agent.state_dict(),
            'mm_agent': self.mm_agent.state_dict(),
            'inventory_update_freq': self.inventory_update_freq,
            'target_range': self.target_range,
        }, path)
    
    def load(self, path: str):
        """Load both agents."""
        checkpoint = torch.load(path, map_location=self.device)
        self.inventory_agent.load_state_dict(checkpoint['inventory_agent'])
        self.mm_agent.load_state_dict(checkpoint['mm_agent'])
        self.inventory_update_freq = checkpoint.get('inventory_update_freq', 100)
        self.target_range = checkpoint.get('target_range', 0.1)


def create_hierarchical_policy(
    inventory_update_freq: int = 100,
    device: str = 'cpu',
) -> HierarchicalPolicy:
    """Factory function to create hierarchical policy with defaults."""
    return HierarchicalPolicy(
        inventory_update_freq=inventory_update_freq,
        inventory_hidden=(64, 32),
        mm_hidden=128,
        mm_lstm=64,
        target_range=0.1,
        device=device,
    )

