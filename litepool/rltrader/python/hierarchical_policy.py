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

from inventory_agent import InventoryAgent
from mm_agent import MMAgent
from shared_encoder import SharedEncoder


class HierarchicalPolicy(nn.Module):
    """
    Hierarchical Policy combining Inventory and MM agents.
    
    Inventory Agent:
        - Updates every `inventory_update_freq` steps (default: 1, smoothed by EMA in C++)
        - Learns from total P&L (spread capture + unrealized)
        - Outputs: target_inventory ∈ [-target_range, +target_range], risk_aversion ∈ [0, 1]
        
    MM Agent:
        - Updates every step
        - Learns from spread capture (execution quality)
        - Outputs: bid_spread ∈ [0, 1], ask_spread ∈ [0, 1]
        
    Combined action sent to environment: [bid_spread, ask_spread, target_inventory, risk_aversion]
    (4 dimensions total)
    
    Note: Requote is handled automatically by the environment (smart requote).
    """
    
    def __init__(
        self,
        obs_dim: int = 42,  # Full observation dimension (must match environment)
        inventory_update_freq: int = 1,  # Default: every step (smoothed by EMA in C++)
        inventory_lstm: int = 32,
        mm_lstm: int = 64,
        hidden_dim: int = 128,  # Shared encoder hidden dimension
        target_range: float = 1.0,  # Default from hierarchical_config.py
        attention_heads: int = 4,
        device: str = 'cpu',
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.inventory_update_freq = inventory_update_freq
        self.target_range = target_range
        self.device = device
        
        # Create shared encoder (both agents use this)
        self.shared_encoder = SharedEncoder(obs_dim, hidden_dim)
        
        # Create agents with shared encoder
        self.inventory_agent = InventoryAgent(
            obs_dim=obs_dim,
            shared_encoder=self.shared_encoder,
            hidden_dim=hidden_dim,
            lstm_hidden=inventory_lstm,
            target_range=target_range,
            attention_heads=attention_heads,
        )
        
        self.mm_agent = MMAgent(
            obs_dim=obs_dim,
            shared_encoder=self.shared_encoder,
            hidden_dim=hidden_dim,
            lstm_hidden=mm_lstm,
            attention_heads=attention_heads,
        )
        
        # State tracking (per environment) - using torch tensors for device compatibility
        self.current_targets: Optional[torch.Tensor] = None
        self.current_risk_aversion: Optional[torch.Tensor] = None
        self.step_counters: Optional[torch.Tensor] = None  # Use torch instead of numpy for device compatibility
        self.mm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.inv_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Move to device
        self.to(device)
    
    def reset(self, num_envs: int):
        """Reset state for new episodes (all environments)."""
        self.current_targets = torch.zeros(num_envs, 1, device=self.device)
        self.current_risk_aversion = torch.ones(num_envs, 1, device=self.device) * 0.5  # Default risk_aversion = 0.5 [0, 1]
        self.step_counters = torch.zeros(num_envs, dtype=torch.int64, device=self.device)  # Use torch for device compatibility
        # Initialize LSTM hidden states for all environments (will be created on first forward pass)
        # Shape: (1, num_envs, lstm_hidden) for both h and c
        self.mm_hidden = None  # Will be initialized on first forward pass
        self.inv_hidden = None  # Will be initialized on first forward pass
    
    def reset_env(self, env_id: int):
        """Reset state for a specific environment (after episode end)."""
        if self.current_targets is not None:
            self.current_targets[env_id] = 0.0
        if self.current_risk_aversion is not None:
            self.current_risk_aversion[env_id] = 0.5
        if self.step_counters is not None:
            self.step_counters[env_id] = 0
        # Reset LSTM hidden state for this specific environment
        # LSTM hidden state shape: (1, batch_size, lstm_hidden) for both h and c
        # Detach from computation graph before modifying to avoid gradient errors
        if self.mm_hidden is not None:
            h, c = self.mm_hidden
            # Safety check: ensure batch size matches
            if env_id < h.shape[1] and env_id < c.shape[1]:
                # Detach and clone to avoid modifying tensors in computation graph
                h = h.detach().clone()
                c = c.detach().clone()
                # Zero out hidden state for this environment in the batch
                h[:, env_id, :] = 0.0
                c[:, env_id, :] = 0.0
                self.mm_hidden = (h, c)
        
        # Reset inventory agent LSTM hidden state for this specific environment
        if self.inv_hidden is not None:
            h, c = self.inv_hidden
            # Safety check: ensure batch size matches
            if env_id < h.shape[1] and env_id < c.shape[1]:
                # Detach and clone to avoid modifying tensors in computation graph
                h = h.detach().clone()
                c = c.detach().clone()
                # Zero out hidden state for this environment in the batch
                h[:, env_id, :] = 0.0
                c[:, env_id, :] = 0.0
                self.inv_hidden = (h, c)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass for both agents.
        
        Args:
            obs: Full observations [batch, 42]
            deterministic: If True, use mean actions (no exploration)
            temperature: Scale exploration noise (0.0 = deterministic, 1.0 = full exploration)
            
        Returns:
            action: Combined action [batch, 5] for environment (bid_spread, ask_spread, requote, target_inventory, risk_aversion)
            log_probs: Dict with 'inventory' and 'mm' log probs
            values: Dict with 'inventory' and 'mm' values
        """
        batch_size = obs.shape[0]
        
        # Initialize if needed
        if self.current_targets is None or self.current_targets.shape[0] != batch_size:
            self.reset(batch_size)
        
        # Both agents see all observations (shared encoder + attention handles filtering)
        # Initialize LSTM hidden states if needed (first call or batch size changed)
        if self.inv_hidden is None:
            # Initialize hidden state for all environments
            self.inv_hidden = self.inventory_agent._init_hidden(batch_size, self.device)
        if self.mm_hidden is None:
            # Initialize hidden state for all environments
            self.mm_hidden = self.mm_agent._init_hidden(batch_size, self.device)
        
        # Inventory agent: update target if it's time
        update_inventory = self._should_update_inventory()
        
        # Clone and detach hidden states BEFORE passing to LSTM to prevent in-place modification errors
        # PyTorch's LSTM can modify hidden states in-place, so we need to clone them
        inv_hidden_input = None
        if self.inv_hidden is not None:
            h, c = self.inv_hidden
            inv_hidden_input = (h.detach().clone(), c.detach().clone())
        
        inv_action, inv_log_prob, inv_value, inv_hidden_new = self.inventory_agent.get_action(
            obs, inv_hidden_input, deterministic=deterministic, temperature=temperature
        )
        # Detach and clone hidden states after LSTM forward pass
        if inv_hidden_new is not None:
            h, c = inv_hidden_new
            self.inv_hidden = (h.detach().clone(), c.detach().clone())
        
        # Update targets and risk aversion for envs that need updating
        if update_inventory.any():
            self.current_targets[update_inventory] = inv_action[update_inventory, 0:1]  # target_inventory
            self.current_risk_aversion[update_inventory] = inv_action[update_inventory, 1:2]  # risk_aversion
        
        # Increment step counters
        self.step_counters += 1
        
        # MM agent: act every step (attention focuses on microstructure signals)
        # Clone and detach hidden states BEFORE passing to LSTM to prevent in-place modification errors
        # PyTorch's LSTM can modify hidden states in-place, so we need to clone them
        mm_hidden_input = None
        if self.mm_hidden is not None:
            h, c = self.mm_hidden
            mm_hidden_input = (h.detach().clone(), c.detach().clone())
        
        mm_action, mm_log_prob, mm_value, mm_hidden_new = self.mm_agent.get_action(
            obs,
            mm_hidden_input,
            deterministic=deterministic,
            temperature=temperature,
        )
        # Detach and clone hidden states after LSTM forward pass
        if mm_hidden_new is not None:
            h, c = mm_hidden_new
            self.mm_hidden = (h.detach().clone(), c.detach().clone())
        
        # Combine into environment action format: [bid_spread, ask_spread, requote, target_inventory, risk_aversion]
        action = torch.cat([
            mm_action[:, 0:1],  # bid_spread
            mm_action[:, 1:2],  # ask_spread
            mm_action[:, 2:3],  # requote decision
            self.current_targets,  # target_inventory
            self.current_risk_aversion,  # risk_aversion
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
            'current_risk_aversion': self.current_risk_aversion.clone(),
        }
        
        return action, log_probs, values, info
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get action for environment (numpy interface).
        
        Args:
            obs: Full observations [batch, 42]
            deterministic: If True, use mean actions (temperature=0.0 for fully deterministic)
            temperature: Scale exploration noise (0.0 = deterministic, 1.0 = full exploration)
            
        Returns:
            action: Combined action [batch, 5] (bid_spread, ask_spread, requote, target_inventory, risk_aversion)
            info: Additional info (log_probs, values, etc.)
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action, log_probs, values, info = self.forward(obs_tensor, deterministic, temperature)
        
        return action.cpu().numpy(), {
            'log_prob_inv': log_probs['inventory'].cpu().numpy(),
            'log_prob_mm': log_probs['mm'].cpu().numpy(),
            'value_inv': values['inventory'].cpu().numpy(),
            'value_mm': values['mm'].cpu().numpy(),
            'updated_inventory': info['updated_inventory'],
            'targets': info['current_targets'].cpu().numpy(),
            'risk_aversion': info['current_risk_aversion'].cpu().numpy(),
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
            obs: Full observations [batch, obs_dim]
            actions: MM actions [batch, 2] (bid_spread, ask_spread)
            inventory_actions: Inventory actions [batch, 2] (target_inventory, risk_aversion)
            
        Returns:
            Dict with 'inventory' and 'mm' evaluation results
        """
        # Observations should already be fresh tensors from the buffer (numpy arrays converted to tensors)
        # The shared encoder will clone them if needed to ensure fresh computation graph
        # Both agents see all observations (shared encoder + attention handles filtering)
        # Evaluate inventory agent
        inv_log_prob, inv_entropy, inv_value = self.inventory_agent.evaluate_actions(
            obs, inventory_actions
        )
        
        # Evaluate MM agent (attention focuses on microstructure signals)
        mm_log_prob, mm_entropy, mm_value = self.mm_agent.evaluate_actions(
            obs, actions
        )
        
        return {
            'inventory': (inv_log_prob, inv_entropy, inv_value),
            'mm': (mm_log_prob, mm_entropy, mm_value),
        }
    
    def _should_update_inventory(self) -> np.ndarray:
        """Check which environments should update inventory target."""
        # Returns numpy array for compatibility with indexing operations
        return ((self.step_counters % self.inventory_update_freq) == 0).cpu().numpy()
    
    # NOTE: Observation extraction removed - both agents now see all observations via shared encoder + attention
    # NOTE: Reward calculation methods removed - rewards come directly from environment
    # The environment (rltrader_litepool.h) computes mm_reward and inv_reward in the info dict
    # These are extracted in hierarchical_ppo.py collect_rollout() method
    
    def save(self, path: str):
        """Save all components (shared encoder + both agents)."""
        torch.save({
            'shared_encoder': self.shared_encoder.state_dict(),
            'inventory_agent': self.inventory_agent.state_dict(),
            'mm_agent': self.mm_agent.state_dict(),
            'inventory_update_freq': self.inventory_update_freq,
            'target_range': self.target_range,
        }, path)
    
    def load(self, path: str):
        """Load both agents."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'shared_encoder' in checkpoint:
            self.shared_encoder.load_state_dict(checkpoint['shared_encoder'])
        self.inventory_agent.load_state_dict(checkpoint['inventory_agent'])
        self.mm_agent.load_state_dict(checkpoint['mm_agent'])
        self.inventory_update_freq = checkpoint.get('inventory_update_freq', 1)  # Default from config
        self.target_range = checkpoint.get('target_range', 1.0)  # Default from config


def create_hierarchical_policy(
    obs_dim: int = 42,  # Full observation dimension (must match environment)
    inventory_update_freq: int = 1,  # Default from config (1 = every step, smoothed by EMA in C++)
    device: str = 'cpu',
    target_range: float = 1.0,  # Default from config (matches hierarchical_config.py)
    hidden_dim: int = 128,  # Shared encoder hidden dimension
    attention_heads: int = 4,
) -> HierarchicalPolicy:
    """Factory function to create hierarchical policy with defaults matching config."""
    return HierarchicalPolicy(
        obs_dim=obs_dim,
        inventory_update_freq=inventory_update_freq,
        inventory_lstm=32,
        mm_lstm=64,
        hidden_dim=hidden_dim,
        target_range=target_range,
        attention_heads=attention_heads,
        device=device,
    )

