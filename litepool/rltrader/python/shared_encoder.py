# Copyright 2024 Alphaqraft
#
# Shared Encoder with Attention for Hierarchical RL
# 
# Both MM and Inventory agents use a shared encoder that processes all observations,
# then each agent uses attention to focus on relevant signals.

import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """
    Shared encoder that processes all 42 observation dimensions.
    Both MM and Inventory agents use this encoder, then apply task-specific attention.
    """
    
    def __init__(
        self,
        obs_dim: int = 42,  # Full observation dimension (updated from 40)
        hidden_dim: int = 128,
        encoder_layers: int = 2,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        
        # Build encoder layers
        layers = []
        prev_dim = obs_dim
        for i in range(encoder_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, elementwise_affine=True),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize encoder weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations.
        
        Args:
            obs: Observations [batch, obs_dim] or [batch, seq_len, obs_dim]
            
        Returns:
            encoded: Encoded features [batch, hidden_dim] or [batch, seq_len, hidden_dim]
        """
        # Don't clone here - let the caller handle cloning to avoid double cloning
        # The agents already clone observations before calling this
        return self.encoder(obs)


class AttentionModule(nn.Module):
    """
    Attention mechanism that allows agents to focus on relevant observation signals.
    Uses self-attention over the encoded features.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to encoded features.
        
        Args:
            x: Encoded features [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            
        Returns:
            attended: Attended features [batch, hidden_dim] or [batch, seq_len, hidden_dim]
        """
        # Add sequence dimension if needed (for single timestep)
        add_seq_dim = x.ndim == 2
        if add_seq_dim:
            x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Clone inputs to attention to prevent in-place modifications
        # MultiheadAttention might modify inputs in-place in some cases
        x_q = x.clone()
        x_k = x.clone()
        x_v = x.clone()
        
        # Self-attention: query, key, value all from x
        attn_out, _ = self.attention(x_q, x_k, x_v)
        # Residual connection: explicitly create new tensor to avoid in-place modification issues
        x_residual = x + attn_out
        x = self.norm1(x_residual)
        
        # Feedforward
        ff_out = self.ff(x)
        # Residual connection: explicitly create new tensor to avoid in-place modification issues
        x_residual2 = x + ff_out
        x = self.norm2(x_residual2)
        
        # Remove sequence dimension if we added it
        if add_seq_dim:
            x = x.squeeze(1)  # [batch, hidden_dim]
        
        return x

