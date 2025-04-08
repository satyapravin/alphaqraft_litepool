import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IQNCritic(nn.Module):
    def __init__(
        self,
        action_dim=3,
        hidden_dim=128,
        num_quantiles=64,
        quantile_embedding_dim=128,
        gru_hidden_dim=128,
        num_layers=2
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_layers = num_layers
        self.gru_hidden_dim = gru_hidden_dim
        self.feature_dim = 242
        self.time_steps = 10
        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = 218

        # State processing networks
        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.trade_fc = nn.Sequential(
            nn.Linear(self.trade_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # GRU for temporal processing
        self.gru = nn.GRU(96, gru_hidden_dim, num_layers=num_layers, batch_first=True)

        # Quantile processing
        self.cosine_layer = nn.Linear(quantile_embedding_dim, hidden_dim)

        # Final fusion network
        fusion_fc_input_dim = gru_hidden_dim * 2 + action_dim + hidden_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_fc_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Output layer
        self.q_values = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.LayerNorm(1)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action, taus=None, state_h=None):
        device = next(self.parameters()).device
        
        # Convert inputs to tensors if needed
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = torch.as_tensor(action, dtype=torch.float32, device=device)

        # Get original batch size (before quantile expansion)
        original_batch_size = state.size(0)
        
        # Reshape state if needed (handles both flattened and unflattened inputs)
        if state.dim() == 2:  # Flattened input [batch_size * quantiles, features]
            state = state.view(original_batch_size, self.time_steps, self.feature_dim)
        elif state.dim() == 3:  # Already unflattened [batch_size, time_steps, features]
            pass
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")

        # Process state through position/trade/market networks
        market_state = state[:, :, :self.market_dim]
        position_state = state[:, :, self.market_dim:self.market_dim + self.position_dim]
        trade_state = state[:, :, self.market_dim + self.position_dim:]

        position_out = self.position_fc(position_state)  # [batch_size, time_steps, 32]
        trade_out = self.trade_fc(trade_state)          # [batch_size, time_steps, 32]
        market_out = self.market_fc(market_state)       # [batch_size, time_steps, 32]

        # Concatenate and process through GRU
        x = torch.cat([trade_out, market_out, position_out], dim=-1)  # [batch_size, time_steps, 96]
        
        # Initialize hidden state if not provided
        if state_h is None:
            state_h = torch.zeros(self.num_layers, original_batch_size, self.gru_hidden_dim, device=device)
        
        x, new_state_h = self.gru(x, state_h)  # x: [batch_size, time_steps, gru_hidden_dim]
        x = x[:, -1, :]  # Take last timestep: [batch_size, gru_hidden_dim]

        # Combine with action and hidden state
        action = action.view(original_batch_size, -1)[:, :3]  # Ensure correct action dim
        x = torch.cat([x, new_state_h[-1], action], dim=-1)  # [batch_size, gru_hidden_dim + gru_hidden_dim + action_dim]

        # Quantile processing
        if taus is None:
            taus = torch.rand(original_batch_size, self.num_quantiles, device=device)
        elif taus.dim() == 1:  # Handle flattened taus
            taus = taus.view(original_batch_size, self.num_quantiles)
        elif taus.size(0) == original_batch_size * self.num_quantiles:
            taus = taus.view(original_batch_size, self.num_quantiles)

        # Generate quantile embeddings
        i_pi = torch.arange(1, self.quantile_embedding_dim + 1, device=device).float() * np.pi
        cos = torch.cos(taus.unsqueeze(-1) * i_pi)  # [batch_size, num_quantiles, quantile_embedding_dim]
        quantile_embedding = F.relu(self.cosine_layer(cos))  # [batch_size, num_quantiles, hidden_dim]

        # Expand state features for all quantiles
        x = x.unsqueeze(1).expand(-1, self.num_quantiles, -1)  # [batch_size, num_quantiles, feature_dim]
        
        # Reshape for parallel processing
        x = x.reshape(-1, x.size(-1))  # [batch_size * num_quantiles, feature_dim]
        quantile_embedding = quantile_embedding.reshape(-1, quantile_embedding.size(-1))  # [batch_size * num_quantiles, hidden_dim]

        print(x.shape)
        print(quantile_embedding.shape)
        # Concatenate features and quantile embeddings
        x = torch.cat([x, quantile_embedding], dim=-1)  # [batch_size * num_quantiles, feature_dim + hidden_dim]

        # Process through FC layers
        x = self.fusion_fc(x)
        q_values = self.q_values(x).view(original_batch_size, self.num_quantiles)  # [batch_size, num_quantiles]

        return q_values, new_state_h
