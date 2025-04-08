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

        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.trade_fc = nn.Sequential(
            nn.Linear(self.trade_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 32), nn.ReLU()
        )

        self.gru = nn.GRU(96, gru_hidden_dim, num_layers=num_layers, batch_first=True)

        self.cosine_layer = nn.Linear(quantile_embedding_dim, hidden_dim)

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

        self.q_values = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.LayerNorm(1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action, taus=None, state_h=None):
        device = next(self.parameters()).device

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device)

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        state_shape = state.shape

        if len(state_shape) > 2:
            state = state[:, -1, :]

        batch_size = state.shape[0]
        if action.shape[0] != batch_size:
            action = action.expand(batch_size, -1)

        expected_flat_dim = self.time_steps * self.feature_dim
        if state.dim() == 2 and state.shape[1] == expected_flat_dim:
            state = state.view(batch_size, self.time_steps, self.feature_dim)

        market_state = state[:, :, :self.market_dim]
        position_state = state[:, :, self.market_dim:self.market_dim + self.position_dim]
        trade_state = state[:, :, self.market_dim + self.position_dim:]

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)

        x = torch.cat([trade_out, market_out, position_out], dim=-1)

        if state_h is None:
            state_h = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, device=device)

        x, new_state_h = self.gru(x, state_h)
        x = x[:, -1, :]

        x = torch.cat([x, new_state_h[-1], action], dim=-1)

        if taus is None:
            taus = torch.rand(batch_size, self.num_quantiles, device=device)

        i_pi = torch.arange(1, self.quantile_embedding_dim + 1, device=device).float() * np.pi
        cos = torch.cos(taus.unsqueeze(-1) * i_pi)
        quantile_embedding = F.relu(self.cosine_layer(cos))

        x = x.unsqueeze(1).expand(-1, self.num_quantiles, -1)
        x = torch.cat([x, quantile_embedding], dim=-1)

        x = x.view(-1, x.shape[-1])
        x = self.fusion_fc(x)
        q_values = self.q_values(x)
        q_values = q_values.view(batch_size, self.num_quantiles)

        return q_values, new_state_h
