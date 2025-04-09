import torch
import torch.nn as nn
from tianshou.data import Batch


class RecurrentActor(nn.Module):
    def __init__(self, device, action_dim=3, hidden_dim=64, gru_hidden_dim=128, num_layers=2, predict_steps=10):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers
        self.feature_dim = 242
        self.time_steps = 10
        self.max_action = 1
        self.position_dim = 18
        self.trade_dim = 6
        self.market_dim = 218
        self.device = device
        self.action_dim = action_dim
        self.predict_steps = predict_steps

        self.trade_fc = nn.Sequential(
            nn.Linear(self.trade_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 32), nn.ReLU()
        )

        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU()
        )

        self.gru = nn.GRU(96, gru_hidden_dim, num_layers=num_layers, batch_first=True)

        self.fusion_fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU()
        )

        self.pnl_predictor = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.log_std.weight.data.uniform_(-1, 1)
        self.log_std.bias.data.uniform_(-1, 1)

    def forward(self, obs, state=None):
        if isinstance(obs, Batch):
            obs = obs.obs

        obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        batch_size = obs.shape[0]
        obs = obs.view(batch_size, self.time_steps, -1)

        # State shape handling (input)
        if state is not None:
            if state.dim() == 3 and state.shape[0] == batch_size:  # [B, L, H]
                state = state.transpose(0, 1).contiguous()  # -> [L, B, H]
            elif state.dim() != 3 or state.shape[1] != batch_size:
                raise ValueError(f"Invalid state shape: {state.shape}")

        # Default state if None
        if state is None:
            state = torch.zeros(self.num_layers, batch_size, self.gru_hidden_dim, 
                              device=obs.device)

        # Original processing
        market_state = obs[:, :, :self.market_dim]
        position_state = obs[:, :, self.market_dim:self.market_dim + self.position_dim]
        trade_state = obs[:, :, self.market_dim + self.position_dim:self.market_dim + self.position_dim + self.trade_dim]

        position_out = self.position_fc(position_state)
        trade_out = self.trade_fc(trade_state)
        market_out = self.market_fc(market_state)
        x = torch.cat([trade_out, market_out, position_out], dim=-1)

        # GRU processing
        x, new_state = self.gru(x, state)
        x_last = x[:, -1, :]
        
        # Prepare outputs
        predictor_input = torch.cat([new_state[-1], x_last], dim=-1)
        predicted_pnl = self.pnl_predictor(predictor_input)
        x = self.fusion_fc(predictor_input)

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-10, 2)
        std = log_std.exp() + 1e-6

        return mean, std, new_state.transpose(0, 1), predicted_pnl
