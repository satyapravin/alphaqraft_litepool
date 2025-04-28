import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentActorCritic(nn.Module):
    def __init__(self, action_dim=4, hidden_dim=128, gru_hidden_dim=128, num_layers=2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature dimensions
        self.market_dim = 218
        self.position_dim = 18
        self.trade_dim = 6

        self.feature_dim = self.market_dim + self.position_dim + self.trade_dim  # 242
        self.time_steps = 10
        self.input_dim = self.feature_dim * self.time_steps  # 2420

        self.action_dim = action_dim

        # Feature extractors
        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 32),
            nn.ReLU()
        )
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

        self.gru = nn.GRU(
            input_size=96,  # 32 + 32 + 32
            hidden_size=gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.actor_fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.critic_fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    def forward(self, obs, state=None):
        """Forward for sampling actions."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        batch_size = obs.shape[0]

        # Reshape input
        obs = obs.view(batch_size, self.time_steps, self.feature_dim)

        # Split into market, position, trade
        market = obs[:, :, :self.market_dim]
        position = obs[:, :, self.market_dim:self.market_dim + self.position_dim]
        trade = obs[:, :, self.market_dim + self.position_dim:]

        # Feature extraction
        market_feat = self.market_fc(market)
        position_feat = self.position_fc(position)
        trade_feat = self.trade_fc(trade)

        x = torch.cat([market_feat, position_feat, trade_feat], dim=-1)

        # GRU
        if state is None:
            state = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=self.device)

        gru_out, new_state = self.gru(x, state)
        features = gru_out[:, -1, :]  # last time step output

        # Actor and Critic heads
        actor_feat = self.actor_fc(features)
        critic_feat = self.critic_fc(features)

        mean = self.mean(actor_feat)
        log_std = self.log_std(actor_feat).clamp(-10, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(critic_feat).squeeze(-1)

        return dist, value, new_state

    def forward_sequence(self, obs_seq):
        """Forward for training with sequences."""
        seq_len, batch_size, _ = obs_seq.shape
        obs_seq = obs_seq.to(self.device)

        # Reshape input
        obs_seq = obs_seq.view(seq_len, batch_size, self.time_steps, self.feature_dim)

        market = obs_seq[:, :, :, :self.market_dim]
        position = obs_seq[:, :, :, self.market_dim:self.market_dim + self.position_dim]
        trade = obs_seq[:, :, :, self.market_dim + self.position_dim:]

        market_feat = self.market_fc(market)
        position_feat = self.position_fc(position)
        trade_feat = self.trade_fc(trade)

        x = torch.cat([market_feat, position_feat, trade_feat], dim=-1)

        # Merge time_steps inside batch
        x = x.view(seq_len * batch_size, self.time_steps, -1)

        gru_out, _ = self.gru(x)
        features = gru_out[:, -1, :]  # last time step output

        features = features.view(seq_len, batch_size, -1)

        actor_feat = self.actor_fc(features)
        critic_feat = self.critic_fc(features)

        mean = self.mean(actor_feat)
        log_std = self.log_std(actor_feat).clamp(-10, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(critic_feat).squeeze(-1)

        return dist, value

    def init_hidden_state(self, batch_size):
        """Initialize GRU hidden state for a batch."""
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=self.device)
