import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentActorCritic(nn.Module):
    def __init__(self, action_dim=4, hidden_dim=128, gru_hidden_dim=128, num_layers=2, n_heads=4):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature dimensions
        self.market_dim = 218
        self.position_dim = 18
        self.trade_dim = 6
        self.embedding_dim = 32  # Reduced embedding dimension

        self.feature_dim = self.market_dim + self.position_dim + self.trade_dim
        self.time_steps = 2
        self.input_dim = self.feature_dim * self.time_steps
        self.action_dim = action_dim

        self.market_fc = nn.Sequential(
            nn.Linear(self.market_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim)
        )
        self.position_fc = nn.Sequential(
            nn.Linear(self.position_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim)
        )
        self.trade_fc = nn.Sequential(
            nn.Linear(self.trade_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim)
        )

        # GRU initialization - using PyTorch nn.GRU
        self.market_gru = nn.GRU(self.embedding_dim, gru_hidden_dim, num_layers=1, batch_first=False)
        self.position_gru = nn.GRU(self.embedding_dim, gru_hidden_dim, num_layers=1, batch_first=False)
        self.trade_gru = nn.GRU(self.embedding_dim, gru_hidden_dim, num_layers=1, batch_first=False)

        # Multihead Attention remains standard
        self.attn_input_dim = gru_hidden_dim * 3
        self.attention = nn.MultiheadAttention(embed_dim=self.attn_input_dim, num_heads=n_heads, batch_first=True)

        self.actor_fc = nn.Sequential(
            nn.Linear(self.attn_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Critic head - Deterministic with uncertainty penalty
        self.critic_fc = nn.Sequential(
            nn.Linear(self.attn_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Deterministic value head with uncertainty estimation
        self.value_head = nn.Linear(hidden_dim, 1)
        self.value_uncertainty = nn.Linear(hidden_dim, 1)  # Estimates log variance
        
        self.to(self.device)

    def forward(self, obs, state=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        batch_size = obs.shape[0]
        obs = obs.view(batch_size, self.time_steps, self.feature_dim)

        # Split and embed features
        market = obs[:, :, :self.market_dim]
        position = obs[:, :, self.market_dim:self.market_dim + self.position_dim]
        trade = obs[:, :, self.market_dim + self.position_dim:]

        market_feat = self.market_fc(market)
        position_feat = self.position_fc(position)
        trade_feat = self.trade_fc(trade)

        # GRU states
        if state is None:
            state = self.init_hidden_state(batch_size)

        # GRU expects input shape [seq_len, batch, input_size]
        market_feat = market_feat.transpose(0, 1)  # [time_steps, batch, features]
        position_feat = position_feat.transpose(0, 1)
        trade_feat = trade_feat.transpose(0, 1)

        market_out, market_h = self.market_gru(market_feat, state[0].contiguous())
        position_out, position_h = self.position_gru(position_feat, state[1].contiguous())
        trade_out, trade_h = self.trade_gru(trade_feat, state[2].contiguous())

        # Transpose back to [batch, time_steps, features]
        market_out = market_out.transpose(0, 1)
        position_out = position_out.transpose(0, 1)
        trade_out = trade_out.transpose(0, 1)

        # Take last time step output from each GRU
        combined = torch.cat([
            market_out[:, -1, :],
            position_out[:, -1, :],
            trade_out[:, -1, :]
        ], dim=-1).unsqueeze(1)

        # Multihead attention
        attn_out, _ = self.attention(combined, combined, combined)
        attn_out = attn_out.squeeze(1)

        # Actor-Critic heads
        actor_feat = self.actor_fc(attn_out)
        critic_feat = self.critic_fc(attn_out)

        # Policy distribution
        mean = self.mean(actor_feat)
        log_std = self.log_std(actor_feat).clamp(-5, 2)
        std = log_std.exp() + 1e-6
        dist = torch.distributions.Normal(mean, std)
        
        # Value with uncertainty penalty
        value = self.value_head(critic_feat).squeeze(-1)
        log_var = self.value_uncertainty(critic_feat).squeeze(-1)
        value_uncertainty = torch.exp(log_var)  # Convert to variance
        
        # Apply uncertainty penalty (higher variance -> lower value)
        value = value - 0.5 * value_uncertainty  # Penalize uncertain value estimates
        
        entropy = dist.entropy().sum(-1)
        new_state = (market_h, position_h, trade_h)
        
        return dist, value, entropy, new_state

    def forward_sequence(self, obs_seq, state=None):
        """
        obs_seq: [seq_len, batch_size, input_dim]
        state: tuple of 3 GRU states, each [num_layers, batch_size, hidden_dim]
        """
        seq_len, batch_size, _ = obs_seq.shape
        obs_seq = obs_seq.to(self.device)

        # Reshape to [seq_len, batch_size, time_steps, feature_dim]
        obs_seq = obs_seq.view(seq_len, batch_size, self.time_steps, self.feature_dim)

        # Split into components
        market = obs_seq[:, :, :, :self.market_dim]  # [seq_len, batch_size, time_steps, market_dim]
        position = obs_seq[:, :, :, self.market_dim:self.market_dim + self.position_dim]
        trade = obs_seq[:, :, :, self.market_dim + self.position_dim:]

        # Encode features (flatten to merge seq_len and batch_size)
        market_feat = self.market_fc(market.view(-1, self.time_steps, self.market_dim))
        position_feat = self.position_fc(position.view(-1, self.time_steps, self.position_dim))
        trade_feat = self.trade_fc(trade.view(-1, self.time_steps, self.trade_dim))

        # GRU input shape: [batch, time_steps, features]
        # batch = seq_len * batch_size
        flat_batch = seq_len * batch_size

        # Prepare GRU hidden states
        if state is None:
            state = self.init_hidden_state(batch_size)
        # Expand hidden state across seq_len
        expanded_state = tuple(s.repeat(1, seq_len, 1) for s in state)

        # GRU expects input shape [time_steps, batch, features]
        market_feat = market_feat.transpose(0, 1)
        position_feat = position_feat.transpose(0, 1)
        trade_feat = trade_feat.transpose(0, 1)

        # Run GRUs
        market_out, market_h = self.market_gru(market_feat, expanded_state[0].contiguous())
        position_out, position_h = self.position_gru(position_feat, expanded_state[1].contiguous())
        trade_out, trade_h = self.trade_gru(trade_feat, expanded_state[2].contiguous())

        # Transpose and select last time step to get [seq_len, batch_size, hidden_size]
        market_out = market_out.transpose(0, 1).reshape(flat_batch, self.time_steps, -1)[:, -1, :].reshape(seq_len, batch_size, -1)
        position_out = position_out.transpose(0, 1).reshape(flat_batch, self.time_steps, -1)[:, -1, :].reshape(seq_len, batch_size, -1)
        trade_out = trade_out.transpose(0, 1).reshape(flat_batch, self.time_steps, -1)[:, -1, :].reshape(seq_len, batch_size, -1)

        # Combine all last GRU outputs
        combined = torch.cat([market_out, position_out, trade_out], dim=-1)  # [seq_len, batch, hidden*3]

        # Self-attention expects input shape: [seq_len, batch, dim]
        attn_out, _ = self.attention(combined, combined, combined)  # [seq_len, batch, dim]

        # Actor-Critic heads
        actor_feat = self.actor_fc(attn_out)  # [seq_len, batch, hidden]
        critic_feat = self.critic_fc(attn_out)
        
        mean = self.mean(actor_feat)          # [seq_len, batch, action_dim]
        log_std = self.log_std(actor_feat).clamp(-5, 2)
        std = log_std.exp() + 1e-6

        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(critic_feat).squeeze(-1)  # [seq_len, batch]
        log_var = self.value_uncertainty(critic_feat).squeeze(-1)
        value_uncertainty = torch.exp(log_var)
        value = value - 0.5 * value_uncertainty  # Apply uncertainty penalty

        # Compute entropy for each timestep (sum over action dims)
        entropy = dist.entropy().sum(-1)  # [seq_len, batch]
        # Pack final hidden states (only last, not expanded)
        new_state = (market_h, position_h, trade_h)

        return dist, value, entropy, new_state

    def init_hidden_state(self, batch_size):
        return (
            torch.zeros(1, batch_size, 128, device=self.device),
            torch.zeros(1, batch_size, 128, device=self.device),
            torch.zeros(1, batch_size, 128, device=self.device)
        )

    def reset_hidden_state(self, batch_size):
        return self.init_hidden_state(batch_size)
