"""
Simple PPO Policy - no recurrence, no OT, just standard PPO.
"""
import torch
import torch.nn as nn


class SimplePPOPolicy:
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
    
    def get_action(self, obs, deterministic=False, hidden=None):
        """Get action for a single observation."""
        return self.model.get_action(obs, deterministic, hidden)
    
    def get_actions(self, obs, deterministic=False, hidden_states=None):
        """
        Get actions for batch of observations.
        
        Args:
            obs: [batch_size, obs_dim] numpy array
            deterministic: if True, return mean actions and threshold requote
            hidden_states: List of (h, c) tuples per environment, or None for fresh states
        Returns:
            actions: [batch_size, 4] numpy array - [3 quote params (bid_spread, ask_spread, target_inv), 1 requote]
            log_probs: [batch_size] numpy array
            values: [batch_size] numpy array
            new_hidden_states: Tuple of (h, c) where each is [1, batch_size, hidden_dim]
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        batch_size = obs_tensor.shape[0]
        
        # Prepare hidden states: combine per-environment hidden states into batch
        if hidden_states is not None and any(h is not None for h in hidden_states):
            # Stack hidden states from all environments
            h_list = []
            c_list = []
            for hidden in hidden_states:
                if hidden is not None:
                    h, c = hidden
                    h_list.append(h)  # [1, 1, hidden_dim]
                    c_list.append(c)
                else:
                    # Create zero hidden state for this environment
                    device = obs_tensor.device
                    lstm_hidden = self.model.lstm_hidden
                    h_zero = torch.zeros(1, 1, lstm_hidden, device=device)
                    c_zero = torch.zeros(1, 1, lstm_hidden, device=device)
                    h_list.append(h_zero)
                    c_list.append(c_zero)
            
            # Concatenate: [1, batch_size, hidden_dim]
            batch_hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))
        else:
            batch_hidden = None
        
        with torch.no_grad():
            # LSTM model returns (quote_dist, requote_dist, values, hidden_state)
            quote_dist, requote_dist, values, new_hidden_states = self.model(obs_tensor, hidden=batch_hidden)
            
            if deterministic:
                # Use mean for quote params, threshold 0.5 for requote
                quote_actions = quote_dist.mean
                requote_actions = (requote_dist.probs > 0.5).float()
            else:
                # Sample from distributions
                quote_actions = quote_dist.sample()
                requote_actions = requote_dist.sample()
            
            # Clamp quote parameters to [-1, 1]
            quote_actions = torch.clamp(quote_actions, -1.0, 1.0)
            
            # Convert requote from {0, 1} to {-1, 1} for compatibility
            requote_actions = requote_actions * 2.0 - 1.0  # {0,1} -> {-1,1}
            
            # Combine actions
            actions = torch.cat([quote_actions, requote_actions.unsqueeze(-1)], dim=-1)
            
            # Compute log probabilities
            quote_log_probs = quote_dist.log_prob(quote_actions).sum(-1)
            requote_log_probs = requote_dist.log_prob((requote_actions + 1.0) / 2.0)
            log_probs = quote_log_probs + requote_log_probs
        
        return actions.numpy(), log_probs.numpy(), values.numpy(), new_hidden_states
    
    def learn(self, batch):
        """
        Perform one PPO update step.
        
        Args:
            batch: dict with keys 'obs', 'actions', 'log_probs', 'values', 
                   'advantages', 'returns'
        Returns:
            dict with loss info
        """
        obs = batch['obs']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        old_values = batch['values']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Check for NaN in inputs
        if torch.isnan(obs).any() or torch.isnan(actions).any():
            print("WARNING: NaN detected in obs or actions, skipping update")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Clip advantages and returns to prevent numerical instability
        # With C++ scaling by 10000 and REWARD_SCALE=1.0, step rewards are ~1.0 (0.01% of balance)
        # Over 2048 steps with GAMMA=0.99, returns are typically in [-100, 100] range
        advantages = torch.clamp(advantages, -10.0, 10.0)
        returns = torch.clamp(returns, -500.0, 500.0)  # Conservative clamp for safety
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        new_log_probs, values, entropy = self.model.evaluate_actions(obs, actions)
        
        # Check for NaN in model outputs
        if torch.isnan(new_log_probs).any() or torch.isnan(values).any() or torch.isnan(entropy).any():
            print("WARNING: NaN detected in model outputs, skipping update")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Policy loss (clipped surrogate)
        # Clip log prob difference to prevent extreme ratios
        log_ratio = torch.clamp(new_log_probs - old_log_probs, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Check for NaN in policy loss
        if torch.isnan(policy_loss):
            print("WARNING: NaN in policy_loss, skipping update")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Value loss (clipped)
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_eps, self.clip_eps
        )
        value_loss_unclipped = (values - returns).pow(2)
        value_loss_clipped = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Check for NaN gradients before clipping
        has_nan_grad = False
        for param in self.model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("WARNING: NaN gradients detected, skipping update")
            self.optimizer.zero_grad()  # Clear gradients
            return {
                'loss': total_loss.item() if not torch.isnan(total_loss) else 0.0,
                'policy_loss': policy_loss.item() if not torch.isnan(policy_loss) else 0.0,
                'value_loss': value_loss.item() if not torch.isnan(value_loss) else 0.0,
                'entropy': entropy_loss.item() if not torch.isnan(entropy_loss) else 0.0,
                'approx_kl': 0.0,
                'action_std': self.model.quote_log_std.exp().mean().item() if not torch.isnan(self.model.quote_log_std).any() else 0.0,
            }
        
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Compute approx KL for logging
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
        
        # Get action std for logging (quote params only)
        with torch.no_grad():
            action_std = self.model.quote_log_std.exp().mean().item()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl,
            'action_std': action_std,
        }

