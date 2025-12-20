"""
Simple PPO Policy - no recurrence, no OT, just standard PPO.
"""
import torch
import torch.nn as nn
import time


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
            # Time the actual model forward pass
            forward_start = time.perf_counter()
            quote_dist, requote_dist, values, new_hidden_states = self.model(obs_tensor, hidden=batch_hidden)
            forward_time = time.perf_counter() - forward_start
            # Log if it's taking too long (>10ms)
            #if forward_time > 0.01:
            #    print(f"[Model Forward] batch_size={batch_size}, forward_time={forward_time*1000:.2f}ms")
            
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
        # These should already be clamped in collector, but add extra safety here
        # With reward_scale=1.0 and step rewards clamped to [-10, 10],
        # max return ≈ 200 (with gamma=0.995). Use [-200, 200] to allow learning actual scale
        # Clamp advantages more aggressively to prevent gradient explosion
        # Advantages directly affect policy gradients, so tighter clamping helps stability
        advantages = torch.clamp(advantages, -50.0, 50.0)  # More aggressive clamp for advantages
        returns = torch.clamp(returns, -200.0, 200.0)
        
        # CRITICAL: Store unnormalized returns BEFORE normalizing advantages
        # Returns are computed as: returns = advantages + values (in GAE)
        # So returns already reflect the scale of advantages
        # We need to use the original returns (before any normalization) for value loss
        returns_for_value = returns.clone()  # Keep unnormalized returns for value loss
        
        # Normalize advantages for policy learning (standard PPO practice)
        # This reduces variance in policy gradient estimates
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        new_log_probs, values, entropy = self.model.evaluate_actions(obs, actions)
        
        # EARLY DETECTION: Check for exploding values BEFORE any computation
        # If values are too large, skip update to prevent hanging
        if torch.isnan(values).any() or torch.isinf(values).any():
            print("WARNING: NaN/Inf in values, skipping update")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Check if values are exploding (too large) - skip update to prevent hanging
        max_value = torch.abs(values).max().item()
        if max_value > 10000.0:
            print(f"WARNING: Values exploding (max={max_value:.1f}), skipping update to prevent hang")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Clamp values to match returns range BEFORE computing loss
        # This prevents value loss explosion while allowing gradients to flow
        # Clamp values to match returns range [-200, 200] (with reward_scale=1.0)
        # This allows value function to learn the actual scale of returns
        values = torch.clamp(values, -200.0, 200.0)
        old_values = torch.clamp(old_values, -200.0, 200.0)
        
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
        # CRITICAL: Use unnormalized returns for value loss computation
        # Value function should learn to predict actual returns, not normalized ones
        # Advantages are normalized for policy learning, but returns must remain unnormalized
        # With reward_scale=1.0, returns should be in [-200, 200] range
        returns_for_value_clamped = torch.clamp(returns_for_value, -200.0, 200.0)
        
        # EARLY EXIT: If returns are exploding, skip update to prevent hanging
        max_return = torch.abs(returns_for_value_clamped).max().item()
        if max_return > 10000.0:
            print(f"WARNING: Returns exploding (max={max_return:.1f}), skipping update to prevent hang")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Value clipping for PPO (prevent large updates)
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_eps, self.clip_eps
        )
        
        # Compute value loss (squared error)
        # CRITICAL: Use unnormalized returns for value loss
        # Value function should learn to predict actual returns, not normalized ones
        # Advantages are normalized for policy learning, but returns must remain unnormalized
        value_loss_unclipped = (values - returns_for_value_clamped).pow(2)
        value_loss_clipped = (values_clipped - returns_for_value_clamped).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # EARLY EXIT: If value loss is too large, skip update to prevent hanging
        if value_loss.item() > 10000.0:
            print(f"WARNING: Value loss too large ({value_loss.item():.1f}), skipping update to prevent hang")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': value_loss.item(),
                'entropy': 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Additional safety: clamp value loss itself to prevent explosion
        # With vf_coef=0.1, value_loss=100 contributes only 10 to total loss
        # Max possible value loss with range [-200, 200]: 0.5 * (200 - (-200))^2 = 40,000
        # But in practice, value loss is normally much smaller (0.01-1.0 with reward_scale=1.0)
        # Clamp to [0, 100] to allow normal learning while preventing explosion
        value_loss = torch.clamp(value_loss, 0.0, 100.0)
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        
        # EARLY EXIT: If total loss is too large, skip update to prevent hanging
        total_loss_value = total_loss.item()
        if total_loss_value > 10000.0 or not torch.isfinite(total_loss):
            print(f"WARNING: Total loss too large or non-finite ({total_loss_value:.1f}), skipping update to prevent hang")
            return {
                'loss': total_loss_value if torch.isfinite(total_loss) else 0.0,
                'policy_loss': policy_loss.item() if torch.isfinite(policy_loss) else 0.0,
                'value_loss': value_loss.item() if torch.isfinite(value_loss) else 0.0,
                'entropy': entropy_loss.item() if torch.isfinite(entropy_loss) else 0.0,
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Optimize
        self.optimizer.zero_grad()
        
        # Use detach() to prevent gradient explosion if loss is still large
        # This is a safety measure - normally we want gradients, but if loss is large,
        # we want to prevent hanging during backward()
        try:
            total_loss.backward()
        except RuntimeError as e:
            if "non-finite" in str(e) or "NaN" in str(e) or "Inf" in str(e):
                print(f"WARNING: Error during backward(): {e}, skipping update")
                self.optimizer.zero_grad()
                return {
                    'loss': total_loss_value,
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy_loss.item(),
                    'approx_kl': 0.0,
                    'action_std': 0.0,
                }
            raise
        
        # Check for NaN/Inf gradients before clipping (fast check)
        has_bad_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_bad_grad = True
                    break
        
        if has_bad_grad:
            print("WARNING: NaN/Inf gradients detected, skipping update")
            self.optimizer.zero_grad()  # Clear gradients
            return {
                'loss': total_loss_value,
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy_loss.item(),
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Compute global gradient norm (same as clip_grad_norm_ uses)
        # This is the L2 norm across ALL parameters, not per-parameter
        # Per-parameter norms can be large (5000+) while global norm is still reasonable (< 1000)
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # If global gradient norm is extremely large, skip update to prevent hanging
        # With max_grad_norm=0.5, normal gradients should be < 100
        # If global norm > 10000, something is seriously wrong and clipping will be very slow
        if total_norm > 10000.0:
            print(f"WARNING: Global gradient norm too large ({total_norm:.1f}), skipping update to prevent hang")
            self.optimizer.zero_grad()
            return {
                'loss': total_loss_value,
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy_loss.item(),
                'approx_kl': 0.0,
                'action_std': 0.0,
            }
        
        # Gradient clipping (clips global norm to max_grad_norm=0.5)
        # This is fast even if per-parameter norms are large, as long as global norm is reasonable
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

