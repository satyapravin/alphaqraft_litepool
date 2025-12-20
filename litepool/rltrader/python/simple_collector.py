"""
Simple PPO Collector - no recurrence, no OT, just standard rollout collection.
"""
import numpy as np
import torch


class SimpleCollector:
    def __init__(self, env, policy, n_steps=2048, gamma=0.99, gae_lambda=0.95, reward_scale=1000.0, log_episodes=True):
        """
        Args:
            env: Vectorized environment (litepool)
            policy: SimplePPOPolicy
            n_steps: Number of steps to collect per rollout
            gamma: Discount factor
            gae_lambda: GAE lambda
            reward_scale: Scale factor for rewards (raw rewards are often ~0.0001)
            log_episodes: Whether to print episode metrics on reset
        """
        self.env = env
        self.policy = policy
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale = reward_scale
        self.log_episodes = log_episodes
        
        # Infer n_envs from observation shape (will be set on first reset)
        # Try to get it from env attribute, otherwise infer from obs shape
        self.n_envs = getattr(env, 'num_envs', None)
        
        self.obs = None
        self.dones = None
        self.episode_count = 0
        self.episode_rewards = None  # Track cumulative rewards per environment
        self.completed_episode_rewards = []  # Track rewards from completed episodes in this batch
        self.hidden_states = None  # Track LSTM hidden states per environment: list of (h, c) tuples
    
    def collect(self):
        """
        Collect n_steps of experience from all environments.
        
        Returns:
            batch: dict with trajectory data and computed advantages/returns
        """
        n_steps = self.n_steps
        
        # Initialize if first call
        if self.obs is None:
            self.obs, info = self.env.reset()
            # Infer n_envs from observation shape if not already set
            if self.n_envs is None:
                if isinstance(self.obs, np.ndarray):
                    self.n_envs = self.obs.shape[0] if len(self.obs.shape) > 1 else 1
                else:
                    self.n_envs = 1
            self.dones = np.zeros(self.n_envs, dtype=bool)
            self.episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
            # Initialize LSTM hidden states (zeros for all environments)
            self.hidden_states = [None] * self.n_envs
        
        # Reset completed episode rewards at start of each batch collection
        # This ensures we only track episodes completed in THIS batch
        self.completed_episode_rewards = []
        
        # Ensure n_envs is set (should be set by now)
        n_envs = self.n_envs
        
        # Storage
        obs_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        # Rollout
        for _ in range(n_steps):
            # Get actions from policy with current hidden states
            actions, log_probs, values, new_hidden_states = self.policy.get_actions(
                self.obs, hidden_states=self.hidden_states
            )
            
            # Store current step
            obs_list.append(self.obs.copy())
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            values_list.append(values)
            
            # Environment step
            next_obs, rewards, dones, truncs, infos = self.env.step(actions)
            
            rewards_list.append(rewards)
            dones_list.append(dones)
            infos_list.append(infos)
            
            # Update LSTM hidden states, resetting for environments that are done
            for env_id in range(self.n_envs):
                if dones[env_id]:
                    # Reset hidden state for this environment (new episode)
                    self.hidden_states[env_id] = None
                else:
                    # Update hidden state for this environment
                    if new_hidden_states is not None:
                        # Extract hidden state for this environment
                        # new_hidden_states is tuple of (h, c) where each is [1, batch_size, hidden_dim]
                        h, c = new_hidden_states
                        self.hidden_states[env_id] = (
                            h[:, env_id:env_id+1, :].detach().clone(),  # [1, 1, hidden_dim]
                            c[:, env_id:env_id+1, :].detach().clone()
                        )
            
            # Accumulate rewards per environment (before scaling)
            # Rewards from env are already normalized (fraction of balance)
            if isinstance(rewards, np.ndarray):
                self.episode_rewards += rewards
            else:
                self.episode_rewards[0] += rewards
            
            # Log episode metrics on reset
            # When done=True, infos is from the CURRENT step (terminal state).
            # Auto-reset happens on the NEXT step in litepool.
            if self.log_episodes and dones.any():
                self._log_episode_end(infos, dones)
            
            # Update state
            self.obs = next_obs
            self.dones = dones
        
        # Get bootstrap value with current hidden states
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32)
            # Prepare hidden states for bootstrap (combine all env hidden states)
            if self.hidden_states[0] is not None:
                # Stack hidden states: [1, n_envs, hidden_dim]
                h_list = [h for h, _ in self.hidden_states]
                c_list = [c for _, c in self.hidden_states]
                bootstrap_hidden = (
                    torch.cat(h_list, dim=1),  # [1, n_envs, hidden_dim]
                    torch.cat(c_list, dim=1)
                )
            else:
                bootstrap_hidden = None
            # LSTM model returns (quote_dist, requote_dist, value, hidden_state)
            _, _, bootstrap_values, _ = self.policy.model(obs_tensor, hidden=bootstrap_hidden)
            bootstrap_values = bootstrap_values.numpy()
        
        # Convert to tensors
        obs = torch.as_tensor(np.array(obs_list), dtype=torch.float32)
        actions = torch.as_tensor(np.array(actions_list), dtype=torch.float32)
        log_probs = torch.as_tensor(np.array(log_probs_list), dtype=torch.float32)
        values = torch.as_tensor(np.array(values_list), dtype=torch.float32)
        rewards_raw = torch.as_tensor(np.array(rewards_list), dtype=torch.float32)
        dones = torch.as_tensor(np.array(dones_list), dtype=torch.float32)
        
        # Apply reward scaling (if needed)
        # Note: C++ already scales rewards by 10000, so reward_scale should typically be 1.0
        rewards = rewards_raw * self.reward_scale
        
        # Compute GAE
        advantages, returns = self._compute_gae(
            rewards, values, dones, 
            torch.as_tensor(bootstrap_values, dtype=torch.float32)
        )
        
        # Flatten batch: [n_steps, n_envs, ...] -> [n_steps * n_envs, ...]
        batch = {
            'obs': obs.view(-1, obs.shape[-1]),
            'actions': actions.view(-1, actions.shape[-1]),
            'log_probs': log_probs.view(-1),
            'values': values.view(-1),
            'advantages': advantages.view(-1),
            'returns': returns.view(-1),
            'rewards': rewards,  # Scaled rewards for logging [n_steps, n_envs]
            'rewards_raw': rewards_raw,  # Raw rewards for diagnostics
            'dones': dones,  # Done flags [n_steps, n_envs] for goal_manager
            'infos': infos_list,
            'completed_episode_rewards': self.completed_episode_rewards.copy(),  # Cumulative rewards from completed episodes
        }
        
        return batch
    
    def _compute_gae(self, rewards, values, dones, next_values):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: [n_steps, n_envs]
            values: [n_steps, n_envs]
            dones: [n_steps, n_envs]
            next_values: [n_envs] bootstrap values
        Returns:
            advantages: [n_steps, n_envs]
            returns: [n_steps, n_envs]
        """
        n_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.n_envs)
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_values
            else:
                next_val = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def _log_episode_end(self, infos, dones):
        """Log metrics for environments that just finished an episode.
        
        Uses final_* keys which contain the terminal episode metrics cached
        before auto-reset (since regular info keys are reset to 0 after auto-reset).
        """
        for env_id in range(self.n_envs):
            if dones[env_id]:
                self.episode_count += 1
                
                # Use final_* keys which have terminal info cached before reset
                realized_pnl = 0.0
                unrealized_pnl = 0.0
                trade_count = 0.0
                fees = 0.0
                net_amount_btc = 0.0
                
                if isinstance(infos, dict):
                    # First try final_* keys (cached terminal info)
                    if 'final_realized_pnl' in infos:
                        val = infos['final_realized_pnl']
                        realized_pnl = float(val[env_id] if hasattr(val, '__getitem__') else val)
                    if 'final_unrealized_pnl' in infos:
                        val = infos['final_unrealized_pnl']
                        unrealized_pnl = float(val[env_id] if hasattr(val, '__getitem__') else val)
                    if 'final_trade_count' in infos:
                        val = infos['final_trade_count']
                        trade_count = float(val[env_id] if hasattr(val, '__getitem__') else val)
                    if 'final_fees' in infos:
                        val = infos['final_fees']
                        fees = float(val[env_id] if hasattr(val, '__getitem__') else val)
                    if 'final_net_amount_btc' in infos:
                        val = infos['final_net_amount_btc']
                        net_amount_btc = float(val[env_id] if hasattr(val, '__getitem__') else val)
                    
                    # Fallback to regular keys if final_* are 0 (episode ended normally, not auto-reset)
                    if realized_pnl == 0 and trade_count == 0:
                        if 'realized_pnl' in infos:
                            val = infos['realized_pnl']
                            realized_pnl = float(val[env_id] if hasattr(val, '__getitem__') else val)
                        if 'unrealized_pnl' in infos:
                            val = infos['unrealized_pnl']
                            unrealized_pnl = float(val[env_id] if hasattr(val, '__getitem__') else val)
                        if 'trade_count' in infos:
                            val = infos['trade_count']
                            trade_count = float(val[env_id] if hasattr(val, '__getitem__') else val)
                        if 'fees' in infos:
                            val = infos['fees']
                            fees = float(val[env_id] if hasattr(val, '__getitem__') else val)
                        if 'net_amount_btc' in infos:
                            val = infos['net_amount_btc']
                            net_amount_btc = float(val[env_id] if hasattr(val, '__getitem__') else val)
                
                net_pnl = realized_pnl + unrealized_pnl - fees
                
                # Get cumulative reward for this episode (normalized, fraction of balance)
                episode_reward = self.episode_rewards[env_id]
                
                # Track completed episode reward for epoch average
                self.completed_episode_rewards.append(episode_reward)
                
                # Reset reward accumulator for this environment
                self.episode_rewards[env_id] = 0.0
                
                print(f"  [Episode {self.episode_count:4d}] Env {env_id} | "
                      f"Reward {episode_reward:8.4f} | "
                      f"R.PnL ${realized_pnl:8.4f} | "
                      f"U.PnL ${unrealized_pnl:8.4f} | "
                      f"Fees ${fees:6.4f} | "
                      f"Net ${net_pnl:8.4f} | "
                      f"Trades {int(trade_count):3d} | "
                      f"Pos {net_amount_btc:7.5f} BTC")

