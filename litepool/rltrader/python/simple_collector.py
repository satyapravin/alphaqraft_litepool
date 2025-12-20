"""
Simple PPO Collector - no recurrence, no OT, just standard rollout collection.
"""
import numpy as np
import torch
import time


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
        self.episode_steps = None  # Track step count per environment
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
            self.episode_steps = np.zeros(self.n_envs, dtype=np.int32)  # Track steps per episode
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
        truncs_list = []  # Track truncs to check if all envs are done
        infos_list = []
        
        # Timing tracking
        step_times = []
        last_log_step = 0
        
        # Rollout
        for step_idx in range(n_steps):
            # Timing logging every 500 steps
            if step_idx % 500 == 0 and step_idx > 0:
                if len(step_times) > 0:
                    avg_time = np.mean(step_times)
                    total_time = np.sum(step_times)
                    print(f"[Python Timing] Steps {last_log_step}-{step_idx}: avg={avg_time*1000:.2f}ms, total={total_time:.3f}s, count={len(step_times)}")
                    step_times.clear()
                    last_log_step = step_idx
            
            # Early exit check BEFORE calling env.step() to avoid blocking
            # If all environments were done in previous step and we have enough data, break
            if step_idx > 0 and len(dones_list) > 0:
                prev_dones = dones_list[-1]
                prev_truncs = truncs_list[-1] if len(truncs_list) > 0 else np.zeros_like(prev_dones)
                prev_all_done = np.all(prev_dones | prev_truncs)
                min_steps = int(0.75 * n_steps)  # Collect at least 75% of steps
                if prev_all_done and step_idx >= min_steps:
                    # Break early if all environments are done
                    # Pop the last elements to keep lists in sync (they were appended before env.step())
                    if len(obs_list) > len(rewards_list):
                        obs_list.pop()
                        actions_list.pop()
                        log_probs_list.pop()
                        values_list.pop()
                    break
            
            # Get actions from policy with current hidden states
            actions, log_probs, values, new_hidden_states = self.policy.get_actions(
                self.obs, hidden_states=self.hidden_states
            )
            
            # Store current step
            obs_list.append(self.obs.copy())
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            values_list.append(values)
            
            # Environment step with timeout protection
            try:
                step_start = time.time()
                next_obs, rewards, dones, truncs, infos = self.env.step(actions)
                step_time = time.time() - step_start
                step_times.append(step_time)
            except Exception as e:
                print(f"ERROR in env.step() at rollout step {step_idx}: {e}")
                # If we have some data collected, return what we have
                if len(obs_list) > 0:
                    print(f"WARNING: Returning partial batch with {len(obs_list)} steps due to env.step() error")
                    break
                raise
            
            rewards_list.append(rewards)
            dones_list.append(dones)
            truncs_list.append(truncs)  # Track truncs for early exit check
            infos_list.append(infos)
            
            # IMPORTANT: In gymnasium, dones is actually "terminated" (done & ~trunc)
            # But we want to log episodes when they end, regardless of truncation
            # So we need to check both dones (terminated) and truncs
            episode_ended = dones | truncs  # Episode ends if terminated OR truncated
            
            # Increment episode step count only for environments that are NOT done
            # If an episode just ended, it already completed its steps, so don't increment
            # This prevents the off-by-one error where steps show 2049 instead of 2048
            for env_id in range(self.n_envs):
                if not episode_ended[env_id]:
                    self.episode_steps[env_id] += 1
            
            # Update LSTM hidden states, resetting for environments that are done
            for env_id in range(self.n_envs):
                if dones[env_id]:
                    # Reset hidden state for this environment (new episode)
                    self.hidden_states[env_id] = None
                    # Note: episode_steps[env_id] will be reset in _log_episode_end
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
            # Clamp step rewards to prevent cumulative rewards from exploding
            if isinstance(rewards, np.ndarray):
                # Clamp each step reward to prevent cumulative explosion
                rewards_clamped = np.clip(rewards, -10.0, 10.0)  # Much tighter clamp for step rewards
                self.episode_rewards += rewards_clamped
            else:
                rewards_clamped = np.clip(rewards, -10.0, 10.0)
                self.episode_rewards[0] += rewards_clamped
            
            # Log episode metrics on reset
            # When done=True, infos is from the CURRENT step (terminal state).
            # Auto-reset happens on the NEXT step in litepool.
            # Use episode_ended (dones | truncs) to catch all episode endings
            if self.log_episodes and episode_ended.any():
                self._log_episode_end(infos, episode_ended)
            
            # Update state
            self.obs = next_obs
            self.dones = dones
        
        # If we broke early due to all environments being done, we need to handle the partial batch
        # Make sure we have at least some data
        if len(obs_list) == 0:
            raise RuntimeError("No data collected - all environments may be stuck")
        
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
        # Handle case where we collected fewer steps than n_steps
        actual_n_steps = len(obs_list)
        if actual_n_steps < n_steps:
            print(f"INFO: Collected {actual_n_steps} steps instead of {n_steps} (all environments may be done)")
        
        # CRITICAL: Ensure all lists have the same length
        # When we break early, we've already fixed the mismatch by popping the last items
        # But double-check to be safe
        min_length = min(len(obs_list), len(actions_list), len(log_probs_list), 
                        len(values_list), len(rewards_list), len(dones_list))
        if min_length != actual_n_steps:
            print(f"WARNING: Mismatch in list lengths (obs={len(obs_list)}, actions={len(actions_list)}, "
                  f"log_probs={len(log_probs_list)}, values={len(values_list)}, "
                  f"rewards={len(rewards_list)}, dones={len(dones_list)}), trimming to {min_length} steps")
            obs_list = obs_list[:min_length]
            actions_list = actions_list[:min_length]
            log_probs_list = log_probs_list[:min_length]
            values_list = values_list[:min_length]
            rewards_list = rewards_list[:min_length]
            dones_list = dones_list[:min_length]
            actual_n_steps = min_length
        
        obs = torch.as_tensor(np.array(obs_list), dtype=torch.float32)
        actions = torch.as_tensor(np.array(actions_list), dtype=torch.float32)
        log_probs = torch.as_tensor(np.array(log_probs_list), dtype=torch.float32)
        values = torch.as_tensor(np.array(values_list), dtype=torch.float32)
        rewards_raw = torch.as_tensor(np.array(rewards_list), dtype=torch.float32)
        dones = torch.as_tensor(np.array(dones_list), dtype=torch.float32)
        
        # Verify all tensors have the same first dimension
        assert obs.shape[0] == actions.shape[0] == log_probs.shape[0] == values.shape[0] == rewards_raw.shape[0] == dones.shape[0], \
            f"Shape mismatch: obs={obs.shape[0]}, actions={actions.shape[0]}, log_probs={log_probs.shape[0]}, " \
            f"values={values.shape[0]}, rewards={rewards_raw.shape[0]}, dones={dones.shape[0]}"
        
        # Apply reward scaling (if needed)
        # Note: C++ already scales rewards by 10000, so reward_scale should typically be 1.0
        rewards = rewards_raw * self.reward_scale
        
        # Clamp rewards to prevent numerical instability
        # With C++ scaling by 10000 and Python reward_scale=1.0, step rewards are naturally small
        # Step rewards are normalized by balance, so they're fractions (0.01 = 1% of balance)
        # Clamp to [-10, 10] to prevent outliers while allowing normal rewards
        # Over 2048 steps, max return ≈ 200 (with gamma=0.995), which is reasonable
        rewards = torch.clamp(rewards, -10.0, 10.0)  # Conservative clamp for natural reward scale
        
        # Debug: Print reward statistics to diagnose small returns
        # Clamp values BEFORE computing GAE to prevent returns from exploding
        # With max_episode_steps=2048, gamma=0.995, and rewards [-10, 10] (after reward_scale=1.0),
        # max return ≈ 200. But in practice, returns are much smaller due to small step rewards
        # Use [-200, 200] to allow learning actual scale while preventing explosion
        values = torch.clamp(values, -200.0, 200.0)
        bootstrap_values_tensor = torch.as_tensor(bootstrap_values, dtype=torch.float32)
        bootstrap_values_tensor = torch.clamp(bootstrap_values_tensor, -200.0, 200.0)
        
        # Compute GAE
        advantages, returns = self._compute_gae(
            rewards, values, dones, 
            bootstrap_values_tensor
        )
        
        # Clamp returns and advantages to prevent numerical instability
        # With reward_scale=1.0, returns are naturally small (e.g., [-3.5, 0.5])
        # Clamp to [-200, 200] to allow learning actual scale while preventing explosionreturns = torch.clamp(returns, -200.0, 200.0)
        advantages = torch.clamp(advantages, -200.0, 200.0)  # Advantages are typically smaller
        
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
                
                # Get episode step count before resetting
                episode_steps = int(self.episode_steps[env_id])
                
                # Reset reward accumulator and step count for this environment
                self.episode_rewards[env_id] = 0.0
                self.episode_steps[env_id] = 0
                
                print(f"  [Episode {self.episode_count:4d}] Env {env_id} | Steps {episode_steps:5d} | "
                      f"Reward {episode_reward:8.4f} | "
                      f"R.PnL ${realized_pnl:8.4f} | "
                      f"U.PnL ${unrealized_pnl:8.4f} | "
                      f"Fees ${fees:6.4f} | "
                      f"Net ${net_pnl:8.4f} | "
                      f"Trades {int(trade_count):3d} | "
                      f"Pos {net_amount_btc:7.5f} BTC")

