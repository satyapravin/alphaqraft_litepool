import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time

class PPOCollector:
    def __init__(self, env, policy, n_steps, gamma=0.99, gae_lambda=0.95, device="cuda", use_ot=False, ot_reg=0.01):
        self.env = env
        self.policy = policy
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.last_obs = None
        self.last_hidden_state = None
        self.use_ot = use_ot
        self.ot_reg = ot_reg  # Regularization parameter for Sinkhorn

    def collect(self):
        n_envs = self.env.num_envs

        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_values = []
        batch_rewards = []
        batch_dones = []
        batch_infos = []
        batch_states = []

        if self.last_obs is None:
            print("Resetting env")
            obs, info = self.env.reset()
            hidden_state = self.policy.init_hidden_state(batch_size=self.env.num_envs)
        else:
            obs = self.last_obs
            hidden_state = self.last_hidden_state
        
        hidden_state = self._to_device(hidden_state)

        for _ in tqdm(range(self.n_steps)):
            # Policy forward
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob, value, entropy, next_hidden_state = self.policy.forward(obs_tensor, hidden_state)

            action_np = action.detach().cpu().numpy()
            next_obs, reward, done, truncated, info = self.env.step(action_np)

            self.last_obs = obs_tensor.detach()
            self.last_hidden_state = tuple(h.detach() for h in next_hidden_state)

            # Save batch data (detached and on CPU)
            batch_obs.append(obs_tensor.detach().cpu())
            batch_actions.append(action.detach().cpu())
            batch_log_probs.append(log_prob.detach().cpu())
            batch_values.append(value.detach().cpu())
            batch_rewards.append(torch.as_tensor(reward, dtype=torch.float32).detach().cpu())
            batch_dones.append(torch.as_tensor(done, dtype=torch.float32).detach().cpu())
            batch_infos.append(info)
            batch_states.append(tuple(h.clone().detach().cpu() for h in hidden_state))

            # Reset hidden states and obs for envs that finished
            finished = np.logical_or(done, truncated)
            for env_id in range(n_envs):
                if finished[env_id]:
                    reset_obs, reset_info = self.env.reset(env_id)
                    next_obs[env_id] = reset_obs

                    # Handle tuple of 3 GRUs (market, position, trade)
                    if isinstance(hidden_state, tuple) and len(hidden_state) == 3:
                        hidden_state = tuple(h.detach().clone() for h in hidden_state)
                        next_hidden_state = tuple(h.detach().clone() for h in next_hidden_state)
                        for i in range(3):
                            hidden_state[i][:, env_id] = 0.0
                            next_hidden_state[i][:, env_id] = 0.0

                    # Handle LSTM: (h, c)
                    elif isinstance(hidden_state, tuple):
                        hidden_state = (
                            hidden_state[0].detach().clone(),
                            hidden_state[1].detach().clone()
                        )
                        next_hidden_state = (
                            next_hidden_state[0].detach().clone(),
                            next_hidden_state[1].detach().clone()
                        )
                        hidden_state[0][:, env_id] = 0.0
                        hidden_state[1][:, env_id] = 0.0
                        next_hidden_state[0][:, env_id] = 0.0
                        next_hidden_state[1][:, env_id] = 0.0

                    # Handle single GRU
                    else:
                        hidden_state = hidden_state.detach().clone()
                        next_hidden_state = next_hidden_state.detach().clone()
                        hidden_state[:, env_id] = 0.0
                        next_hidden_state[:, env_id] = 0.0

            obs = next_obs
            hidden_state = next_hidden_state

        # Bootstrap value for final obs
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            _, _, next_value, entropy, state = self.policy.forward(obs_tensor, hidden_state)

        next_value = next_value.detach().cpu()

        # Stack batch (all tensors are detached and on CPU)
        batch_obs = torch.stack(batch_obs)
        batch_actions = torch.stack(batch_actions)
        batch_log_probs = torch.stack(batch_log_probs)
        batch_values = torch.stack(batch_values)
        batch_rewards = torch.stack(batch_rewards)
        batch_dones = torch.stack(batch_dones)
        batch_states = tuple(torch.stack([s[i] for s in batch_states], dim=0) for i in range(len(batch_states[0])))

        # Compute advantages and returns
        start_time = time.time()
        advantages, returns = self._compute_advantages(
            rewards=batch_rewards,
            values=batch_values,
            dones=batch_dones,
            next_value=next_value,
            obs=batch_obs,
            states=batch_states
        )
        print(f"_compute_advantages took {time.time() - start_time:.2f} seconds")

        # Ensure advantages and returns are detached
        advantages = advantages.detach()
        returns = returns.detach()

        # Construct batch without moving to device
        batch = {
            "obs": batch_obs,  # [n_steps, n_envs, obs_dim], CPU
            "actions": batch_actions,  # [n_steps, n_envs, action_dim], CPU
            "log_probs": batch_log_probs,  # [n_steps, n_envs], CPU
            "values": batch_values,  # [n_steps, n_envs], CPU
            "rewards": batch_rewards,  # [n_steps, n_envs], CPU
            "dones": batch_dones,  # [n_steps, n_envs], CPU
            "advantages": advantages,  # [n_steps, n_envs], CPU
            "returns": returns,  # [n_steps, n_envs], CPU
            "infos": batch_infos,  # List of info dicts
            "states": batch_states,  # Tuple of [n_steps, num_layers, n_envs, hidden_dim], CPU
        }

        return batch

    def compute_sinkhorn_plan(self, cost_matrix, reg=0.01, max_iter=50):
        """
        Compute the Sinkhorn transport plan for OT.
        Args:
            cost_matrix: [n, m] cost matrix (e.g., KL divergence between action distributions)
            reg: Entropy regularization parameter
            max_iter: Maximum number of Sinkhorn iterations (reduced for speed)
        Returns:
            transport_plan: [n, m] transport plan matrix
        """
        # Normalize cost matrix
        cost_matrix = cost_matrix / (cost_matrix.max() + 1e-8)
        
        # Initialize dual variables
        n, m = cost_matrix.shape
        u = torch.ones(n, device=self.device) / n
        v = torch.ones(m, device=self.device) / m
        
        # Precompute kernel
        K = torch.exp(-cost_matrix / reg)
        
        # Sinkhorn iterations
        for _ in range(max_iter):
            u_new = 1.0 / (n * torch.matmul(K, v))
            v = 1.0 / (m * torch.matmul(K.t(), u_new))
            u = u_new
        
        # Compute transport plan
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        return transport_plan

    def _compute_advantages(self, rewards, values, dones, next_value, obs, states):
        """
        Compute advantages using OT-based credit assignment with conditional action distributions or GAE.
        Args:
            rewards: [n_steps, n_envs] rewards
            values: [n_steps, n_envs] value estimates
            dones: [n_steps, n_envs] done flags
            next_value: [n_envs] bootstrap value
            obs: [n_steps, n_envs, obs_dim] observations
            states: tuple of [n_steps, num_layers, n_envs, hidden_dim] GRU states
        Returns:
            advantages: [n_steps, n_envs] computed advantages
            returns: [n_steps, n_envs] computed returns
        """
        if not self.use_ot:
            # Fallback to original GAE
            return self._compute_gae(rewards, values, dones, next_value)

        n_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)  # [n_steps, n_envs]
        returns = torch.zeros_like(rewards)     # [n_steps, n_envs]

        # Move observations to device for policy forward pass
        obs = obs.to(self.device)

        for env in range(n_envs):
            # Extract per-environment data, move rewards to device
            env_rewards = rewards[:, env].to(self.device)  # [n_steps]
            env_values = values[:, env]    # [n_steps]
            env_dones = dones[:, env]      # [n_steps]
            env_obs = obs[:, env]          # [n_steps, obs_dim]
            noise_scale = 0.005 * torch.std(env_obs, dim=0, keepdim=True)
            env_obs = env_obs + torch.randn_like(env_obs) * noise_scale
           
            # Use initial hidden state for the environment, move to device
            env_states = tuple(s[0, :, env].unsqueeze(1).to(self.device) for s in states)  # [num_layers, 1, hidden_dim]

            # Compute conditional action distributions
            with torch.no_grad():
                start_time = time.time()
                dist, _, _, _ = self.policy.forward_train(env_obs.unsqueeze(1), env_states)  # [n_steps, 1, action_dim]
                print(f"Policy forward for env {env} took {time.time() - start_time:.2f} seconds")
                mean = dist.mean.squeeze(1)  # [n_steps, action_dim]
                std = dist.stddev.squeeze(1)  # [n_steps, action_dim]

            # Compute cost matrix using vectorized KL divergence
            start_time = time.time()
            log_std = torch.log(std)  # [n_steps, action_dim]
            # Reshape for broadcasting: [n_steps, 1, action_dim] and [1, n_steps, action_dim]
            mean_i = mean.unsqueeze(1)  # [n_steps, 1, action_dim]
            mean_j = mean.unsqueeze(0)  # [1, n_steps, action_dim]
            std_i = std.unsqueeze(1)    # [n_steps, 1, action_dim]
            std_j = std.unsqueeze(0)    # [1, n_steps, action_dim]
            log_std_i = log_std.unsqueeze(1)  # [n_steps, 1, action_dim]
            log_std_j = log_std.unsqueeze(0)  # [1, n_steps, action_dim]

            # KL divergence between Gaussians: D_KL(N(mean_i, std_i) || N(mean_j, std_j))
            kl = (log_std_j - log_std_i +
                  (std_i.pow(2) + (mean_i - mean_j).pow(2)) / (2 * std_j.pow(2)) -
                  0.5).sum(dim=-1)  # [n_steps, n_steps]
            cost_matrix = kl
            print(f"KL divergence for env {env} took {time.time() - start_time:.2f} seconds")

            # Compute OT transport plan
            start_time = time.time()
            transport_plan = self.compute_sinkhorn_plan(cost_matrix, reg=self.ot_reg)  # [n_steps, n_steps]
            print(f"Sinkhorn for env {env} took {time.time() - start_time:.2f} seconds")

            # Compute returns using OT-weighted rewards
            start_time = time.time()
            discounted_rewards = torch.zeros_like(env_rewards)
            last_return = next_value[env]
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_value_t = next_value[env]
                    next_non_terminal = 1.0 - env_dones[t]
                else:
                    next_value_t = env_values[t + 1]
                    next_non_terminal = 1.0 - env_dones[t]
                
                # OT-weighted reward: sum transport plan-weighted rewards from future steps
                ot_weighted_reward = torch.sum(transport_plan[t, t:] * env_rewards[t:], dim=-1)
                discounted_rewards[t] = ot_weighted_reward + self.gamma * next_value_t * next_non_terminal
                last_return = discounted_rewards[t]
            print(f"OT-weighted rewards for env {env} took {time.time() - start_time:.2f} seconds")

            returns[:, env] = discounted_rewards
            advantages[:, env] = discounted_rewards - env_values.to(self.device)

        # Normalize advantages across all environments and time steps
        advantages = advantages.to(self.device)
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        print(f"Advantage statistics: Mean = {adv_mean.item():.4f}, Std = {adv_std.item():.4f}")
        return advantages, returns

    def _compute_gae(self, rewards, values, dones, next_value):
        """
        Original GAE computation (unchanged).
        """
        n_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(n_envs, dtype=torch.float32)

        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_vals = next_value
                next_non_terminal = 1.0 - dones[step]
            else:
                next_vals = values[step + 1]
                next_non_terminal = 1.0 - dones[step]

            delta = rewards[step] + self.gamma * next_vals * next_non_terminal - values[step]
            advantages[step] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def _to_device(self, hidden_state):
        if isinstance(hidden_state, tuple):
            return tuple(h.to(self.device) for h in hidden_state)
        else:
            return hidden_state.to(self.device)
