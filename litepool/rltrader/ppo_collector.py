import numpy as np
import torch
from tqdm import tqdm

class PPOCollector:
    def __init__(self, env, policy, n_steps, gamma=0.99, gae_lambda=0.95, device="cuda"):
        self.env = env
        self.policy = policy
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def collect(self):
        obs, info = self.env.reset()
        n_envs = self.env.num_envs

        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_values = []
        batch_rewards = []
        batch_dones = []
        batch_infos = []
        batch_states = []

        # Initialize per-env hidden states
        hidden_state = self.policy.init_hidden_state(batch_size=n_envs)
        hidden_state = self._to_device(hidden_state)

        for _ in tqdm(range(self.n_steps)):
            # Policy forward
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob, value, next_hidden_state = self.policy.forward(obs_tensor, hidden_state)

            action_np = action.detach().cpu().numpy()
            next_obs, reward, done, truncated, info = self.env.step(action_np)

            # Save batch data
            batch_obs.append(obs_tensor.cpu())
            batch_actions.append(action.detach().cpu())
            batch_log_probs.append(log_prob.detach().cpu())
            batch_values.append(value.detach().cpu())
            batch_rewards.append(torch.as_tensor(reward, dtype=torch.float32))
            batch_dones.append(torch.as_tensor(done, dtype=torch.float32))
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
            _, _, next_value, state  = self.policy.forward(obs_tensor, hidden_state)

        next_value = next_value.detach().cpu()

        # Stack batch
        batch_obs = torch.stack(batch_obs)
        batch_actions = torch.stack(batch_actions)
        batch_log_probs = torch.stack(batch_log_probs)
        batch_values = torch.stack(batch_values)
        batch_rewards = torch.stack(batch_rewards)
        batch_dones = torch.stack(batch_dones)
        batch_states = tuple(torch.stack([s[i] for s in batch_states], dim=0) for i in range(len(batch_states[0])))

        # Compute advantages and returns
        advantages, returns = self._compute_gae(
            rewards=batch_rewards,
            values=batch_values,
            dones=batch_dones,
            next_value=next_value
        )

        # Move everything to device
        batch = {
            "obs": batch_obs.to(self.device),
            "actions": batch_actions.to(self.device),
            "log_probs": batch_log_probs.to(self.device),
            "values": batch_values.to(self.device),
            "rewards": batch_rewards.to(self.device),
            "dones": batch_dones.to(self.device),
            "advantages": advantages.to(self.device),
            "returns": returns.to(self.device),
            "infos": batch_infos,
            "states": batch_states,
        }

        return batch

    def _compute_gae(self, rewards, values, dones, next_value):
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
