import numpy as np
import torch

class PPOCollector:
    def __init__(self, env, policy, buffer_size, device, gamma=0.99, gae_lambda=0.95):
        self.env = env
        self.policy = policy
        self.device = device
        self.buffer_size = buffer_size  # Total steps per environment
        self.n_envs = env.num_envs  # Parallel environments
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def collect(self):
        batch = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "infos": [],
        }

        # Reset environment
        obs, _ = self.env.reset()
        obs = np.asarray(obs)
        hidden_state = None  # For recurrent policies (optional)

        for step in range(self.buffer_size):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                action, log_prob, value, hidden_state = self.policy.forward(obs_tensor, hidden_state)

            action_cpu = action.cpu().numpy()

            # Step environment
            next_obs, reward, done, truncated, info_list = self.env.step(action_cpu)
            next_obs = np.asarray(next_obs)

            # Sanity check
            assert next_obs.shape[0] == self.n_envs, f"next_obs shape mismatch: got {next_obs.shape[0]}, expected {self.n_envs}"
            assert reward.shape[0] == self.n_envs, f"reward shape mismatch: got {reward.shape[0]}, expected {self.n_envs}"

            # Store
            batch["obs"].append(obs)
            batch["actions"].append(action_cpu)
            batch["rewards"].append(reward)
            batch["dones"].append(done)
            batch["log_probs"].append(log_prob.cpu().numpy())
            batch["values"].append(value.cpu().numpy())
            batch["infos"].append(info_list)

            # Move to next
            obs = next_obs

        # Convert lists to arrays
        for key in ["obs", "actions", "rewards", "dones", "log_probs", "values"]:
            batch[key] = np.asarray(batch[key])

        # Compute advantages and returns
        batch["advantages"], batch["returns"] = self._compute_gae_values(
            rewards=batch["rewards"],
            dones=batch["dones"],
            values=batch["values"]
        )

        return batch

    def _compute_gae_values(self, rewards, dones, values):
        """
        rewards: [T, n_envs]
        dones: [T, n_envs]
        values: [T, n_envs]
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        # Bootstrap with the last value
        next_value = np.zeros((self.n_envs,))
        gae = np.zeros((self.n_envs,))

        for step in reversed(range(self.buffer_size)):
            delta = rewards[step] + self.gamma * next_value * (1.0 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
            next_value = values[step]

        return advantages, returns
