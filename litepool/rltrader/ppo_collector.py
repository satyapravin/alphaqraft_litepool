import torch
import numpy as np

class PPOCollector:
    def __init__(self, env, policy, rollout_len, device):
        self.env = env
        self.policy = policy
        self.rollout_len = rollout_len
        self.device = device

    def collect(self):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        infos = []
        ep_rewards = [[] for _ in range(self.env.num_envs)]

        hidden_state = None

        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        n_envs = obs.shape[0]

        for step in range(self.rollout_len):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                action, log_prob, value, hidden_state = self.policy.forward(obs_tensor, hidden_state)

                action = action.detach().cpu()
                log_prob = log_prob.detach().cpu()
                value = value.detach().cpu()

            action_env = action.numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_env)

            done = np.logical_or(terminated, truncated)

            next_obs = np.asarray(next_obs, dtype=np.float32)
            hidden_state = self.reset_hidden_state(hidden_state, done)

            obs_buf.append(torch.as_tensor(obs, device='cpu'))
            act_buf.append(action)
            logp_buf.append(log_prob)
            rew_buf.append(torch.as_tensor(reward, dtype=torch.float32))
            done_buf.append(torch.as_tensor(done, dtype=torch.float32))
            val_buf.append(value)
            infos.append(info)

            for i in range(n_envs):
                ep_rewards[i].append(reward[i])
                if done[i]:
                    ep_rewards[i] = []

            obs = next_obs

        batch = {
            'obs': torch.stack(obs_buf),    # [rollout_len, num_envs, feature_dim]
            'act': torch.stack(act_buf),
            'logp': torch.stack(logp_buf),
            'rew': torch.stack(rew_buf),
            'done': torch.stack(done_buf),
            'val': torch.stack(val_buf),
            'infos': infos,
            'ep_rewards': ep_rewards
        }

        return batch

    @staticmethod
    def reset_hidden_state(hidden_state, done):
        """Reset hidden states where episodes ended."""
        if hidden_state is None:
            return None

        done = torch.as_tensor(done, device=hidden_state.device, dtype=torch.float32).view(-1, 1)

        if isinstance(hidden_state, tuple):
            return tuple(h * (1.0 - done) for h in hidden_state)
        else:
            return hidden_state * (1.0 - done)
