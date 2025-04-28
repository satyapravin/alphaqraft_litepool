import torch
import numpy as np
from tianshou.data import Batch

class PPOCollector:
    def __init__(self, env, policy, rollout_len):
        self.env = env
        self.policy = policy
        self.rollout_len = rollout_len
        self.device = policy.device
        self.obs, _ = env.reset()
        self.done = np.zeros(env.num_envs, dtype=bool)
        self.num_envs = env.num_envs
        self.state = torch.zeros(policy.model.gru.num_layers, self.num_envs, policy.model.gru.hidden_size, device=self.device)

    def collect(self):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, info_buf = [], [], [], [], [], [], []

        for _ in range(self.rollout_len):
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
            action, log_prob, value, new_state = self.policy.forward(obs_tensor, self.state)

            # Step environment
            next_obs, reward, terminated, truncated, infos = self.env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            obs_buf.append(self.obs)
            act_buf.append(action.cpu().numpy())
            logp_buf.append(log_prob.cpu().detach().numpy())
            val_buf.append(value.cpu().detach().numpy())
            rew_buf.append(reward)
            done_buf.append(done)
            info_buf.append(infos)  # <-- Save info at each step

            # Reset GRU hidden state for finished environments
            for env_idx, done_flag in enumerate(done):
                if done_flag:
                    new_state[:, env_idx, :] = 0.0

            self.obs = next_obs
            self.done = done
            self.state = new_state

        batch = Batch(
            obs=torch.tensor(np.array(obs_buf), dtype=torch.float32).reshape(-1, *obs_buf[0].shape[2:]),
            act=torch.tensor(np.array(act_buf), dtype=torch.float32).reshape(-1, *act_buf[0].shape[2:]),
            log_prob=torch.tensor(np.array(logp_buf), dtype=torch.float32).reshape(-1),
            value=torch.tensor(np.array(val_buf), dtype=torch.float32).reshape(-1),
            rew=torch.tensor(np.array(rew_buf), dtype=torch.float32).reshape(-1),
            done=torch.tensor(np.array(done_buf), dtype=torch.float32).reshape(-1),
            infos=info_buf  # <-- Include all info dicts
        )
        return batch
