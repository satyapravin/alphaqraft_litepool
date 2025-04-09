import time
import numpy as np
import torch
from tianshou.data import Collector, Batch
from dataclasses import dataclass

@dataclass
class CollectStats:
    n_ep: int
    n_st: int
    rews: np.ndarray
    lens: np.ndarray
    rew: float
    len: float
    rew_std: float
    len_std: float
    n_collected_steps: int
    n_collected_episodes: int
    returns_stat: object
    lens_stat: object
    rewards_stat: object
    episodes: int 
    reward_sum: float
    length_sum: int
    collect_time: float
    step_time: float 
    returns: np.ndarray
    lengths: np.ndarray
    continuous_step: int

class CPUCollector(Collector):
    def __init__(self, policy, env, num_of_envs, buffer=None, device='cpu', seq_len=300, **kwargs):
        super().__init__(policy, env, buffer, **kwargs)
        self.device = device
        self.model_device = 'cuda'
        self.num_of_envs = num_of_envs
        self.seq_len = seq_len
        self.env_active = False
        self.continuous_step_count = 0
        self.data = Batch()
        self.reset_batch_data()

        self.state_h1 = torch.zeros(
            policy.critic1.num_layers, env.num_envs, policy.critic1.gru_hidden_dim,
            device=self.model_device
        )
        self.state_h2 = torch.zeros(
            policy.critic2.num_layers, env.num_envs, policy.critic2.gru_hidden_dim,
            device=self.model_device
        )
        # Initialize state for policy
        self.data.state = torch.zeros(
            self.num_of_envs, policy.critic1.num_layers, policy.critic1.gru_hidden_dim,
            device=self.model_device
        )  # [64, 2, 128]

    def reset_batch_data(self):
        self.data.obs = None
        self.data.info = None
        self.data.act = None
        self.data.state = torch.zeros(
            self.num_of_envs, self.policy.critic1.num_layers, self.policy.critic1.gru_hidden_dim,
            device=self.model_device
        )  # [64, 2, 128]
        self.env_active = False

    def reset_env(self, gym_reset_kwargs=None):
        if not self.env_active:
            obs = self.env.reset(**gym_reset_kwargs or {})
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = None
            self.data.obs = obs if isinstance(obs, np.ndarray) else np.array(obs)
            self.data.info = info
            self.env_active = True
            self.state_h1.zero_()
            self.state_h2.zero_()
            self.data.state.zero_()  # Reset state to [64, 2, 128]
        return self.data.obs, self.data.info

    def _collect(self, random=None, render=None, n_step=None, n_episode=None, gym_reset_kwargs=None):
        start_time = time.time()
        local_step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        step_rews = []
        episode_lens_dict = {i: 0 for i in range(self.num_of_envs)}
        episode_rews_dict = {i: 0.0 for i in range(self.num_of_envs)}

        obs_seq = [[] for _ in range(self.num_of_envs)]
        act_seq = [[] for _ in range(self.num_of_envs)]
        rew_seq = [[] for _ in range(self.num_of_envs)]
        done_seq = [[] for _ in range(self.num_of_envs)]
        terminated_seq = [[] for _ in range(self.num_of_envs)]
        truncated_seq = [[] for _ in range(self.num_of_envs)]
        obs_next_seq = [[] for _ in range(self.num_of_envs)]
        state_seq = [[] for _ in range(self.num_of_envs)]
        state_h1_seq = [[] for _ in range(self.num_of_envs)]
        state_h2_seq = [[] for _ in range(self.num_of_envs)]

        while True:
            if self.data.obs is None:
                obs, info = self.reset_env(gym_reset_kwargs)
                self.data.obs = obs
                self.data.info = info

            obs_batch = Batch(obs=torch.as_tensor(self.data.obs, device=self.model_device))  # [64, 2420]
            # Ensure state is [num_envs, num_layers, hidden_dim] for policy
            if self.data.state is not None and self.data.state.shape[0] != self.num_of_envs:
                self.data.state = self.data.state.transpose(0, 1).contiguous()  # Fix if needed

            with torch.no_grad():
                if random:
                    action_space = self.env.action_space
                    act = np.array([action_space.sample() for _ in range(self.num_of_envs)])
                    result = Batch(act=torch.as_tensor(act, device=self.model_device))
                else:
                    result = self.policy(obs_batch, state=self.data.state.transpose(0, 1).contiguous() if self.data.state is not None else None)
            self.data.act = result.act.cpu().numpy()
            self.data.state = result.state.transpose(0, 1).contiguous() if result.state is not None else None

            result = self.env.step(self.data.act)
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            else:
                obs_next, rew, done, info = result
                terminated = done
                truncated = np.zeros_like(done)

            self.data.obs_next = obs_next if isinstance(obs_next, np.ndarray) else np.array(obs_next)
            self.data.rew = rew if isinstance(rew, np.ndarray) else np.array(rew)
            self.data.done = done
            self.data.terminated = terminated
            self.data.truncated = truncated
            self.data.info = info

            self.state_h1, self.state_h2 = self.policy.update_hidden_states(
                torch.as_tensor(self.data.obs_next, device=self.model_device),
                torch.as_tensor(self.data.act, device=self.model_device),
                self.state_h1, self.state_h2
            )

            for env_idx in range(self.num_of_envs):
                obs_seq[env_idx].append(self.data.obs[env_idx])  # [2420]
                act_seq[env_idx].append(self.data.act[env_idx])  # [3]
                rew_seq[env_idx].append(self.data.rew[env_idx])  # scalar
                done_seq[env_idx].append(self.data.done[env_idx])  # scalar
                terminated_seq[env_idx].append(self.data.terminated[env_idx])
                truncated_seq[env_idx].append(self.data.truncated[env_idx])
                obs_next_seq[env_idx].append(self.data.obs_next[env_idx])  # [2420]
                state = (self.data.state[:, env_idx, :].cpu().numpy() if self.data.state is not None 
                         else np.zeros((2, 128)))  # [2, 128]
                state_seq[env_idx].append(state)
                state_h1_seq[env_idx].append(self.state_h1[:, env_idx].cpu().numpy())  # [2, 128]
                state_h2_seq[env_idx].append(self.state_h2[:, env_idx].cpu().numpy())  # [2, 128]

            step_rews.extend(rew)
            local_step_count += 1
            self.continuous_step_count += 1

            for idx in range(self.num_of_envs):
                episode_lens_dict[idx] += 1
                episode_rews_dict[idx] += rew[idx]
                env_reset = done[idx] or (isinstance(info, dict) and 'reset' in info and info['reset'][idx])

                if env_reset:
                    episode_lens.append(episode_lens_dict[idx])
                    episode_rews.append(episode_rews_dict[idx])
                    episode_count += 1
                    episode_lens_dict[idx] = 0
                    episode_rews_dict[idx] = 0.0
                    single_obs, _ = self.env.reset()
                    self.data.obs_next[idx] = single_obs[0] if isinstance(single_obs, tuple) else single_obs

            if local_step_count >= self.seq_len:
                batch = Batch(
                    obs=np.array(obs_seq),           # [64, 300, 2420]
                    act=np.array(act_seq),           # [64, 300, 3]
                    rew=np.array(rew_seq),           # [64, 300]
                    done=np.array(done_seq),         # [64, 300]
                    terminated=np.array(terminated_seq),
                    truncated=np.array(truncated_seq),
                    obs_next=np.array(obs_next_seq), # [64, 300, 2420]
                    info=self.data.info,
                    state=np.array(state_seq),       # [64, 300, 2, 128]
                    state_h1=np.array(state_h1_seq), # [64, 300, 2, 128]
                    state_h2=np.array(state_h2_seq)  # [64, 300, 2, 128]
                )
                ptr, ep_rew, ep_len, ep_idx = self.buffer.add(batch)
                obs_seq = [[] for _ in range(self.num_of_envs)]
                act_seq = [[] for _ in range(self.num_of_envs)]
                rew_seq = [[] for _ in range(self.num_of_envs)]
                done_seq = [[] for _ in range(self.num_of_envs)]
                terminated_seq = [[] for _ in range(self.num_of_envs)]
                truncated_seq = [[] for _ in range(self.num_of_envs)]
                obs_next_seq = [[] for _ in range(self.num_of_envs)]
                state_seq = [[] for _ in range(self.num_of_envs)]
                state_h1_seq = [[] for _ in range(self.num_of_envs)]
                state_h2_seq = [[] for _ in range(self.num_of_envs)]

            if n_episode and episode_count >= n_episode:
                break
            if n_step and local_step_count >= n_step:
                break

            self.data.obs = self.data.obs_next

        collect_time = time.time() - start_time
        rews = np.array(episode_rews) if episode_count > 0 else np.array([])
        lens = np.array(episode_lens) if episode_count > 0 else np.array([])
        step_rews = np.array(step_rews) if local_step_count > 0 else np.array([])

        if episode_count > 0:
            mean_rew = rews.mean()
            mean_len = lens.mean()
            std_rew = rews.std()
            std_len = lens.std()
            reward_sum = rews.sum()
            length_sum = lens.sum()
        else:
            mean_rew = mean_len = std_rew = std_len = reward_sum = length_sum = 0.0

        step_time = collect_time / local_step_count if local_step_count > 0 else 0.0

        return CollectStats(
            n_ep=episode_count,
            n_st=local_step_count,
            rews=step_rews,
            lens=lens,
            rew=mean_rew,
            len=mean_len,
            rew_std=std_rew,
            len_std=std_len,
            n_collected_steps=local_step_count,
            n_collected_episodes=episode_count,
            returns_stat=None,
            lens_stat=None,
            rewards_stat=None,
            episodes=episode_count,
            reward_sum=reward_sum,
            length_sum=int(length_sum),
            collect_time=collect_time,
            step_time=step_time,
            returns=rews,
            lengths=lens,
            continuous_step=self.continuous_step_count
        )

    def collect(self, n_step=None, n_episode=None, random=False, render=None, no_grad=True, gym_reset_kwargs=None):
        return self._collect(
            random=random,
            render=render,
            n_step=n_step,
            n_episode=n_episode,
            gym_reset_kwargs=gym_reset_kwargs
        )

    def reset(self, reset_buffer=True):
        if reset_buffer:
            self.buffer.reset()
        else:
            self.buffer.reset(keep_statistics=True)
        self.reset_batch_data()
        self.continuous_step_count = 0
        self.state_h1.zero_()
        self.state_h2.zero_()
