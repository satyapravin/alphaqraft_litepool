from dataclasses import dataclass

import numpy as np
import time
import torch
from tianshou.data import Collector, Batch


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


class StatClass:
    def __init__(self, mean_val, std_val):
        self.mean = mean_val
        self._std = std_val

    def std(self):
        return self._std


class GPUCollector(Collector):
    def __init__(self, policy, env, num_of_envs, buffer=None, device='cuda', **kwargs):
        super().__init__(policy, env, buffer, **kwargs)
        self.device = device
        self.env_active = False
        self.num_of_envs = num_of_envs
        self.continuous_step_count = 0
        self.data = Batch()
        self.reset_batch_data()

        self.state_h1 = torch.zeros(
            policy.critic1.num_layers, env.num_envs, policy.critic1.gru_hidden_dim, device=device
        )
        self.state_h2 = torch.zeros(
            policy.critic2.num_layers, env.num_envs, policy.critic2.gru_hidden_dim, device=device
        )

    def reset_batch_data(self):
        if not self.env_active:
            self.data.obs_next = None
            self.data.obs = None
        self.data.act = None
        self.data.rew = None
        self.data.done = None
        self.data.terminated = None
        self.data.truncated = None
        self.data.info = None
        self.data.policy = Batch()
        self.data.state = None

    def reset_env(self, gym_reset_kwargs=None):
        if not self.env_active:
            gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
            obs = self.env.reset(**gym_reset_kwargs)
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = None

            if isinstance(obs, torch.Tensor):
                obs = obs.to(self.device)
            else:
                obs = torch.as_tensor(obs, device=self.device)

            self.data.obs = obs
            self.data.info = info
            self.env_active = True

            self.state_h1.zero_()
            self.state_h2.zero_()

        return self.data.obs, self.data.info

    def _reset_hidden_states(self, indices):
        self.state_h1[:, indices, :] = 0
        self.state_h2[:, indices, :] = 0

    def _collect(self, random=None, render=None, n_step=None, n_episode=None, gym_reset_kwargs=None):
        if n_step is not None:
            assert n_episode is None, "Only one of n_step or n_episode is allowed"
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."

        start_time = time.time()
        local_step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_lens_dict = {i: 0 for i in range(self.env.num_envs)}
        episode_rews_dict = {i: 0.0 for i in range(self.env.num_envs)}

        while True:
            if self.data.obs is None:
                obs, info = self.reset_env()
                self.data.obs = obs
                self.data.info = info

            if isinstance(self.data.obs, torch.Tensor):
                obs_batch = Batch(obs=self.data.obs.to(self.device))
            else:
                obs_batch = Batch(obs=torch.as_tensor(self.data.obs, device=self.device))

            # Ensure state is in [B, L, H] format
            if self.data.state is not None and self.data.state.dim() == 3 and self.data.state.shape[1] == self.num_of_envs:
                self.data.state = self.data.state.transpose(0, 1).contiguous()

            with torch.no_grad():
                result = self.policy(obs_batch, state=self.data.state)

            self.data.act = result.act
            self.data.state = result.state if hasattr(result, 'state') else None
            if self.data.state is not None:
                self.data.state = self.data.state.transpose(0, 1).contiguous()

            if isinstance(self.data.act, torch.Tensor):
                action = self.data.act.cpu().numpy()
            else:
                action = np.array(self.data.act)

            result = self.env.step(action)
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            else:
                obs_next, rew, done, info = result
                terminated = done
                truncated = False

            if hasattr(self, 'logger'):
                self.logger.log(self.continuous_step_count, info, rew, self.policy)

            if isinstance(obs_next, torch.Tensor):
                obs_next = obs_next.to(self.device)
            else:
                obs_next = torch.as_tensor(obs_next, device=self.device)

            if isinstance(rew, torch.Tensor):
                rew = rew.to(self.device)
            else:
                rew = torch.as_tensor(rew, device=self.device)

            self.state_h1, self.state_h2 = self.policy.update_hidden_states(
                obs_next, self.data.act, self.state_h1, self.state_h2
            )

            self.data.obs_next = obs_next
            self.data.rew = rew
            self.data.done = done
            self.data.terminated = terminated
            self.data.truncated = truncated
            self.data.info = info

            if isinstance(self.data.obs, torch.Tensor):
                obs = self.data.obs.cpu().numpy()
            else:
                obs = self.data.obs

            if isinstance(self.data.act, torch.Tensor):
                act = self.data.act.cpu().numpy()
            else:
                act = self.data.act

            if isinstance(self.data.rew, torch.Tensor):
                rew = self.data.rew.cpu().numpy()
            else:
                rew = self.data.rew

            if isinstance(self.data.obs_next, torch.Tensor):
                obs_next = self.data.obs_next.cpu().numpy()
            else:
                obs_next = self.data.obs_next

            batch = Batch(
                obs=obs,
                act=act,
                rew=rew,
                done=self.data.done,
                terminated=self.data.terminated,
                truncated=self.data.truncated,
                obs_next=obs_next,
                info=self.data.info,
                policy=self.data.policy if hasattr(self.data, 'policy') else None,
                state=self.data.state if hasattr(self.data, 'state') else None
            )

            batch.state_h1 = self.state_h1.clone().detach().cpu()
            batch.state_h2 = self.state_h2.clone().detach().cpu()
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(batch)

            local_step_count += 1
            self.continuous_step_count += 1

            for idx in range(self.num_of_envs):
                episode_lens_dict[idx] += 1
                episode_rews_dict[idx] += rew[idx]

                env_reset = done[idx] or (isinstance(info, dict) and
                                          'reset' in info and
                                          info['reset'][idx])

            if env_reset:
                episode_lens.append(episode_lens_dict[idx])
                episode_rews.append(episode_rews_dict[idx])
                episode_count += 1

                episode_lens_dict[idx] = 0
                episode_rews_dict[idx] = 0.0

                single_obs, _ = self.env.reset()
                if isinstance(self.data.obs_next, torch.Tensor):
                    self.data.obs_next[idx] = torch.as_tensor(single_obs[0], device=self.device)
                else:
                    self.data.obs_next[idx] = single_obs[0]

            if n_episode and episode_count >= n_episode:
                break
            if n_step and local_step_count >= n_step:
                break

            self.data.obs = self.data.obs_next

        if episode_count > 0:
            rews = np.array(episode_rews)
            lens = np.array(episode_lens)
            mean_rew = rews.mean()
            mean_len = lens.mean()
            std_rew = rews.std()
            std_len = lens.std()
        else:
            empty_arr = np.array([])
            rews = lens = empty_arr
            mean_rew = mean_len = std_rew = std_len = 0.0

        return_stat = StatClass(mean_rew, std_rew)
        return_stat.n_ep = episode_count
        return_stat.n_st = local_step_count
        return_stat.rews = rews
        return_stat.lens = lens
        return_stat.rew = mean_rew
        return_stat.len = mean_len
        return_stat.rew_std = std_rew
        return_stat.len_std = std_len

        lens_stat = StatClass(mean_len, std_len)
        lens_stat.n_ep = episode_count
        lens_stat.n_st = local_step_count
        lens_stat.lens = lens
        lens_stat.len = mean_len
        lens_stat.len_std = std_len

        rewards_stat = StatClass(mean_rew, std_rew)
        rewards_stat.n_ep = episode_count
        rewards_stat.n_st = local_step_count
        rewards_stat.rews = rews
        rewards_stat.rew = mean_rew
        rewards_stat.rew_std = std_rew

        collect_time = time.time() - start_time
        step_time = collect_time / local_step_count if local_step_count else 0

        return CollectStats(
            n_ep=episode_count,
            n_st=local_step_count,
            rews=rews,
            lens=lens,
            rew=mean_rew,
            len=mean_len,
            rew_std=std_rew,
            len_std=std_len,
            n_collected_steps=local_step_count,
            n_collected_episodes=episode_count,
            returns_stat=return_stat,
            lens_stat=lens_stat,
            rewards_stat=rewards_stat,
            episodes=episode_count,
            reward_sum=float(np.sum(rews)) if len(rews) > 0 else 0.0,
            length_sum=int(np.sum(lens)) if len(lens) > 0 else 0,
            collect_time=collect_time,
            step_time=step_time,
            returns=rews,
            lengths=lens,
            continuous_step=self.continuous_step_count
        )
