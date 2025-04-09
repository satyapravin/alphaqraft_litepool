import time
import numpy as np
import torch
from tianshou.data import Collector, Batch
from tianshou.utils.statistics import RunningMeanStd  # Correct import
from dataclasses import dataclass
from metric_logger import MetricLogger

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
    returns_stat: RunningMeanStd  # Updated to RunningMeanStd
    lens_stat: RunningMeanStd     # Updated to RunningMeanStd
    rewards_stat: RunningMeanStd  # Updated to RunningMeanStd
    episodes: int 
    reward_sum: float
    length_sum: int
    collect_time: float
    step_time: float 
    returns: np.ndarray
    lengths: np.ndarray
    continuous_step: int
    info: object

class CPUCollector(Collector):
    def __init__(self, policy, env, num_of_envs, print_interval=1000, buffer=None, device='cpu', seq_len=300, **kwargs):
        super().__init__(policy, env, buffer, **kwargs)
        self.device = device
        self.model_device = 'cuda'
        self.num_of_envs = num_of_envs
        self.print_interval = print_interval
        self.seq_len = seq_len
        self.env_active = False
        self.continuous_step_count = 0
        self.data = Batch()
        self.reset_batch_data()
        # Initialize running statistics
        self.returns_stat = RunningMeanStd()
        self.lens_stat = RunningMeanStd()
        self.rewards_stat = RunningMeanStd()

        self.state_h1 = torch.zeros(
            policy.critic1.num_layers, env.num_envs, policy.critic1.gru_hidden_dim,
            device=self.model_device
        )
        self.state_h2 = torch.zeros(
            policy.critic2.num_layers, env.num_envs, policy.critic2.gru_hidden_dim,
            device=self.model_device
        )
        self.data.state = torch.zeros(
            self.num_of_envs, policy.critic1.num_layers, policy.critic1.gru_hidden_dim,
            device=self.model_device
        )

        self.metric_logger = MetricLogger(print_interval=print_interval)

    def reset_batch_data(self):
        self.data.obs = None
        self.data.info = None
        self.data.act = None
        self.data.state = torch.zeros(
            self.num_of_envs, self.policy.critic1.num_layers, self.policy.critic1.gru_hidden_dim,
            device=self.model_device
        )
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
            self.data.state.zero_()
        return self.data.obs, self.data.info

    def _collect(self, random=None, render=None, n_step=None, n_episode=None, gym_reset_kwargs=None):
        start_time = time.time()
        local_step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        step_rews = []
        
        env_episode_counts = np.zeros(self.num_of_envs, dtype=int)
        env_reward_sums = np.zeros(self.num_of_envs)
        env_length_sums = np.zeros(self.num_of_envs, dtype=int)

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
        
        info_dict_seq = {k: [[] for _ in range(self.num_of_envs)] for k in self.data.info.keys()} if self.data.info else {}
        info_is_dict = True

        if self.data.obs is None or not self.env_active:
            obs, info = self.reset_env(gym_reset_kwargs)
            self.data.obs = obs
            self.data.info = info
            info_dict_seq = {k: [[] for _ in range(self.num_of_envs)] for k in info.keys()}
            print(f"Initial reset, trade_count: {info['trade_count']}")

        while True:
            with torch.no_grad():
                if random:
                    act = np.array([self.env.action_space.sample() for _ in range(self.num_of_envs)])
                    result = Batch(act=torch.as_tensor(act, device=self.model_device))
                else:
                    result = self.policy(
                        Batch(obs=torch.as_tensor(self.data.obs, device=self.model_device)),
                        state=self.data.state.transpose(0, 1).contiguous() if self.data.state is not None else None
                    )
            
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


            for env_idx in range(self.num_of_envs):
                obs_seq[env_idx].append(self.data.obs[env_idx])
                act_seq[env_idx].append(self.data.act[env_idx])
                rew_seq[env_idx].append(rew[env_idx])
                done_seq[env_idx].append(done[env_idx])
                terminated_seq[env_idx].append(terminated[env_idx])
                truncated_seq[env_idx].append(truncated[env_idx])
                obs_next_seq[env_idx].append(obs_next[env_idx])
                state_seq[env_idx].append(
                    self.data.state[:, env_idx, :].cpu().numpy() if self.data.state is not None else np.zeros((2, 128))
                )
                state_h1_seq[env_idx].append(self.state_h1[:, env_idx].cpu().numpy())
                state_h2_seq[env_idx].append(self.state_h2[:, env_idx].cpu().numpy())
                
                for k, v in info.items():
                    if isinstance(v, (np.ndarray, list)) and len(v) == self.num_of_envs:
                        info_dict_seq[k][env_idx].append(v[env_idx])
                    else:
                        info_dict_seq[k][env_idx].append(v)

                if done[env_idx]:
                    env_episode_counts[env_idx] += 1
                    env_reward_sums[env_idx] += rew[env_idx]
                    env_length_sums[env_idx] += 1
                    episode_count += 1
                    episode_rews.append(env_reward_sums[env_idx])
                    episode_lens.append(env_length_sums[env_idx])
                    self.returns_stat.update(np.array([env_reward_sums[env_idx]]))
                    self.lens_stat.update(np.array([env_length_sums[env_idx]]))
                    env_reward_sums[env_idx] = 0
                    env_length_sums[env_idx] = 0
                    if hasattr(self.env, 'reset_one'):
                        obs_next[env_idx], info_reset = self.env.reset_one(env_idx)
                        for k, v in info_reset.items():
                            info_dict_seq[k][env_idx][-1] = v
                    print(f"Env {env_idx} done at step {local_step_count}, trade_count: {info['trade_count'][env_idx]}")

            self.rewards_stat.update(np.array([np.mean(rew)]))
            step_rews.extend(rew)
            local_step_count += 1
            self.continuous_step_count += 1

            if self.buffer and local_step_count % self.seq_len == 0:
                batch = Batch(
                    obs=np.array(obs_seq),
                    act=np.array(act_seq),
                    rew=np.array(rew_seq),
                    done=np.array(done_seq),
                    terminated=np.array(terminated_seq),
                    truncated=np.array(truncated_seq),
                    obs_next=np.array(obs_next_seq),
                    state=np.array(state_seq),
                    state_h1=np.array(state_h1_seq),
                    state_h2=np.array(state_h2_seq),
                    info=info_dict_seq
                )
                buffer_results = self.buffer.add(batch)
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
                info_dict_seq = {k: [[] for _ in range(self.num_of_envs)] for k in info_dict_seq.keys()}
                print(f"Buffer added at step {local_step_count}")

            # Update state before breaking
            self.data.obs = obs_next
            self.state_h1, self.state_h2 = self.policy.update_hidden_states(
                torch.as_tensor(obs_next, device=self.model_device),
                torch.as_tensor(self.data.act, device=self.model_device),
                self.state_h1, self.state_h2
            )

            # Exit conditions
            if n_step and local_step_count >= n_step:
                print(f"Breaking at n_step={n_step}, local_step_count={local_step_count}")
                break
            if n_episode and episode_count >= n_episode:
                print(f"Breaking at n_episode={n_episode}, episode_count={episode_count}")
                break

            self.metric_logger.log(
                step_count=self.continuous_step_count,
                info=info,
                rew=rew,
                policy=self.policy
            )
        collect_time = time.time() - start_time
        latest_info = info_dict_seq


        return CollectStats(
            n_ep=episode_count,
            n_st=local_step_count,
            rews=np.array(step_rews),
            lens=np.array(episode_lens),
            rew=np.mean(step_rews) if step_rews else 0.0,
            len=np.mean(episode_lens) if episode_lens else 0.0,
            rew_std=np.std(step_rews) if step_rews else 0.0,
            len_std=np.std(episode_lens) if episode_lens else 0.0,
            n_collected_steps=local_step_count,
            n_collected_episodes=episode_count,
            returns_stat=self.returns_stat,
            lens_stat=self.lens_stat,
            rewards_stat=self.rewards_stat,
            episodes=episode_count,
            reward_sum=sum(step_rews),
            length_sum=sum(episode_lens),
            collect_time=collect_time,
            step_time=collect_time / local_step_count if local_step_count > 0 else 0.0,
            returns=np.array(episode_rews) if episode_rews else np.array([]),
            lengths=np.array(episode_lens) if episode_lens else np.array([]),
            continuous_step=self.continuous_step_count,
            info=latest_info
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
        self.returns_stat = RunningMeanStd()
        self.lens_stat = RunningMeanStd()
        self.rewards_stat = RunningMeanStd()
