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
        
        # Initialize per-environment tracking
        env_episode_counts = np.zeros(self.num_of_envs, dtype=int)
        env_reward_sums = np.zeros(self.num_of_envs)
        env_length_sums = np.zeros(self.num_of_envs, dtype=int)

        # Initialize sequence storage
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
        
        # Initialize info storage - handles both dict and array cases
        if hasattr(self.data, 'info') and isinstance(self.data.info, dict):
            info_dict_seq = {k: [[] for _ in range(self.num_of_envs)] for k in self.data.info.keys()}
            info_is_dict = True
        else:
            info_seq = [[] for _ in range(self.num_of_envs)]
            info_is_dict = False

        while True:
            if self.data.obs is None:
                obs, info = self.reset_env(gym_reset_kwargs)
                self.data.obs = obs
                self.data.info = info
                # Reinitialize info tracking after reset
                if isinstance(info, dict):
                    info_dict_seq = {k: [[] for _ in range(self.num_of_envs)] for k in info.keys()}
                    info_is_dict = True
                else:
                    info_seq = [[] for _ in range(self.num_of_envs)]
                    info_is_dict = False

            # Get actions
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

            # Environment step
            result = self.env.step(self.data.act)
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            else:
                obs_next, rew, done, info = result
                terminated = done
                truncated = np.zeros_like(done)

            # Store data in sequences
            for env_idx in range(self.num_of_envs):
                obs_seq[env_idx].append(self.data.obs[env_idx])
                act_seq[env_idx].append(self.data.act[env_idx])
                rew_seq[env_idx].append(rew[env_idx])
                done_seq[env_idx].append(done[env_idx])
                terminated_seq[env_idx].append(terminated[env_idx])
                truncated_seq[env_idx].append(truncated[env_idx])
                obs_next_seq[env_idx].append(obs_next[env_idx])
                state_seq[env_idx].append(
                    self.data.state[:, env_idx, :].cpu().numpy() 
                    if self.data.state is not None 
                    else np.zeros((2, 128))
                )
                state_h1_seq[env_idx].append(self.state_h1[:, env_idx].cpu().numpy())
                state_h2_seq[env_idx].append(self.state_h2[:, env_idx].cpu().numpy())
                
                # Handle info - supports both dict and array-style
                if info_is_dict:
                    for k, v in info.items():
                        if isinstance(v, (np.ndarray, list)) and len(v) == self.num_of_envs:
                            info_dict_seq[k][env_idx].append(v[env_idx])
                        else:
                            info_dict_seq[k][env_idx].append(v)  # Scalar or global value
                else:
                    if isinstance(info, (np.ndarray, list)) and len(info) == self.num_of_envs:
                        info_seq[env_idx].append(info[env_idx])
                    else:
                        info_seq[env_idx].append(info)  # Scalar value

                # Update per-env episode stats
                if done[env_idx]:
                    env_episode_counts[env_idx] += 1
                    env_reward_sums[env_idx] += rew[env_idx]
                    env_length_sums[env_idx] += 1

            # Update hidden states
            self.state_h1, self.state_h2 = self.policy.update_hidden_states(
                torch.as_tensor(obs_next, device=self.model_device),
                torch.as_tensor(self.data.act, device=self.model_device),
                self.state_h1, self.state_h2
            )

            step_rews.extend(rew)
            local_step_count += 1
            self.continuous_step_count += 1

            # When sequence is complete
            if local_step_count >= self.seq_len:
                # Prepare batch with proper info handling
                batch_info = info_dict_seq if info_is_dict else np.array(info_seq)
                
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
                    info=batch_info
                )

                # Add to all buffers (64) and get per-env stats
                buffer_results = self.buffer.add(batch)
                
                # Calculate statistics
                total_ep_rew = sum(r[1] for r in buffer_results)
                total_ep_len = sum(r[2] for r in buffer_results)
                avg_ep_rew = total_ep_rew / len(buffer_results) if buffer_results else 0
                avg_ep_len = total_ep_len / len(buffer_results) if buffer_results else 0
                
                # Reset sequence buffers
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
                
                if info_is_dict:
                    info_dict_seq = {k: [[] for _ in range(self.num_of_envs)] for k in info_dict_seq.keys()}
                else:
                    info_seq = [[] for _ in range(self.num_of_envs)]

            # Termination conditions
            if n_episode and episode_count >= n_episode:
                break
            if n_step and local_step_count >= n_step:
                break

            self.data.obs = obs_next

        # Final statistics
        collect_time = time.time() - start_time
        
        return CollectStats(
            n_ep=episode_count,
            n_st=local_step_count,
            rews=np.array(step_rews),
            lens=np.array(episode_lens),
            rew=avg_ep_rew,
            len=avg_ep_len,
            rew_std=np.std([r[1] for r in buffer_results]) if buffer_results else 0.0,
            len_std=np.std([r[2] for r in buffer_results]) if buffer_results else 0.0,
            n_collected_steps=local_step_count,
            n_collected_episodes=episode_count,
            returns_stat=None,
            lens_stat=None,
            rewards_stat=None,
            episodes=episode_count,
            reward_sum=total_ep_rew,
            length_sum=int(total_ep_len),
            collect_time=collect_time,
            step_time=collect_time / local_step_count if local_step_count > 0 else 0.0,
            returns=np.array(episode_rews),
            lengths=np.array(episode_lens),
            continuous_step=self.continuous_step_count,
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
