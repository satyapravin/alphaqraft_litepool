import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path
import litepool
from tianshou.data import Batch
from vec_normalizer import VecNormalize
from metric_logger import MetricLogger
from datetime import datetime

from recurrent_actor_critic import RecurrentActorCritic
from recurrent_ppo_policy import RecurrentPPOPolicy
from ppo_collector import PPOCollector

device = torch.device("cuda")
num_of_envs = 64

# === Environment setup ===
env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=num_of_envs, batch_size=num_of_envs,
    num_threads=num_of_envs, is_prod=False, is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL", hedge_symbol='BTC-18APR25', tick_size=0.5, min_amount=10,
    maker_fee=-0.0001, taker_fee=0.0005, foldername="./train_files/",
    balance=1.0, start=3601 * 10, max=14400 * 10
)
env.spec.id = 'RlTrader-v0'
env_action_space = env.action_space

env = VecNormalize(
    env,
    device=device,
    num_envs=num_of_envs,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.,
    clip_reward=10.,
    gamma=0.99
)

# === Model initialization ===
torch.manual_seed(42)

model = RecurrentActorCritic(
    action_dim=env_action_space.shape[0],
    hidden_dim=128,
    gru_hidden_dim=128,
    num_layers=2
).to(device)

policy = RecurrentPPOPolicy(
    model=model,
    #action_space=env_action_space,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5
)

# === Directory setup ===
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# === Metric Logger ===
metric_logger = MetricLogger(print_interval=6400)

# === Checkpoint saving ===
def save_checkpoint(epoch, env_step):
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    torch.save({
        'epoch': epoch,
        'env_step': env_step,
        'model_state_dict': policy.model.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
    }, checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{env_step}.pth")
    env.save(results_dir / "vecnorm.pth")
    print(f"Saved checkpoint at epoch {epoch}, step {env_step}")

# === PPO Collector ===
collector = PPOCollector(env, policy, rollout_len=2048, device=device)

# === GAE computation ===
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    rollout_len, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(num_envs, device=rewards.device)

    for t in reversed(range(rollout_len)):
        if t == rollout_len - 1:
            next_value = torch.zeros(num_envs, device=rewards.device)
            next_done = torch.ones(num_envs, device=rewards.device)
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]

        delta = rewards[t] + gamma * next_value * (1.0 - next_done) - values[t]
        advantages[t] = last_advantage = delta + gamma * gae_lambda * (1.0 - next_done) * last_advantage

    returns = advantages + values
    return advantages, returns

# === PPO Training Loop ===
def train(epochs=100, rollout_len=2048, minibatch_seq_len=128, minibatch_envs=8, update_epochs=4):
    global_step = 0
    for epoch in range(epochs):
        batch = collector.collect()

        # Compute advantages
        advantages, returns = compute_gae(
            rewards=batch['rew'],
            values=batch['val'],
            dones=batch['done'],
            gamma=policy.gamma,
            gae_lambda=policy.gae_lambda
        )
        batch['adv'] = advantages
        batch['ret'] = returns

        # Normalize advantages
        adv_flat = batch['adv'].view(-1)
        batch['adv'] = (batch['adv'] - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        rollout_len, num_envs = batch['obs'].shape[:2]


        # Train policy
        for _ in range(update_epochs):
            for start_t in range(0, rollout_len, minibatch_seq_len):
                end_t = start_t + minibatch_seq_len
                if end_t > rollout_len:
                    break

                for start_e in range(0, num_envs, minibatch_envs):
                    end_e = start_e + minibatch_envs
                    if end_e > num_envs:
                        break

                    minibatch = Batch(
                        obs=batch['obs'][start_t:end_t, start_e:end_e],
                        act=batch['act'][start_t:end_t, start_e:end_e],
                        log_prob=batch['logp'][start_t:end_t, start_e:end_e],
                        value=batch['val'][start_t:end_t, start_e:end_e],
                        rew=batch['rew'][start_t:end_t, start_e:end_e],
                        done=batch['done'][start_t:end_t, start_e:end_e],
                        adv=batch['adv'][start_t:end_t, start_e:end_e],
                        ret=batch['ret'][start_t:end_t, start_e:end_e],
                    )

                    minibatch.to_torch(device=device)
                    loss_info = policy.learn(minibatch)

        # === MetricLogger Integration ===
        all_ep_rewards = [r for env_rewards in batch.ep_rewards for r in env_rewards]
        avg_reward = np.mean(all_ep_rewards) if len(all_ep_rewards) > 0 else 0.0

        latest_info = batch.infos[-1] if len(batch.infos) > 0 else {}
        rew = batch.rew[-num_of_envs:].cpu().numpy() if isinstance(batch.rew, torch.Tensor) else batch.rew[-num_of_envs:]

        metric_logger.log(global_step, latest_info, rew, policy)

        print(f"Epoch {epoch+1} | Loss: {loss_info['loss']:.3f} | Policy Loss: {loss_info['actor_loss']:.3f} | Value Loss: {loss_info['value_loss']:.3f} | Entropy: {loss_info['entropy_loss']:.3f} | Avg Reward: {avg_reward:.2f}")

        save_checkpoint(epoch, env_step=global_step)

        global_step += rollout_len

        # Reset episode reward tracking
        collector.reset_episode_rewards()

    # After all epochs, save final weights
    torch.save(policy.model.state_dict(), results_dir / "final_model_inference.pth")
    env.save(results_dir / "vecnorm.pth")
    print(f"Final model for inference saved at {results_dir / 'final_model_inference.pth'}")

# === Run training ===
print("Starting PPO training...")
train()
print("Training complete!")
