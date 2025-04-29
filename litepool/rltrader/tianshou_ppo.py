import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path
import litepool
from tianshou.data import Batch  # Not used now but kept
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
    lr=3e-4,
    gamma=0.999,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.001,
    max_grad_norm=0.5
)

# === Directory setup ===
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
checkpoint_dir = results_dir / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True, parents=True)

# === Metric Logger ===
metric_logger = MetricLogger(print_interval=512)

# === Checkpoint saving ===
def save_checkpoint(epoch, env_step):
    torch.save({
        'epoch': epoch,
        'env_step': env_step,
        'model_state_dict': policy.model.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
    }, checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{env_step}.pth")
    env.save(results_dir / "vecnorm.pth")
    print(f"Saved checkpoint at epoch {epoch}, step {env_step}")

# === Checkpoint loading ===
def load_latest_checkpoint():
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    print(f"Loading checkpoint: {latest.name}")
    checkpoint = torch.load(latest, map_location=device)
    policy.model.load_state_dict(checkpoint['model_state_dict'])
    policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    env.load(results_dir / "vecnorm.pth")
    return checkpoint['epoch'], checkpoint['env_step']

# === PPO Collector ===
collector = PPOCollector(env, policy, n_steps=1024)

# === PPO Training Loop ===
def train(epochs=500, rollout_len=1024, minibatch_seq_len=256, minibatch_envs=64, update_epochs=16):
    # === Try to resume from checkpoint ===
    resume_info = load_latest_checkpoint()
    if resume_info:
        start_epoch, global_step = resume_info
        start_epoch += 1  # resume from next epoch
    else:
        start_epoch = 0
        global_step = 0

    for epoch in range(start_epoch, epochs):
        batch = collector.collect()

        # === Compute advantages ===
        advantages, returns = batch["advantages"], batch["returns"]

        # Normalize advantages
        advantages = advantages.to(device)
        adv_flat = advantages.view(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        batch["advantages"] = advantages

        # === Train policy ===
        for _ in range(update_epochs):
            for start_t in range(0, rollout_len, minibatch_seq_len):
                end_t = start_t + minibatch_seq_len
                if end_t > rollout_len:
                    break

                for start_e in range(0, num_of_envs, minibatch_envs):
                    end_e = start_e + minibatch_envs
                    if end_e > num_of_envs:
                        break

                    minibatch = {
                        'obs': batch['obs'][start_t:end_t, start_e:end_e].to(device),
                        'act': batch['actions'][start_t:end_t, start_e:end_e].to(device),
                        'logp': batch['log_probs'][start_t:end_t, start_e:end_e].to(device),
                        'val': batch['values'][start_t:end_t, start_e:end_e].to(device),
                        'adv': batch['advantages'][start_t:end_t, start_e:end_e].to(device),
                        'ret': batch['returns'][start_t:end_t, start_e:end_e].to(device),
                        'state': tuple(s[start_t].transpose(0, 1)[start_e:end_e].transpose(0, 1).contiguous().to(device) for s in batch['states']),
                    }

                    loss_info = policy.learn(minibatch)

        # === Update global step BEFORE logging ===
        global_step += rollout_len

        # === MetricLogger Integration ===
        rew = batch["rewards"][-num_of_envs:]
        metric_logger.log(global_step, {
            "infos": batch["infos"][-num_of_envs:],
        }, rew, policy)

        print(f"Epoch {epoch+1} | Loss: {loss_info['loss']:.3f} | "
              f"Policy Loss: {loss_info['actor_loss']:.3f} | "
              f"Value Loss: {loss_info['value_loss']:.3f} | "
              f"Entropy: {loss_info['entropy_loss']:.3f} | "
              f"KL: {loss_info['kl_loss']:.6f}")

        # Save checkpoint after each epoch
        save_checkpoint(epoch, env_step=global_step)

        # Reset episode reward tracking if needed
        if hasattr(collector, 'reset_episode_rewards'):
            collector.reset_episode_rewards()

    # Save final model
    torch.save(policy.model.state_dict(), results_dir / "final_model_inference.pth")
    env.save(results_dir / "vecnorm.pth")
    print(f"Final model for inference saved at {results_dir / 'final_model_inference.pth'}")

# === Run training ===
print("Starting PPO training...")
train()
print("Training complete!")
