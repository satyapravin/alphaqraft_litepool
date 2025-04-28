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
metric_logger = MetricLogger(print_interval=512)

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
collector = PPOCollector(env, policy, n_steps=2048)

# === PPO Training Loop ===
def train(epochs=100, rollout_len=2048, minibatch_seq_len=128, minibatch_envs=8, update_epochs=4):
    global_step = 0
    for epoch in range(epochs):
        batch = collector.collect()
        
        # === Compute advantages ===
        advantages, returns = batch["advantages"], batch["returns"]

        # Normalize advantages
        advantages = torch.tensor(advantages, device=device)
        adv_flat = advantages.view(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Save back normalized advantages to batch
        batch["advantages"] = advantages.cpu().detach().numpy()

        rollout_len, num_envs = batch["obs"].shape[:2]

        # === Train policy ===
        for _ in range(update_epochs):
            for start_t in range(0, rollout_len, minibatch_seq_len):
                end_t = start_t + minibatch_seq_len
                if end_t > rollout_len:
                    break

                for start_e in range(0, num_envs, minibatch_envs):
                    end_e = start_e + minibatch_envs
                    if end_e > num_envs:
                        break

                    minibatch = {
                        'obs': torch.tensor(batch['obs'][start_t:end_t, start_e:end_e]).to(device),
                        'act': torch.tensor(batch['actions'][start_t:end_t, start_e:end_e]).to(device),
                        'logp': torch.tensor(batch['log_probs'][start_t:end_t, start_e:end_e]).to(device),
                        'val': torch.tensor(batch['values'][start_t:end_t, start_e:end_e]).to(device),
                        'adv': torch.tensor(batch['advantages'][start_t:end_t, start_e:end_e]).to(device),
                        'ret': torch.tensor(batch['returns'][start_t:end_t, start_e:end_e]).to(device),
                    }

                    loss_info = policy.learn(minibatch)

        # === Update global step BEFORE logging ===
        global_step += rollout_len

        # === MetricLogger Integration ===
        flat_infos = [info for infos_per_step in batch["infos"] for info in infos_per_step]
        avg_realized_pnl = np.mean([info.get('realized_pnl', 0.0) for info in flat_infos])
        total_fees = np.sum([info.get('fees', 0.0) for info in flat_infos])
        avg_trade_count = np.mean([info.get('trade_count', 0) for info in flat_infos])

        rew = batch["rewards"][-num_of_envs:]

        metric_logger.log(global_step, {
            "realized_pnl": avg_realized_pnl,
            "fees": total_fees,
            "trade_count": avg_trade_count
        }, rew, policy)

        print(f"Epoch {epoch+1} | Loss: {loss_info['loss']:.3f} | "
              f"Policy Loss: {loss_info['actor_loss']:.3f} | "
              f"Value Loss: {loss_info['value_loss']:.3f} | "
              f"Entropy: {loss_info['entropy_loss']:.3f} | "
              f"Avg Realized PnL: {avg_realized_pnl:.4f}")

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
