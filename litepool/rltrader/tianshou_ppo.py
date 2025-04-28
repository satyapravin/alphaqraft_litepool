import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path
import litepool
from tianshou.data import Batch
from vec_normalizer import VecNormalize
from metric_logger import MetricLogger  # <-- NEW
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
    action_space=env_action_space,
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
metric_logger = MetricLogger(print_interval=6400)  # 6400 steps (100 episodes if avg 64 envs done)

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
collector = PPOCollector(env, policy, rollout_len=2048)

# === PPO Training Loop ===
def train(epochs=100, rollout_len=2048, batch_size=64, update_epochs=4):
    global_step = 0
    for epoch in range(epochs):
        batch = collector.collect()

        advantages, returns = policy.compute_gae(batch.rew, batch.value, batch.done)
        batch.advantages = advantages
        batch.returns = returns

        dataset_size = len(batch.obs)
        indices = np.arange(dataset_size)

        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                minibatch = Batch(
                    obs=batch.obs[indices[start:end]],
                    act=batch.act[indices[start:end]],
                    log_prob=batch.log_prob[indices[start:end]],
                    value=batch.value[indices[start:end]],
                    advantages=batch.advantages[indices[start:end]],
                    returns=batch.returns[indices[start:end]]
                )
                loss_info = policy.learn(minibatch)

        # === MetricLogger Integration ===
        latest_info = batch.infos[-1] if len(batch.infos) > 0 else {}  # Use last infos
        rew = batch.rew[-num_of_envs:].cpu().numpy() if isinstance(batch.rew, torch.Tensor) else batch.rew[-num_of_envs:]
        metric_logger.log(global_step, latest_info, rew, policy)

        print(f"Epoch {epoch+1} | Loss: {loss_info['loss']:.3f} | Policy Loss: {loss_info['policy_loss']:.3f} | Value Loss: {loss_info['value_loss']:.3f} | Entropy: {loss_info['entropy']:.3f}")

        save_checkpoint(epoch, env_step=global_step)

        global_step += rollout_len

    # After all epochs, save final model for inference
    torch.save(policy.model.state_dict(), results_dir / "final_model_inference.pth")
    print(f"Final model for inference saved at {results_dir / 'final_model_inference.pth'}")

# === Run training ===
print("Starting PPO training...")
train()
print("Training complete!")
