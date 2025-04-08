import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path
import litepool
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from vec_normalizer import VecNormalize
from recurrent_actor import RecurrentActor
from iqn_critic import IQNCritic
from custom_sac_policy import CustomSACPolicy
from replay_buffer import StackedVectorReplayBuffer
from gpu_collector import GPUCollector
import logging
from datetime import datetime

device = torch.device("cuda")
num_of_envs = 64
stack_num = 60

# Environment setup
env = litepool.make(
    "RlTrader-v0", env_type="gymnasium", num_envs=num_of_envs, batch_size=num_of_envs,
    num_threads=num_of_envs, is_prod=False, is_inverse_instr=True, api_key="",
    api_secret="", symbol="BTC-PERPETUAL", tick_size=0.5, min_amount=10,
    maker_fee=0.00001, taker_fee=0.0005, foldername="./train_files/",
    balance=1.0, start=3601 * 20, max=3600 * 10
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


# Checkpoint saving function
def save_checkpoint_fn(epoch, env_step, gradient_step):
    try:
        checkpoint_dir = results_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{env_step}.pth"
        torch.save({
            'epoch': epoch,
            'env_step': env_step,
            'gradient_step': gradient_step,
            'policy_state_dict': policy.state_dict(),
            'actor_optim_state_dict': policy.actor_optim.state_dict(),
            'critic1_optim_state_dict': policy.critic1_optim.state_dict(),
            'critic2_optim_state_dict': policy.critic2_optim.state_dict(),
            'alpha_optim_state_dict': policy.alpha_optim.state_dict(),
            'buffer_config': {
                'total_size': buffer.total_size,
                'buffer_num': buffer.buffer_num,
                'stack_num': buffer.stack_num,
                'env_segment_size': buffer.env_segment_size
            }
        }, checkpoint_path)

        if env_step % (6401 * num_of_envs) == 0:
            buffer_path = checkpoint_dir / f"buffer_epoch_{epoch}_step_{env_step}.h5"
            super(StackedVectorReplayBuffer, buffer).save_hdf5(buffer_path)

            meta_data = {
                'buffer_type': 'StackedVectorReplayBuffer',
                'config': {
                    'total_size': buffer.total_size,
                    'buffer_num': buffer.buffer_num,
                    'stack_num': buffer.stack_num,
                    'env_segment_size': buffer.env_segment_size
                },
                'device': str(device)
            }
            torch.save(meta_data, checkpoint_dir / f"buffer_metadata_{epoch}_{env_step}.pt")

        print(f"Saved checkpoint at epoch {epoch}, step {env_step}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False


# Directory setup
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Model initialization
torch.manual_seed(42)
actor = RecurrentActor(device).to(device)
critic1 = IQNCritic(action_dim=3, num_quantiles=32, hidden_dim=128, quantile_embedding_dim=32, gru_hidden_dim=128, num_layers=2).to(device)
torch.manual_seed(1221)
critic2 = IQNCritic(action_dim=3, num_quantiles=32, hidden_dim=128, quantile_embedding_dim=32, gru_hidden_dim=128, num_layers=2).to(device)

critic1_optim = Adam(critic1.parameters(), lr=3e-4)
critic2_optim = Adam(critic2.parameters(), lr=3e-4)

policy = CustomSACPolicy(
    device=device,
    actor=actor,
    critic1=critic1,
    critic2=critic2,
    actor_optim=Adam(actor.parameters(), lr=3e-4),
    critic1_optim=critic1_optim,
    critic2_optim=critic2_optim,
    tau=0.01, gamma=0.99, init_alpha=5.0,
    action_space=env_action_space
)

policy = policy.to(device)

# Checkpoint loading
final_checkpoint_path = results_dir / "final_model.pth"
final_buffer_path = results_dir / "final_buffer.h5"
metadata_path = results_dir / "final_buffer_metadata.pt"
start_epoch = 0

if final_checkpoint_path.exists():
    print(f"Loading model from {final_checkpoint_path}")
    saved_model = torch.load(final_checkpoint_path)

    policy.load_state_dict(saved_model['policy_state_dict'])
    policy.actor_optim.load_state_dict(saved_model['actor_optim_state_dict'])
    policy.critic1_optim.load_state_dict(saved_model['critic1_optim_state_dict'])
    policy.critic2_optim.load_state_dict(saved_model['critic2_optim_state_dict'])
    policy.alpha_optim.load_state_dict(saved_model['alpha_optim_state_dict'])

    start_epoch = saved_model.get('epoch', 0)
    print(f"Resumed from epoch {start_epoch}")
    print(f"Alpha value: {policy.get_alpha.item():.6f}")
else:
    print(f"Could not find model checkpoint at {final_checkpoint_path}")

if final_buffer_path.exists():
    print(f"Loading buffer from {final_buffer_path}")

    if metadata_path.exists():
        metadata = torch.load(metadata_path)
        buffer_config = metadata['config']
        print(f"Loaded buffer config: {buffer_config}")
    else:
        buffer_config = {
            'total_size': num_of_envs * 100,
            'buffer_num': num_of_envs,
            'stack_num': 60,
            'env_segment_size': 100
        }

    buffer = StackedVectorReplayBuffer(
        total_size=buffer_config['total_size'],
        buffer_num=buffer_config['buffer_num'],
        stack_num=buffer_config['stack_num'],
        device=device
    )

    temp_buffer = VectorReplayBuffer.load_hdf5(final_buffer_path)
    buffer._meta = temp_buffer._meta
    buffer._index = temp_buffer._index
    buffer._size = temp_buffer._size

    if hasattr(buffer, 'env_segment_size'):
        expected_segment = buffer_config['total_size'] // buffer_config['buffer_num']
        if buffer.env_segment_size != expected_segment:
            print(f"Warning: Adjusting env_segment_size from {buffer.env_segment_size} to {expected_segment}")
            buffer.env_segment_size = expected_segment

    print(f"Buffer loaded with {len(buffer)} transitions")
else:
    print(f"No buffer found at {final_buffer_path}, creating new buffer")
    buffer = StackedVectorReplayBuffer(
        total_size=num_of_envs * 100,  # 6400 transitions total
        buffer_num=num_of_envs,  # 64 envs
        stack_num=60,  # 60 steps stacked
        device=device
    )

# Collector setup
collector = GPUCollector(
    num_of_envs=num_of_envs,
    policy=policy,
    env=env,
    buffer=buffer,
    device=device
)

trainer = OffpolicyTrainer(
    policy=policy,
    train_collector=collector,
    test_collector=None,  # No test env for now; add if needed
    max_epoch=50,  # Number of epochs to train
    step_per_epoch=10,  # Steps per epoch (adjust based on your needs)
    step_per_collect=64*10,  # Collect 64 steps (1 step per env) per iteration
    episode_per_test=0,  # No test episodes; set if you add a test env
    batch_size=64,  # Match num_of_envs for efficient sampling
    update_per_step=0.1,  # Update policy once per collected step
    train_fn=lambda epoch, env_step: None,  # Optional training hooks
    test_fn=None,  # No testing for now
    stop_fn=lambda mean_rewards: mean_rewards >= 1000,  # Stop if mean reward exceeds threshold
    save_checkpoint_fn=save_checkpoint_fn,
    resume_from_log=start_epoch > 0,
    reward_metric=lambda rews: np.mean(rews)  # Average reward across envs
)

# Run the trainer
print("Starting training...")
result = trainer.run()
print(f"Training completed: {result}")

# Save final model and buffer
torch.save({
    'policy_state_dict': policy.state_dict(),
    'actor_optim_state_dict': policy.actor_optim.state_dict(),
    'critic1_optim_state_dict': policy.critic1_optim.state_dict(),
    'critic2_optim_state_dict': policy.critic2_optim.state_dict(),
    'alpha_optim_state_dict': policy.alpha_optim.state_dict(),
    'epoch': result['epoch'],
    'env_step': result['env_step'],
    'gradient_step': result['gradient_step']
}, final_checkpoint_path)

buffer.save_hdf5(final_buffer_path)
torch.save({
    'buffer_type': 'StackedVectorReplayBuffer',
    'total_size': num_of_envs * 100,
    'buffer_num': num_of_envs,
    'stack_num': 60,
    'device': str(device)
}, metadata_path)

print(f"Final model saved to {final_checkpoint_path}")
print(f"Final buffer saved to {final_buffer_path}")
