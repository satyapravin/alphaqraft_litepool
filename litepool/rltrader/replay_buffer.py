import numpy as np
import torch
from tianshou.data import VectorReplayBuffer


class StackedVectorReplayBuffer(VectorReplayBuffer):
    def __init__(self, total_size, buffer_num, stack_num=60, device='cuda', **kwargs):
        super().__init__(total_size=total_size, buffer_num=buffer_num, stack_num=stack_num, **kwargs)
        self.stack_num = stack_num
        self.device = device
        self.env_segment_size = total_size // buffer_num  # Size per environment
        print(f"Buffer initialized with total_size={total_size}, buffer_num={buffer_num}, stack_num={stack_num}")

    def sample(self, batch_size):
        batch_size = int(batch_size)
        print(f"sample: batch_size type: {type(batch_size)}, value: {batch_size}")
        
        # Sample indices from each environment
        indices_per_env = batch_size // self.buffer_num
        if indices_per_env < 1:
            indices_per_env = 1
        
        all_indices = []
        for env_idx in range(self.buffer_num):
            # Get valid start indices for this environment (avoid going before start of segment)
            start = env_idx * self.env_segment_size
            end = start + self.env_segment_size - self.stack_num
            
            # Sample indices within this environment's segment
            env_indices = np.random.randint(start, end, size=indices_per_env)
            all_indices.extend(env_indices)
        
        # Convert to numpy array and ensure we have exactly batch_size indices
        indices = np.array(all_indices[:batch_size])
        print(f"sample: indices shape: {indices.shape}")
        
        # Get the stacked indices for each sampled index
        offsets = np.arange(self.stack_num)
        stacked_indices = indices[:, None] - offsets  # Shape: [batch_size, stack_num]
        
        # Get the batch data
        batch = super().__getitem__(stacked_indices.reshape(-1))
        
        # Reshape the data to maintain the stack structure
        for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
            if hasattr(batch, key):
                val = getattr(batch, key)
                if isinstance(val, np.ndarray):
                    setattr(batch, key, val.reshape(batch_size, self.stack_num, *val.shape[1:]))
                elif isinstance(val, torch.Tensor):
                    setattr(batch, key, val.view(batch_size, self.stack_num, *val.shape[1:]))
        
        print(f"sample: batch.rew shape: {batch.rew.shape}")
        return batch, indices

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            # Handle single index
            index = np.array([index])
        
        if isinstance(index, (np.ndarray, torch.Tensor)):
            if isinstance(index, torch.Tensor):
                index = index.cpu().numpy()
            
            # Get the data for all indices
            batch = super().__getitem__(index)
            
            # If the indices were stacked (2D array), reshape the data
            if index.ndim == 2:
                batch_size, stack_num = index.shape
                for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
                    if hasattr(batch, key):
                        val = getattr(batch, key)
                        if isinstance(val, np.ndarray):
                            setattr(batch, key, val.reshape(batch_size, stack_num, *val.shape[1:]))
                        elif isinstance(val, torch.Tensor):
                            setattr(batch, key, val.view(batch_size, stack_num, *val.shape[1:]))
            
            return batch
        
        return super().__getitem__(index)
