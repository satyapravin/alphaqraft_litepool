import numpy as np
import torch
from tianshou.data import VectorReplayBuffer

class SequentialReplayBuffer(VectorReplayBuffer):
    def __init__(self, total_size, seq_len=600, buffer_num=1, device='cuda'):
        """
        Args:
            total_size: Total buffer size (number of steps)
            seq_len: Length of sequential chunks to sample
            buffer_num: Number of parallel environments
            device: Target device for tensors
        """
        super().__init__(total_size, buffer_num)
        self.seq_len = seq_len
        self.device = device
        self.env_segment_size = total_size // buffer_num
        
    def sample(self, batch_size):
        """Sample batch_size sequences of seq_len consecutive steps"""
        # Calculate valid start indices for each environment
        valid_starts = self.env_segment_size - self.seq_len
        if valid_starts <= 0:
            raise ValueError(
                f"Not enough samples in buffer (needs {self.seq_len} steps per env, "
                f"but only {self.env_segment_size} available per env)"
            )
        
        # Sample one sequence per environment
        indices = []
        for env_idx in range(self.buffer_num):
            start = env_idx * self.env_segment_size + np.random.randint(0, valid_starts)
            indices.extend(range(start, start + self.seq_len))
        
        # Get the sequential batch
        batch = super().__getitem__(indices)
        
        # Convert to tensors and reshape
        for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
            val = getattr(batch, key)
            if isinstance(val, np.ndarray):
                setattr(batch, key, torch.as_tensor(val, device=self.device))
        
        # Reshape observations: [batch_size*seq_len, ...] -> [batch_size, seq_len, ...]
        batch_size = self.buffer_num
        for key in ['obs', 'act', 'rew', 'done', 'obs_next']:
            val = getattr(batch, key)
            if val is not None:
                setattr(batch, key, val.view(batch_size, self.seq_len, *val.shape[1:]))
        
        # Special handling for 2420-dim observations
        if batch.obs.dim() == 3:  # [batch, seq_len, 2420]
            batch.obs = batch.obs.view(batch_size, self.seq_len, 10, 242)
        if batch.obs_next.dim() == 3:
            batch.obs_next = batch.obs_next.view(batch_size, self.seq_len, 10, 242)
        
        return batch, indices

    def __getitem__(self, index):
        """Override to maintain proper batching"""
        if isinstance(index, slice):
            # Handle sequential sampling
            return super().__getitem__(index)
        elif isinstance(index, (list, np.ndarray)):
            # Handle batch sampling
            return super().__getitem__(index)
        else:
            raise TypeError("Index must be slice, list, or np.ndarray")
