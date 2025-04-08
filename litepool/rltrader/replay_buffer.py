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

class StackedVectorReplayBuffer(VectorReplayBuffer):
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self), size=batch_size)
        batch = super().__getitem__(indices)

        # Reshape obs: [batch, 60, 2420] -> [batch, 60, 10, 242]
        obs = batch.obs.view(batch_size, self.stack_num, 10, 242)
        batch.obs = obs.to(self.device)
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
