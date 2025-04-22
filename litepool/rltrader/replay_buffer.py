import numpy as np
import torch
from tianshou.data import VectorReplayBuffer, Batch, ReplayBuffer

class SequentialReplayBuffer(VectorReplayBuffer):
    def __init__(self, total_size, seq_len=60, buffer_num=1, device='cpu'):
        assert total_size % seq_len == 0, "total_size must be divisible by seq_len"
        self.seq_len = seq_len
        buffer_size = total_size // buffer_num
        self.device = device
        self._size = buffer_size // seq_len
        self._meta_keys = ['state', 'state_h1', 'state_h2']
        super().__init__(buffer_size, buffer_num)
        self.reset()

    def _get_shape(self, key):
        """Adjust shapes for 2-layer GRU."""
        if key == 'obs' or key == 'obs_next':
            return (2420,)  # [size, 60, 2420]
        elif key == 'act':
            return (4,)     # [size, 60, 4]
        elif key in ['rew', 'done', 'terminated', 'truncated']:
            return ()
        elif key == 'state':
            return (2, 128)  # [size, 60, 2, 128] for full GRU state
        elif key == 'state_h1' or key == 'state_h2':
            return (2, 128)  # [size, 60, 2, 128]
        return None

    def add(self, batch, buffer_ids=None):
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)

        ptrs = []
        for buf_idx in buffer_ids:
            buf = self.buffers[buf_idx]
            idx = buf._index % self._size
            steps = self.seq_len  # Always write a full sequence

            # Store vectorized data (overwriting if necessary)
            for key in ['obs', 'act', 'rew', 'done', 'terminated', 'truncated', 'obs_next']:
                data = batch[key][buf_idx][:steps]
                buf.data[key][idx, :steps] = data

            # Store hidden states
            for key in self._meta_keys:
                if key in batch and batch[key] is not None:
                    data = batch[key][buf_idx][:steps]
                    buf.data[key][idx, :steps] = data

            # Handle info (simplified for brevity)
            start_idx = idx * self.seq_len
            end_idx = start_idx + steps
            if isinstance(batch.info, dict):
                for k, v in batch.info.items():
                    if k not in buf.data:
                        buf.data[k] = [None] * (self._size * self.seq_len)
                    buf.data[k][start_idx:end_idx] = v[buf_idx][:steps]

            # Update pointers
            buf._index += 1
            buf._reserved = min(buf._reserved + steps, buf.max_size)  # Still caps _reserved, but allows overwriting
            ep_rew = batch.rew[buf_idx].sum()
            ep_len = steps
            ptrs.append((idx, ep_rew, ep_len, buf_idx))

        return ptrs

    def sample(self, batch_size, train_device='cuda'):
        indices = self.sample_indices(batch_size)
        if indices.size == 0:
            return Batch(), np.array([])

        batch = Batch()
        info_list = []
        is_dict_info = None

        for buf_idx, seq_idx in indices:
            buf = self.buffers[buf_idx]
            for key in ['obs', 'act', 'rew', 'done', 'terminated', 'truncated', 'obs_next']:
                val = buf.data[key][seq_idx]
                batch[key] = np.concatenate([batch[key], val[None]], axis=0) if key in batch else val[None]
            for key in self._meta_keys:
                if key in buf.data:
                    val = buf.data[key][seq_idx]
                    batch[key] = np.concatenate([batch[key], val[None]], axis=0) if key in batch else val[None]

            start = seq_idx * self.seq_len
            end = start + self.seq_len
            if is_dict_info is None:
                is_dict_info = 'info' not in buf.data and any(k in buf.data for k in ['env_id', 'realized_pnl'])
            if is_dict_info:
                info_dict = {key: buf.data[key][start:end] for key in buf.data.keys()
                             if key not in ['obs', 'act', 'rew', 'done', 'terminated', 'truncated', 'obs_next'] + self._meta_keys}
                info_list.append(info_dict)
            else:
                info_list.append(buf.data['info'][start:end])

        if is_dict_info:
            batch.info = {key: [d[key] for d in info_list] for key in info_list[0].keys()}
        else:
            batch.info = info_list

        # Convert to tensors with correct shape
        for key in batch.keys():
            if key != 'info':
                val = batch[key]
                expected_shape = self._get_shape(key)
                if expected_shape:
                    if key in ['rew', 'done', 'terminated', 'truncated']:
                        # Scalar fields: [batch_size, seq_len] -> [batch_size, seq_len, 1]
                        batch[key] = torch.as_tensor(val, device=train_device).view(batch_size, self.seq_len, 1)
                    else:
                        # Vector fields: [batch_size, seq_len, dim]
                        batch[key] = torch.as_tensor(val, device=train_device).view(batch_size, self.seq_len, *expected_shape)
                else:
                    batch[key] = torch.as_tensor(val, device=train_device)

        return batch, indices

    def sample_indices(self, batch_size):
        valid_buffers = [i for i in range(self.buffer_num) if self.buffers[i]._reserved >= self.seq_len]
        if not valid_buffers:
            print("invalid buffers", self.buffers[0]._reserved)
            return np.array([])

        total_sequences = sum((self.buffers[i]._reserved // self.seq_len) for i in valid_buffers)
        batch_size = min(batch_size, total_sequences)
        if batch_size <= 0:
            print("batch_size < 0")
            return np.array([])

        indices = []
        for _ in range(batch_size):
            buf_idx = np.random.choice(valid_buffers)
            max_idx = (self.buffers[buf_idx]._reserved // self.seq_len) - 1
            seq_idx = np.random.randint(0, max_idx + 1)
            indices.append((buf_idx, seq_idx))
        return np.array(indices)

    def reset(self, keep_statistics=False):
        for buf in self.buffers:
            buf._index = 0
            buf._reserved = 0
            if not keep_statistics:
                buf.data = {}
                for key in ['obs', 'act', 'rew', 'done', 'terminated', 'truncated', 'obs_next']:
                    shape = self._get_shape(key)
                    buf.data[key] = np.empty((self._size, self.seq_len) + shape, dtype=np.float32)
                for key in self._meta_keys:
                    shape = self._get_shape(key)
                    if shape:
                        buf.data[key] = np.empty((self._size, self.seq_len) + shape, dtype=np.float32)
                buf.data['info'] = [None] * self._size * self.seq_len
            buf.max_size = self.seq_len * self._size
            buf._meta = Batch()
