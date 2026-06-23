"""
buffer.py
=========
Episodic Replay Buffer for Continual Learning.

Per the paper:
  - Fixed buffer capacity of 2,500 samples (determined via grid search).
  - Dynamically balanced: equal proportion from all preceding tasks.
  - 27-30% of each training batch is drawn from the buffer.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset


class ReplayBuffer:
    """
    A reservoir-sampled episodic memory buffer that maintains balanced
    representation across all previous tasks.
    """
    def __init__(self, capacity=2500):
        self.capacity = capacity
        self.buffer_X = []          # Feature arrays
        self.buffer_y = []          # Target arrays
        self.task_ids = []          # Which task each sample came from
        self.num_tasks_seen = 0

    def update(self, X_task, y_task, task_id):
        """
        Adds data from a new task to the buffer, maintaining balanced representation.

        Per paper: "the buffer dynamically adjusts its composition to maintain an
        equal proportion of data from all preceding tasks (e.g., 100% from the first
        task during the second, a 50/50 split during the third...)"
        
        Args:
            X_task (np.ndarray): Features for the new task.
            y_task (np.ndarray): Targets for the new task.
            task_id (int): Integer identifier for the task.
        """
        self.num_tasks_seen += 1

        # Samples to keep per task with balanced representation
        samples_per_task = self.capacity // self.num_tasks_seen

        # Subsample existing buffer to make room
        new_buffer_X = []
        new_buffer_y = []
        new_task_ids = []

        for prev_task in range(task_id):
            # Get indices of this previous task in the current buffer
            mask = [i for i, tid in enumerate(self.task_ids) if tid == prev_task]
            if len(mask) > 0:
                selected = np.random.choice(mask, size=min(samples_per_task, len(mask)), replace=False)
                for idx in selected:
                    new_buffer_X.append(self.buffer_X[idx])
                    new_buffer_y.append(self.buffer_y[idx])
                    new_task_ids.append(self.task_ids[idx])

        # Add new task samples (randomly sampled)
        n_new = min(samples_per_task, len(X_task))
        indices = np.random.choice(len(X_task), size=n_new, replace=False)
        for idx in indices:
            new_buffer_X.append(X_task[idx])
            new_buffer_y.append(y_task[idx])
            new_task_ids.append(task_id)

        self.buffer_X = new_buffer_X
        self.buffer_y = new_buffer_y
        self.task_ids = new_task_ids

    def sample(self, batch_size):
        """
        Randomly samples a batch from the buffer.

        Args:
            batch_size (int): Number of samples to draw.
            
        Returns:
            X_replay (torch.Tensor), y_replay (torch.Tensor)
        """
        if len(self.buffer_X) == 0:
            return None, None

        n = min(batch_size, len(self.buffer_X))
        indices = np.random.choice(len(self.buffer_X), size=n, replace=False)

        X = np.array([self.buffer_X[i] for i in indices])
        y = np.array([self.buffer_y[i] for i in indices])

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.buffer_X)
