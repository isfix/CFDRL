import numpy as np
import random
from config import Settings

class SumTree:
    """
    SumTree structure for Prioritized Experience Replay.
    Leaf nodes store priorities. Internal nodes store sum of children.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree array size: 2 * capacity - 1
        # Data array size: capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Wrapper for SumTree to handle PER logic (alpha, beta, etc.)
    """
    def __init__(self, capacity=10000, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Priority exponent (0 = random, 1 = full priority)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done):
        """
        Add new experience with max priority.
        """
        # New experiences get max priority to ensure they are replayed at least once
        max_prio = np.max(self.tree.tree[-self.tree.capacity:]) 
        if max_prio == 0:
            max_prio = 1.0 # Default if empty
            
        data = (state, action, reward, next_state, done)
        self.tree.add(max_prio, data)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch based on priority.
        beta: Importance Sampling exponent (anneals from 0.4 to 1.0)
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, p, data) = self.tree.get(s)
            
            # If tree isn't full/perfect, data might be 0/None. Retry or handle.
            if data == 0 or data is None:
                 # Fallback: Random sample if tree retrieval fails (edge case)
                 # Re-roll s in valid range or pick random filled index
                 valid_idx = random.randint(0, self.tree.count - 1)
                 data = self.tree.data[valid_idx]
                 idx = valid_idx + self.capacity - 1
                 p = self.tree.tree[idx]

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Calculate Importance Sampling Weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.count * sampling_probabilities, -beta)
        is_weights /= is_weights.max() # Normalize

        # Prepare batch tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            list(states),                          # Raw list (may be composite tuples)
            np.array(actions), 
            np.array(rewards, dtype=np.float32), 
            list(next_states),                     # Raw list (may be composite tuples)
            np.array(dones, dtype=bool),
            idxs,
            np.array(is_weights, dtype=np.float32)
        )

    def update_priorities(self, idxs, errors):
        """
        Update priorities based on TD errors.
        """
        for idx, error in zip(idxs, errors):
            p = (abs(error) + 1e-5) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.count
