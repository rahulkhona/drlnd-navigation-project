import typing
import logging
from typing import List, Optional, Tuple
from collections import deque, namedtuple
import torch
import random
import numpy as np


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
rb_logger = logging.getLogger("ReplayBufferLogger")
DEFAULT_BUFFER_SIZE=1000
DEFAULT_BATCH_SIZE=64
DEFAULT_SEED=0x42
class ReplayBuffer:
    """Basic Experience Replay buffer"""

    def __init__(self, buffer_size:int=DEFAULT_BUFFER_SIZE, batch_size:int=DEFAULT_BATCH_SIZE, seed:int=DEFAULT_SEED):
        """Constructor

        Parameters:
            buffer_size (int) : Number of samples to hold in the buffer. Default is 1000.
            batch_size (int) : batch size for training. Default is 64
            seed (int) : seed for random number generator. Default is 0x42
        """
        rb_logger.debug("buffer_size %d batch_size %d seed %d", buffer_size, batch_size, seed)
        if buffer_size < batch_size:
            raise ValueError("Buffer size must be larger than or equal to batch size")
        if batch_size <= 0:
            raise ValueError("Batch Size must be greater than 0")
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory=deque(maxlen=self.buffer_size)

    def add(self, state: List[float], action:int, reward: float, next_state:List[float], done:bool):
        """Add a sample to the buffer.
        
        Parameters:
            state (List[float]) : current state to be saved in memory
            action (int) : action that was last taken
            reward (float) : reward that was received for last state, action pair
            next_state (List[float]): Next state
            done bool : Wether the episode is done or not
        """
        self.memory.append(Experience(state, action, reward, next_state, done))
    def __len__(self):
        """Returns current length of buffer"""
        return len(self.memory)

    def sample(self, batch_size:int=None, device:str='cpu')->Optional[List[Tuple[torch.Tensor,
                                                                                               torch.Tensor,
                                                                                               torch.Tensor,
                                                                                               torch.Tensor,
                                                                                               torch.Tensor]]]:
        """Returns a sample for training

        Parameters:
            batch_size (int) : size of the sample that needs to be returned, default is 64
        """
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size <= 0 or batch_size > self.buffer_size:
            raise ValueError(f"batch size must be greater than 0 and less than or equal to {self.buffer_size}")
        rb_logger.debug("Batch size is %d", batch_size)

        experiences = random.sample(self.memory, k=batch_size)
        if len(experiences) < batch_size:
            rb_logger.warning("sample method should not be called if sufficient experiences are not gathered yet")
            return None
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).int().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences])).bool().to(device)

        return (states, actions, rewards, next_states, dones)