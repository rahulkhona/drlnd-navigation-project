import typing
import logging
from typing import List, Optional, Tuple
from collections import deque, namedtuple
import torch
import random
import numpy as np
import hyper_parameter_providers as hpp


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
rb_logger = logging.getLogger("ReplayBufferLogger")
DEFAULT_BUFFER_SIZE=1000
DEFAULT_BATCH_SIZE=64
DEFAULT_SEED=0x42
DEFAULT_EPSILON=1e-5
DEFAULT_MAX_ERROR=1000
DEFAULT_UNIFORMITY=0.01
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

    def add(self, state: np.ndarray, action:int, reward: float, next_state:np.ndarray, done:bool):
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

    def sample(self, device:str='cpu')->Optional[List[Tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]]]:
        """Returns a sample for training

        Parameters:
            device (str) : device on which to run this computation
        """

        batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)
        assert len(experiences) == batch_size
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).int().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences])).bool().to(device)

        return (states, actions, rewards, next_states, dones)


class PriorityReplayBuffer(ReplayBuffer):
    """ Priority buffer implementation"""
    def __init__(self, betaProvider:hpp.HyperParameterProvider, buffer_size:int=DEFAULT_BUFFER_SIZE, batch_size:int=DEFAULT_BATCH_SIZE,
                 seed:int=DEFAULT_SEED, eps:float=DEFAULT_EPSILON,max_error=DEFAULT_MAX_ERROR, uniformity=DEFAULT_UNIFORMITY):
        """Constructor

        Parameters:
            betaProvider (HyperParameterProvider) : provider of beta values as exeuction progresses
            buffer_size (int) : see ReplayBuffer
            batch_size (int) : see ReplayBuffer
            seed (int) : see ReplayBuffer
            eps (float) : small number to add to errors in case errors are 0
            max_error(float) : Initial error to use for new samples
            uniformity (float):  controls degree of uniform random sampling of samples. uniformity = 0 results
                in completely uniform random selection and value of 1 selects solely based on relatvie size
                of error.
        """
        super(PriorityReplayBuffer, self).__init__(buffer_size, batch_size, seed)
        if uniformity <= 0 or uniformity > 1:
            raise ValueError("Uniformity should be between 0 and 1")
        self.eps = eps
        self.betaProvider = betaProvider
        self.errors = deque(maxlen=self.buffer_size)
        self.max_error = max_error
        self.uniformity=uniformity
    
    def add(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, done:bool):
        """Add a sample to experience buffe

        Parameters:
            state : see ReplayBuffer
            action : see ReplayBuffer
            reward : see ReplayBuffer
            next_sate : see ReplayBuffer
            done: see ReplayBuffer
        """
        super(PriorityReplayBuffer, self).add(state, action, reward, next_state, done)
        self.errors.append(self.max_error)
    
    def compute_probabilities(self)->np.ndarray:
        """Compute probabilities based on relative size of errors

        Returns:
            array of probabilities for each experience in the buffer
        """
        errors = np.vstack([e for e in self.errors])
        s = np.sum(errors)
        return errors.squeeze()/s

    def compute_importance_sampling(self, sample_probs:np.ndarray)->np.ndarray:
        """Compute importance score for each of the sample being considered. This would be
        multiplied to learning rate for each sample and is given by following formulat 
        1/(N*P(i))^beta for sample i

        Parameters:
            sample_probs (ndarray) : Array of probabilities corresponding to sample experiences
            beta (float) : Hyperparameter to control how much weight is given to importance sampling
            while learning. beta = 1 uses importance sampling weight to normalize learning from each
            experience where as smaller values give result in more reliance on size of the error to
            give its weight. This parameter should be varied from very small value to 1 as learning progresses.

        Returns :
            array of importance weights adjusted by beta
        """
        beta = self.betaProvider.get()
        if beta < 0. or beta > 1.0:
            raise ValueError("beta should be between 0 and 1")
        n = sample_probs.shape[0]
        return 1/((n*sample_probs)**beta)

    def sample(self, device:str="cpu")->List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                             torch.Tensor, torch.Tensor], np.ndarray, np.ndarray]]:
        """Return a sample of batch_size

        Parameters:
            device (str) : device to move the tensors onto

        Returns:
            List of tuple of 3 elements
                1st element tuple of states, actions, rewards, next_states, dones
                2nd element importance_weights
                3re element indices of samples 
        """
        batch_size = self.batch_size
        probs = self.compute_probabilities()
        ind_exps = random.choices(list(enumerate(self.memory)), weights=probs, k=batch_size)
        indices, experiences = list(zip(*ind_exps))
        sample_probs = probs[[indices]].squeeze()
        importance_samples = self.compute_importance_sampling(sample_probs)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).int().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences])).bool().to(device)
        importances = torch.from_numpy(np.vstack([i for i in importance_samples])).float().to(device)

        return ((states, actions, rewards, next_states, dones), importances, indices)

    def update_errors(self, errors:np.ndarray, indices:np.ndarray):
        """Update errors once they are computed for the sample"""
        for ind, err in zip(indices, errors):
            if ind < 0 or ind > len(self.errors):
                raise ValueError("indices contain out of bounds values")
            self.errors[ind] = err
