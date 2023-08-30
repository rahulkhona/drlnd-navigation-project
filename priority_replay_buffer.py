import typing
import logging
from typing import List, Tuple, Optional
from collections import deque, namedtuple
import torch
import random
import numpy as np
from replay_buffer import ReplayBuffer, DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE, DEFAULT_SEED

pe_logger = logging.getLogger("PrioirtyExperienceLogger")
Experience = namedtuple("PriorityExperience", field_names=[
    "state", "action", "reward", "next_state", "done", "error"
])
DEFAULT_UNIFORMITY=0.01
DEFAULT_EPSILON=1e-5

class PriorityReplayBuffer(ReplayBuffer):
    """priority experience replay buffer"""

    def __init__(self, buffer_size:int=DEFAULT_BUFFER_SIZE, batch_size:int=DEFAULT_BATCH_SIZE, seed:int=DEFAULT_SEED,
                 uniformity:float=DEFAULT_UNIFORMITY, epsilon=DEFAULT_EPSILON):
        """Constructor.

        Parameters:
            buffer_size (int) : See ReplayBuffer
            batch_size (int) : See ReplayBuffer
            seed (int) : See ReplayBuffer
            uniformity (float):  controls degree of uniform random sampling of samples. uniformity = 0 results
                in completely uniform random selection and value of 1 selects solely based on relatvie size
                of error.
        """
        super(PriorityReplayBuffer, self).__init__(buffer_size, batch_size, seed)
        self.epsilon=epsilon
        if uniformity <= 0 or uniformity > 1:
            raise ValueError("Uniformity should be between 0 and 1")
        self.uniformity = uniformity

    def add(self, state:List[float], action:int, reward:float, next_state:List[float], done:bool, error:float):
        """Add an experience to the buffer

        Parameters:
            state : see ReplayBuffer
            action : see ReplayBuffer
            reward : see ReplayBuffer
            next_sate : see ReplayBuffer
            done: see ReplayBuffer
            error : TD error of the the step that agent took
        """
        error = (np.abs(error) + self.epsilon) ** self.uniformity
        self.memory.append(Experience(state, action, reward, next_state, done, error))

    def compute_probabilities(self)->np.ndarray:
        """Compute probabilities based on relative size of errors

        Returns:
            array of probabilities for each experience in the buffer
        """
        errors = np.vstack([e.error for e in self.memory])
        s = np.sum(errors)
        return errors.squeeze()/s

    def compute_importance_sampling(self, sample_probs:np.ndarray, beta:float)->np.ndarray:
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
        if beta < 0. or beta > 1.0:
            raise ValueError("beta should be between 0 and 1")
        n = sample_probs.shape[0]
        return 1/((n*sample_probs)**beta)

    def sample(self, batch_size:int=None, device:str="cpu", beta:float=1)->Optional[List[Tuple[torch.Tensor,
                                                                                                torch.Tensor,
                                                                                                torch.Tensor,
                                                                                                torch.Tensor,
                                                                                                torch.Tensor,
                                                                                                torch.Tensor]]]:
        """ Generate samples for training

        Parameters:
            batch_size (int) : see ReplayBuffer
            device (str) : Which device should the resulting tensors be sent to
            beta (float) : Beta value to use for importance sampling weight computation

        Returns :
            list of tuple of tensors for states, actions, rewards, next_states, dones and importance weights
        """
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size <= 0 or batch_size > self.buffer_size:
            raise ValueError(f"batch size should greater than 0 and less than equal to {self.buffer_size}")
        
        probs = self.compute_probabilities()
        ind_exps = random.choices(list(enumerate(self.memory)), weights=probs, k=batch_size)
        indices, experiences = list(zip(*ind_exps))
        if len(self) < batch_size:
            pe_logger.warning("sample method should not be called if sufficient experiences are not gathered yet")
            return None
        sample_probs = probs[[indices]].squeeze()
        importance_samples = self.compute_importance_sampling(sample_probs, beta)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).int().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences])).bool().to(device)
        importances = torch.from_numpy(np.vstack([i for i in importance_samples])).float().to(device)

        return (states, actions, rewards, next_states, dones, importances)
