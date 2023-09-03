import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer, PriorityReplayBuffer, DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE, DEFAULT_SEED
from models import DQN, DuelingDQN
from hyper_parameter_providers import HyperParameterProvider
from models import DQNBase
import typing
from typing import List, Tuple, Union

DEFAULT_UPDATE_EVERY:int=4
DEFAULT_TARGET_UPDATE:int=10
class FixedTargetDQNAgent:
    """FixedTarget DQN agent"""
    def __init__(self, state_size:int, action_size:int,epsProvider:HyperParameterProvider,gammaProvider:HyperParameterProvider,
                replayBuffer:ReplayBuffer, model:DQNBase, optimizer:optim.Optimizer, lossFn, update_every:int=DEFAULT_UPDATE_EVERY,
                update_target_every:int=DEFAULT_TARGET_UPDATE, arch:List[Union[int, Tuple[str, int]]]=[128,64],
                useTau:int=0, tau:float=0.001):
        """Constructor

        Parameters:
            state_size (int) : Size of the state space
            action_size (int) : Size of the action space
            epsProvder (HyperParameterProvider) : class that returns hyper paramter value for exploration
            gammaProvder (HyperParameterProvider) : class that returns hyper paramter value for discount rate
            replayBuffer (ReplayBuffer) : instance of either ReplayBuffer or ProritityReplayBuffer
            model (DQNBase) : Model that is sub-class fo DQNBase that should be constructed for approximating Q-value
            optimizer (Optimzer) : An instance of partial function wrapping Adam, RMSE or other optimizer that should be used for learning
            lossFn : loss function that should be used for graident descent
            update_every (int) : How frequently should the agent learn
            update_target_every (int) : How often should the target model be replaced with local model. We are using hard copy
                as opposed to moving average update
            arch (List) : List of layers that should be used in the model

        """
        if not callable(lossFn):
            raise ValueError("lossFn should be callable")
        self.state_size = state_size
        self.action_size = action_size
        self.epsProvider = epsProvider
        self.gammProvider = gammaProvider
        self.replayBuffer = replayBuffer
        self.lossFn = lossFn
        self.update_every = update_every
        self.t_step = 0
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        self.lmodel = model(state_size, action_size, arch).to(self.device)
        self.tmodel = model(state_size, action_size, arch).to(self.device)
        self.optimizer = optimizer(self.tmodel.parameters())
        self.update_target_every = update_target_every
        self.useTau = useTau,
        self.tau = tau
        self.update = 0

    def is_priority_replay(self):
        """Returns true if agent should use priority replay"""
        return isinstance(self.replayBuffer, PriorityReplayBuffer)

    def prepare_for_new_episode(self):
        """Prepare for next episode, essentially udpate hyper parameters"""
        self.eps = self.epsProvider.get()
        self.gamma = self.gammProvider.get()

    def get_target_value(self, actions, states):
        """Get the target value given the next state. This is used to compuate TD error"""
        target_q_table = self.tmodel(states)
        targets = target_q_table.detach().max(1, keepdims=True)[0]
        return targets

    def learn(self):
        """Learn"""
        self.t_step += 1
        if (self.t_step % self.update_every) > 0:
            return
        if len(self.replayBuffer) < self.replayBuffer.batch_size:
            return
        experiences, priorities, indices = self.replayBuffer.sample()
        states = experiences[0]
        actions = experiences[1]
        rewards = experiences[2]
        next_states = experiences[3]
        dones = experiences[4]
        current_q_table = self.lmodel(states)
        expected_value = current_q_table.gather(1, actions)
        target_next = self.get_target_value(actions, next_states)
        target_value = (rewards + self.gamma * target_next * (1 - dones))
        element_loss = self.lossFn(expected_value, target_value, reduction="none")
        errors = element_loss.detach()
        if priorities is not None:
            element_loss *= priorities
        loss = torch.mean(element_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replayBuffer.update_errors(errors.cpu().numpy(), indices)
        self.update += 1
        if self.useTau:
            self.tmodel.updateFrom(self.lmodel, self.tau)
        elif self.update % self.update_target_every == 0:
            # This is hard update, we can also use a moving every update every
            # learning step and update target model with small fraction of local model
            self.tmodel.copyFrom(self.lmodel)


    def act(self, state:np.ndarray):
        """Get the next action given the current state"""
        if np.random.random() < self.eps:
            return np.random.choice(np.arange(self.action_size))
        else:
            with torch.no_grad():
                return self.lmodel(torch.from_numpy(state).float().unsqueeze(0).to(self.device)).max(-1)[1].cpu().numpy()

    def step(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, done:bool):
        """Record rewards and next state information from the step that env had taken based on our action"""
        self.replayBuffer.add(state, action, reward, next_state, done)
        self.learn()


class DoubleDQNAgent(FixedTargetDQNAgent):
    """DoubleDQN Agent implementation"""
    def __init__(self, state_size:int, action_size:int,epsProvider:HyperParameterProvider,gammaProvider:HyperParameterProvider,
                replayBuffer:ReplayBuffer, model:DQNBase, optimizer:optim.Optimizer, lossFn, update_every:int=DEFAULT_UPDATE_EVERY,
                update_target_every:int=DEFAULT_TARGET_UPDATE, arch:List[Union[int, Tuple[str, int]]]=[128, 64], useTau:int=0,tau:float=0.001):
        """Constructor

        Paramters:
            ssee FixedTargetDQNAgent, we use same parameters
        """
        super(DoubleDQNAgent, self).__init__(state_size, action_size, epsProvider, gammaProvider, replayBuffer, model, optimizer, lossFn, update_every,
                                             update_target_every, arch, useTau, tau)

    def get_target_value(self, actions, states):
        """Get the target value given a next state, used for computing TD error"""
        target_q_table = self.tmodel(states)
        current_q_table = self.lmodel(states)
        best_actions = current_q_table.max(1)[1].unsqueeze(1)
        targets = target_q_table.gather(1, best_actions)
        return targets