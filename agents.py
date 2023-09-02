import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer, PriorityReplayBuffer, DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE, DEFAULT_SEED
from models import DQN, DuelingDQN
from hyper_parameter_providers import HyperParameterProvider
import typing
from typing import Union

DEFAULT_UPDATE_EVERY:int=4
DEFAULT_TARGET_UPDATE:int=10
class FixedTargetDQNAgent:
    def __init__(self, state_size:int, action_size:int,epsProvider:HyperParameterProvider,gammaProvider:HyperParameterProvider,
                replayBuffer:ReplayBuffer, model:Union[DQN,DuelingDQN], optimizer:optim.Optimizer, lossFn, update_every:int=DEFAULT_UPDATE_EVERY,
                update_target_every:int=DEFAULT_TARGET_UPDATE):
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
        self.lmodel = model(state_size, action_size).to(self.device)
        self.tmodel = model(state_size, action_size).to(self.device)
        self.tmodel.copyFrom(self.lmodel)
        self.optimizer = optimizer(self.tmodel.parameters())
        self.update_target_every = update_target_every
        self.update = 0

    def is_priority_replay(self):
        return isinstance(self.replayBuffer, PriorityReplayBuffer)

    def prepare_for_new_episode(self):
        self.eps = self.epsProvider.get()
        self.gamma = self.gammProvider.get()

    def get_target_value(self, actions, states):
        target_q_table = self.tmodel(states)
        targets = target_q_table.detach().max(1, keepdims=True)[0]
        return targets

    def learn(self):
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
        if self.update % self.update_target_every == 0:
            # This is hard update, we can also use a moving every update every
            # learning step and update target model with small fraction of local model
            self.tmodel.copyFrom(self.lmodel)


    def act(self, state:np.ndarray):
        if np.random.random() < self.eps:
            return np.random.choice(np.arange(self.action_size))
        else:
            with torch.no_grad():
                return self.lmodel(torch.from_numpy(state).float().unsqueeze(0).to(self.device)).max(-1)[1].cpu().numpy()

    def step(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, done:bool):
        self.replayBuffer.add(state, action, reward, next_state, done)
        self.learn()


class DoubleDQNAgent(FixedTargetDQNAgent):
    def __init__(self, state_size:int, action_size:int,epsProvider:HyperParameterProvider,gammaProvider:HyperParameterProvider,
                replayBuffer:ReplayBuffer, model:Union[DQN,DuelingDQN], optimizer:optim.Optimizer, lossFn, update_every:int=DEFAULT_UPDATE_EVERY,
                update_target_every:int=DEFAULT_TARGET_UPDATE):
        super(DoubleDQNAgent, self).__init__(state_size, action_size, epsProvider, gammaProvider, replayBuffer, model, optimizer, lossFn, update_every, update_target_every)

    def get_target_value(self, actions, states):
        target_q_table = self.tmodel(states)
        current_q_table = self.lmodel(states)
        best_actions = current_q_table.max(1)[1].unsqueeze(1)
        targets = target_q_table.gather(1, best_actions)
        return targets