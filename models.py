import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import typing
import logging

ml_logger = logging.getLogger("ModelsLogger")

class DQN(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int):
        super(DQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(in_features = num_inputs, out_features=64)
        self.fc2 = nn.Linear(in_features = 128, out_features=64)
        self.fc3 = nn.Linear(in_features = 64, out_features=num_outputs)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

    def copyFrom(self, model:nn.Module):
        if type(self) != type(model):
            raise ValueError(f"invalid model type {type(model)}")
        self.load_state_dict(model.state_dict())


class DuelingDQN(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int):
        super(DuelingDQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(in_features = num_inputs, out_features=128)
        self.fc2 = nn.Linear(in_features = 128, out_features=64)
        self.value = nn.Linear(in_features = 64, out_features=1)
        self.advantage = nn.Linear(in_features = 64, out_features=num_outputs)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        value = self.value(output)
        advantage = self.advantage(output)
        return value + advantage - advantage.mean(dim=1, keepdims=True)

    def copyFrom(self, model:nn.Module):
        if type(self) != type(model):
            raise ValueError(f"invalid model type {type(model)}")
        self.load_state_dict(model.state_dict())