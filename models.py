import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import typing
from typing import List, Union, Tuple
import logging

ml_logger = logging.getLogger("ModelsLogger")

class DQNBase(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int):
        super(DQNBase, self).__init__()
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        pass
    
    def reset_noise(self):
        pass


class DQN(DQNBase):
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


class DuelingDQN(DQNBase):
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

class NoisyLayer(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, std_init:float=0.5):
        super(NoisyLayer, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.std_init = std_init

        # F.linear applies y = xA.transposed + b. Hence dimensions are
        # transposed for weights

        self.weight_mu = nn.Parameter(torch.Tensor(num_outputs, num_inputs))
        self.weight_sigma = nn.Parameter(torch.Tensor(num_outputs, num_inputs))
        self.register_buffer(
            "weight_epsilon", torch.Tensor(num_outputs, num_inputs)
        )
        self.bias_mu = nn.Parameter(torch.Tensor(num_outputs))
        self.bias_sigma = nn.Parameter(torch.Tensor(num_outputs))
        self.register_buffer(
            "bias_epsilon", torch.Tensor(num_outputs)
        )

        self.init_params()
        self.reset_noise()

    def init_params(self):
        mu_range = 1 / np.sqrt(self.num_inputs)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.std_init * mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.bias_sigma, self.std_init/np.sqrt(self.num_outputs))

    def factorized_noise(self, size:int)->torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self.factorized_noise(self.num_inputs)
        epsilon_out = self.factorized_noise(self.num_outputs)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # F.linear applies y = xA.transposed + b
        return F.linear(x,
                        self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon
                        )

def build_sequential_layers(num_inputs:int, num_outputs:int, arch:List[Union[int,Tuple[str, int]]])->List[nn.Module]:
    layers = []
    prev = num_inputs
    for a in arch:
        key = "L"
        if type(a) == int:
            value = a
        else:
            key = a[0]
            value = a[1]
        if key == 'L':
            layers.append(nn.Linear(prev, value))
        elif key == 'NL':
            layers.append(NoisyLayer(prev, value))
        elif key == 'R':
            layers.append(nn.ReLU())
        else:
            raise ValueError("Invalid entry ", key)
        prev = value

    return layers, prev

class DQN2(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, arch:List[Union[int, Tuple[str, int]]]):
        super(DQN2, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        #layers = []
        #prev = num_inputs
        #for i in range(1, len(arch)):
        #    layers.append(nn.Linear(prev, arch[i]))
        #    layers.append(nn.ReLU())
        #    prev = arch[i]
        #layers.append(nn.Linear(prev, num_outputs))
        layers, prev = build_sequential_layers(num_inputs, num_outputs, arch)
        layers.append(nn.Linear(prev, num_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.layers(x)

    def copyFrom(self, model:nn.Module):
        if type(self) != type(model):
            raise ValueError(f"invalid model type {type(model)}")
        self.load_state_dict(model.state_dict())

    def reset_noise(self):
        for l in self.layers:
            if isinstance(l, NoisyLayer):
                l.reset_noise


class DuelingDQN2(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, arch:Union[List[int], OrderedDict]):
        super(DuelingDQN2, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        #layers = []
        #prev = num_inputs
        #for i in range(1, len(arch)):
        #    layers.append(nn.Linear(prev, arch[i]))
        #    layers.append(nn.ReLU())
        #    prev = arch[i]
        layers, prev = build_sequential_layers(num_inputs, num_outputs, arch)
        self.layers = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)
        self.advantage = nn.Linear(prev, num_outputs)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        output = self.layers(x)
        value = self.value(output)
        advantage = self.advantage(output)
        return value + advantage - advantage.mean(dim=1, keepdims=True)

    def copyFrom(self, model:nn.Module):
        if type(self) != type(model):
            raise ValueError(f"invalid model type {type(model)}")
        self.load_state_dict(model.state_dict())

    def reset_noise(self):
        for l in self.layers:
            if isinstance(l, NoisyLayer):
                l.reset_noise