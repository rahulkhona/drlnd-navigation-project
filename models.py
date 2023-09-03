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
    """Base class for DQN models that we woud use"""
    def __init__(self, num_inputs:int, num_outputs:int):
        """Constructor
        Parameters:
            num_inputs (int) : size of state space
            num_outputs (int) : size of action space
        """
        super(DQNBase, self).__init__()
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        pass
    
    def reset_noise(self):
        pass

    def copyFrom(self, model:nn.Module):
        """Copy parameters from a source model"""
        if type(self) != type(model):
            raise ValueError(f"invalid model type {type(model)}")
        self.load_state_dict(model.state_dict())

    def updateFrom(self, model:nn.Module, tau:float=0.001):
        with torch.no_grad():
            for my_param, source_param in zip(self.parameters(), model.parameters()):
                my_param.copy_(tau*source_param.data + (1.0-tau)*my_param.data)


class DQN(DQNBase):
    """Deep Q network model"""
    def __init__(self, num_inputs:int, num_outputs:int):
        """Constructor

        Parameters:
            see DQBBase
        """
        super(DQN, self).__init__(num_inputs, num_outputs)
        self.fc1 = nn.Linear(in_features = num_inputs, out_features=64)
        self.fc2 = nn.Linear(in_features = 128, out_features=64)
        self.fc3 = nn.Linear(in_features = 64, out_features=num_outputs)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Forward pass"""
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

class DuelingDQN(DQNBase):
    """Dueling DQN Model"""
    def __init__(self, num_inputs:int, num_outputs:int):
        """Constructor.

        Parameters:
            See DQNBase documentation
        """
        super(DuelingDQN, self).__init__(num_inputs, num_outputs)
        self.fc1 = nn.Linear(in_features = num_inputs, out_features=128)
        self.fc2 = nn.Linear(in_features = 128, out_features=64)
        self.value = nn.Linear(in_features = 64, out_features=1)
        self.advantage = nn.Linear(in_features = 64, out_features=num_outputs)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Forward pass"""
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        value = self.value(output)
        advantage = self.advantage(output)
        return value + advantage - advantage.mean(dim=1, keepdims=True)

class NoisyLayer(nn.Module):
    """Layer to implement noisey network. Its a linear layer with wieghts factorized in mean and variance
       and the resultant weights and bias are equal mean + noise * variance
    """
    def __init__(self, num_inputs:int, num_outputs:int, std_init:float=0.5):
        """Constructor

        Parameters:
            num_inputs (int) : number of input features for the layer
            num_outputs (int) : number of output features for the layer
            std_int (int) : initial standard deviation to be used to compuate noise for variance.
        """
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
        """Initialie parameters for the layer"""
        mu_range = 1 / np.sqrt(self.num_inputs)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.std_init * mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.bias_sigma, self.std_init/np.sqrt(self.num_outputs))

    def factorized_noise(self, size:int)->torch.Tensor:
        """Generate 2 random vectors whose outer product can be used to compuate the noise matrix.
        This is more efficient then generating noise fo every cell of the noise matrix
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """We are at the end of new update cycle, reset noise for next update cycle
        """
        epsilon_in = self.factorized_noise(self.num_inputs)
        epsilon_out = self.factorized_noise(self.num_outputs)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """ Forward pass
        """
        # F.linear applies y = xA.transposed + b
        return F.linear(x,
                        self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon
                        )

def build_sequential_layers(num_inputs:int, num_outputs:int, arch:List[Union[int,Tuple[str, int]]])->List[nn.Module]:
    """Generic method to build sequential layers based on the architecture specified as an hyper parameter"""
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

class DQN2(DQNBase):
    """Version 2 of DQN Model, lays down layer based on architecture hyper parameter"""
    def __init__(self, num_inputs:int, num_outputs:int, arch:List[Union[int, Tuple[str, int]]]):
        """Constructor

        Paramters:

            num_inputs (int) : size of state space
            num_ouputs (int) : size of action space
            arch : List of layers either specified as layer name, size tuple or layer size for linear layers
                Layer names are L=>Linear, NL=>NoisyLayer, R=>Relu
        """
        super(DQN2, self).__init__(num_inputs, num_outputs)
        layers, prev = build_sequential_layers(num_inputs, num_outputs, arch)
        layers.append(nn.Linear(prev, num_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Froward pass"""
        return self.layers(x)

    def reset_noise(self):
        """Reset noise for any noisey layers that are in the model"""
        for l in self.layers:
            if isinstance(l, NoisyLayer):
                l.reset_noise


class DuelingDQN2(DQNBase):
    """Version 2 of DQN Model, lays down layer based on architecture hyper parameter"""
    def __init__(self, num_inputs:int, num_outputs:int, arch:List[Union[int, Tuple[str, int]]]):
        """Constructor
        Paramters:

            num_inputs (int) : size of state space
            num_ouputs (int) : size of action space
            arch : List of layers either specified as layer name, size tuple or layer size for linear layers
                Layer names are L=>Linear, NL=>NoisyLayer, R=>Relu
        """
        super(DuelingDQN2, self).__init__(num_inputs, num_outputs)
        layers, prev = build_sequential_layers(num_inputs, num_outputs, arch)
        self.layers = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)
        self.advantage = nn.Linear(prev, num_outputs)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Forward layer"""
        output = self.layers(x)
        value = self.value(output)
        advantage = self.advantage(output)
        return value + advantage - advantage.mean(dim=1, keepdims=True)

    def reset_noise(self):
        """Reset noise for any noisey layers"""
        for l in self.layers:
            if isinstance(l, NoisyLayer):
                l.reset_noise