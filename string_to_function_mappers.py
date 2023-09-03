import hyper_parameter_providers as hpp
import torch.optim as optim
import torch.nn.functional as F
from models import DQN2, DuelingDQN2, DQN, DuelingDQN
from replay_buffer import ReplayBuffer, PriorityReplayBuffer
from agents import FixedTargetDQNAgent, DoubleDQNAgent

hpp_mapper = {
   hpp.ConstantParameterProvider.__name__  : hpp.ConstantParameterProvider,
   hpp.ExponentialChangeParameterProvider.__name__  : hpp.ExponentialChangeParameterProvider,
   hpp.LinearChangeParameterProvider.__name__  : hpp.LinearChangeParameterProvider
}

loss_fn_mapper = {
    F.mse_loss.__name__ : F.mse_loss,
    F.l1_loss.__name__ : F.l1_loss,
    F.smooth_l1_loss.__name__ : F.smooth_l1_loss
}

optim_mapper = {
    optim.Adam.__name__ : optim.Adam,
    optim.RMSprop.__name__ : optim.RMSprop
}

model_mapper = {
    DQN2.__name__ : DQN2,
    DuelingDQN2.__name__ : DuelingDQN2,
    DQN.__name__ : DQN,
    DuelingDQN.__name__ : DuelingDQN,
}

replay_buffer_mapper = {
    ReplayBuffer.__name__ : ReplayBuffer,
    PriorityReplayBuffer.__name__ : PriorityReplayBuffer
}

agent_mapper = {
    FixedTargetDQNAgent.__name__ : FixedTargetDQNAgent,
    DoubleDQNAgent.__name__ : DoubleDQNAgent
}
