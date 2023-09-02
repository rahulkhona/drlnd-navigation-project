import torch
import torch.nn.functional as F
import torch.optim as optim
import hyper_parameter_providers as hpp
from models import DQN, DuelingDQN, DuelingDQN2, DQN2
from agents import FixedTargetDQNAgent, DoubleDQNAgent
from replay_buffer import ReplayBuffer, PriorityReplayBuffer
from functools import partial
from unityagents import UnityEnvironment
import numpy as np
from collections import deque, OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime

# hyper parameters
#epsProvider = hpp.ExponentialChangeParameterProvider(1, 0.05, -0.001)
num_episodes=20000
epsProvider = hpp.LinearChangeParameterProvider(1.0, 0.1, -1/(0.75*num_episodes))
gammaProvider = hpp.ConstantParameterProvider(0.9)
alphaProvider = hpp.ConstantParameterProvider(0.05)
uniformityProvider = hpp.ConstantParameterProvider(0.9)
betaProvider = hpp.ExponentialChangeParameterProvider(0.1, 1, 0.1)
lossFun = F.mse_loss
optimizer = partial(optim.Adam, lr=alphaProvider.get())
model = DuelingDQN2
agent = FixedTargetDQNAgent
buffer_size=2000
batch_size=64
seed=0x100
update_every=4
update_target_every=150
max_steps=300
score_window_len=100
good_score_threshold=13

env = UnityEnvironment(file_name="Banana.app", no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
state_size =  len(env_info.vector_observations[0])
action_size = brain.vector_action_space_size
#arch=[1024, 256, 128]
arch = [("NL", 1024), ("R", 1024), ("L", 256), ("L", 64),("R", 64)]
#replay_buffer = PriorityReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed, betaProvider=betaProvider, uniformity=uniformityProvider.get())
replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed)
agent = DoubleDQNAgent(state_size, action_size, epsProvider, gammaProvider, replay_buffer, model, optimizer, lossFun,
                            update_every=update_every, update_target_every=update_target_every, arch=arch)

scores=[]
scores_window = deque(maxlen=score_window_len)
saved=False
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

for episode in range(1, num_episodes):
    if (episode -1) % 100 == 0:
        print("Starting Episode number:", episode)
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    agent.prepare_for_new_episode()
    for t in range(max_steps):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done)
        score += reward
        if done:
            break
        state = next_state
    scores_window.append(score)
    scores.append(score)
    if (episode - 1) % 100 == 0:
        print("latest score is", score)
    if np.mean(scores_window) > good_score_threshold:
        torch.save(agent.tmode.state_dict(), "checkpoint.pth")
        print("solved environmnet in ", episode, " episodes with avg score of ", np.mean(scores_window))
        saved = True
        break

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
print(scores)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()

