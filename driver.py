import torch
import torch.nn.functional as F
import torch.optim as optim
import hyper_parameter_providers as hpp
from models import DQN, DuelingDQN
from agents import FixedTargetDQNAgent, DoubleDQNAgent
from replay_buffer import ReplayBuffer, PriorityReplayBuffer
from functools import partial
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# hyper parameters
epsProvider = hpp.ExponentialChangeParameterProvider(0.9, 0.05, -0.1)
gammaProvider = hpp.ConstantParameterProvider(0.9)
alphaProvider = hpp.ConstantParameterProvider(0.05)
uniformityProvider = hpp.ConstantParameterProvider(0.9)
betaProvider = hpp.ExponentialChangeParameterProvider(0.1, 1, 0.1)
lossFun = F.mse_loss
optimizer = partial(optim.Adam, lr=alphaProvider.get())
model = DQN
agent = FixedTargetDQNAgent
buffer_size=2000
batch_size=64
seed=0x42
update_every=4
update_target_every=10
max_steps=200
num_episodes=10
score_window_len=1000
good_score_threshold=100

env = UnityEnvironment(file_name="Banana.app", no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
state_size =  len(env_info.vector_observations[0])
action_size = brain.vector_action_space_size
replay_buffer = PriorityReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed, betaProvider=betaProvider, uniformity=uniformityProvider.get())
agent = FixedTargetDQNAgent(state_size, action_size, epsProvider, gammaProvider, replay_buffer, model, optimizer, lossFun,
                            update_every=update_every, update_target_every=update_target_every)

scores=[]
scores_window = deque(maxlen=score_window_len)
saved=False
for episode in range(1, num_episodes):
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
    if np.mean(scores_window) > good_score_threshold:
        torch.save(agent.tmode.state_dict(), "checkpoint.pth")
        saved = True
        break

print(scores)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()

