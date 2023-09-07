# High level report
## Learning Algorithm
The learning algorithm uses 3 dense layeer DQN model. The 2 hidden layers have 64 features and the final output layer  produces 4 outputs, one for each of the actions. The agent uses Fixed Q-Target algorithm and it uses non priority replay buffer. They key change in the system is that the system concatenates current state returned by environemnt along with 3 prior remembered prior states to form the state that the agent and the models use. This has allowed the agent to learn and solve the environment in 451 states. Without use of state history, the agent was not able to learn with several different combinations of hyper paramters.

## Hyper Parameters used
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 64         
GAMMA = 0.99            
TAU = 1e-3              
LR = 5e-4               
UPDATE_EVERY = 4        
HISTORY_SIZE = 3 resulting in 4 times the state size. This is hard-coded in the model implementation.
seed=0x21

### DQN Model architecture

L(Nx64)->ReLU->L(64x64)->Relu->L(64x4)
Optimizer = Adam with learning rate of 5e-4

### Agent Architecture
Fixed Q-Target with ReplayBuffer with e-greedy policy. Epison starts with 1 and decays at the rate of .995 to .01 and then stays constant at .01

## Additional functionality implemented
In addition to the working implementation in navigation.ipnyb, i also experimented with  DuelingDQN models, NoiseyLayer, PrioirtyBuffer, Double Q agent with several combinations of hyper parameters. I also implemented to perform model search by randomly chosing hyper parameters, model architectures, agent architecture and replay buffer type. This code can be found in following files

1. <b>modelys.py</b> This file contains implementation for DQN model, DuelingDQN Mode as well as NoisyLayer
2. <b>agents.py</b> This file contains implementation for FixedTargetDQNAgent as well as DoubleDQNAgent.
3. <b>replay_buffer.y</b> This file contains implementation for ReplayBuffer as well as PriorityReplayBuffer
4. <b>hyper_parameter_providers.py</b> This file contains implementation for providers for constant parameter values, linearchange as well as exponential change parameter values. These are used to provide values for alpha, gamma, eps, beta, etc.
5. <b>filenames.py</b> constant filenames for variety of files that are being written in output folder
6. <b>Banana.app</b> the unity app environment
7. <b>trainer.py</b> This file contains functionality to randomly generate hyper parameters, model architectures as well as randomly choose from agent archtiecture and type of replay buffer and train a model with those parameters.
8. <b>model_search.py</b> This is the main driver that invokes trainer in a loop for upto a specified amount of time in hours to generate and train different models. It also has options to find the best models from models trained so far as well as sort all the models in the order of their performance. you can invoke <code>python model_search.py -h </code> to see all the options
9. <b>requirements.txt</b> list of all the packages needed
10. <b>*_test.py</b> unit tests

## Ideas Future work
Model search implementation was done randomly, this can be improved by using algorithms such as 
1. Hill climbing
2. Steepest ascent hill climbing
3. Simlated Healing
4. Adaptinve noise scaling
5. evolution strategies
to explore hyperparameter space in a bit more directed fashion and find better models faster.

We I would like to explore policy gradient methods to help find best hyper parameters by training 2 agents simultaneously 1 agent to learn optimal policies for hyper parameters and second agent to train on banana game. Scores of the 2nd agent can be used to train the policy model.