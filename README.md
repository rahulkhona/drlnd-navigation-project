[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Instructions
1. [Download the project code from] (https://github.com/rahulkhona/drlnd-navigation-project)
1. [Follow the instructions to download course repostitory and setup the environment](https://github.com/udacity/Value-based-methods#dependencies). 
   - Note use requirements.txt from downloaded project to install packages instead of python/requirements.txt as many of the libraries dont work with old env.
   - I have not been able to build a model that solves the environment, so have created functionality to randomly search for different hyper parameters and try model with those to find the best model.
   
### Code structure
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