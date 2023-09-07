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

## Project details
For the Banana.app the state has 37 dimensions of continous features and has 4 discrete actions as shown in Introduction. The agent internally uses the latest state + 3 previous states to form a larger 148 dimenstional state space. This has helped the model converge and solve the environemnt in 451 steps.

## Getting started
1. [Download the project code from] (https://github.com/rahulkhona/drlnd-navigation-project)
2.  [Follow the instructions to download course repostitory and setup the environment](https://github.com/udacity/Value-based-methods#dependencies). 
   - <i>Note:</i> replace requirements.txt under python folder with following as that env is old and dependencies dont install well. Essentially relax the version constraints and also remove tensorflow.
      - pillow
      - matplotlib
      - numpy
      - jupyter
      - pytest
      - docopt
      - pyyaml
      - protobuf==3.5.2
      - grpcio
      - torch
      - pandas
      - scipy
      - ipykernel
3. in addition you would need to pip install requirements.txt that you checkout from this repository as shown in the instructions below. 

## Instructions to training the model
1. Create a virtual environment using conda or tool of your choice with python 3.6
2. activate the new environment
3. install dependencies using `python -m pip install requirements.txt`
4. Read report.md to learn more about the implementation
5. start the Jupyter Notebook in your environment. For macos use following command
   1. `~/opt/anaconda3/bin/jupyter_mac.command`
6. Run all the cells in the notebook starting from 1st cell except cell for Taking Random action cell

