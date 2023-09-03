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
import os
import random
import pickle
from typing import List, Tuple
from filenames import CHECKPOINT_FILE, HYPER_PARAMETERS_FILE

class Tester:
    def __init__(self, iteration_path:str):
        self.checkpoint_file = os.path.join(iteration_path, CHECKPOINT_FILE)
        self.hyper_parameters_file = os.path.join(iteration_path, HYPER_PARAMETERS_FILE)
    
    def get_hyper_parameters(self)->dict:
        return pickle.load(open(self.hyper_parameters_file, "rb"))

    def test(self, num_episodes:int=100):
        hpdict = self.get_hyper_parameters()
        modelType = hpdict['model']
        env = UnityEnvironment(file_name="Banana.app", no_graphics=True)
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=False)[brain_name]
        model=model(hpdict['state_size'], hpdict['action_size']).to("cpu")
        if scores_window_len is None:
            scores_window_len = hpdict['scores_window_len']
        if num_episodes is None:
            num_episodes = hpdict['num_episodes']
        scores = []
        for i in range(num_episodes):
            env_info = env.reset(train_mode=False)[brain_name]
            state = env_info.vector_observations[0]
            step_t = 0
            score = 0
            while step_t <= 300:
                step_t += 1
                action = model.forward(state).argmax(dim=1)
                env_info = env.step(action)[brain_name]
                state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                score += reward
                if done:
                    break
            scores.append(score)
        print("average score is ", np.mean(scores))
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()