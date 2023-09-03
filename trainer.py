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
import typing
from typing import List, Tuple


class Trainer():
    def __init__(self, iteration_name : str, output_path : str, seed:int=0x21):
        self.name = iteration_name
        self.output_path = output_path

        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        self.my_path = os.path.join(self.output_path, self.name)
        os.makedirs(self.my_path)
        random.seed(seed)
        self.seed = seed

    def build_hyper_paramers(self)->dict:
        num_layers = np.random.randint(1, 4)
        epsMax = np.random.uniform(0, 1)
        num_episodes = np.random.randint(100, 200000)
        betaMin=np.random.uniform(0.0001, 0.5)
        max_steps = 300
        alpha = np.random.uniform(0.0001, 0.20)
        gamma = np.random.uniform(0.5, 1)
        buffer_size = np.random.randint(200, 2000)
        batch_size = np.random.randint(16, 128)
        hpdict = {
            "num_episodes" : num_episodes,
            "max_steps" : max_steps,
            "num_layers" : num_layers,
            "layer_sizes" : np.random.randint(10, 2048, size=num_layers),
            "layer_types" : np.random.choice(["NL", "L"], size=num_layers),
            "model" : np.random.choice([DQN2, DuelingDQN2]),
            "replay_buffer" : np.random.choice([ReplayBuffer, PriorityReplayBuffer]),
            "agent" : np.random.choice([FixedTargetDQNAgent, DoubleDQNAgent]),
            "buffer_size" : np.random.randint(1000, 4000),
            "batch_size" : np.random.randint(16, 256),
            "update_every" : np.random.randint(2, 10),
            "update_target_every" : np.random.randint(10, 200),
            "scores_window_len" : 300,
            "good_score_threshold" : 13,
            "alpha" : alpha,
            "gamma" : gamma,
            "uniformity" : np.random.uniform(0, 1),
            "epsMax" :  epsMax,
            "epsMin" : np.random.uniform(0, epsMax),
            "epsDecay" : -1 / (np.random.uniform(0.01, 1) * num_episodes),
            "epsProvider" : np.random.choice([hpp.ExponentialChangeParameterProvider, hpp.LinearChangeParameterProvider, hpp.ConstantParameterProvider]),
            "eps" : None,
            "betaMin" : betaMin,
            "betaMax" : 1,
            "betaGrowth" : 1 / (np.random.uniform(0.1, 1) * num_episodes * max_steps),
            "betaProvider": np.random.choice([hpp.LinearChangeParameterProvider, hpp.ExponentialChangeParameterProvider, hpp.ConstantParameterProvider]),
            "beta" : None,
            "lossFn" : np.random.choice([F.mse_loss, F.l1_loss, F.smooth_l1_loss]),
            "optimizer" : np.random.choice([partial(optim.Adam,lr=alpha), partial(optim.RMSprop,lr=alpha)])
        }

        if hpdict['epsProvider'] == hpp.ConstantParameterProvider :
            hpdict['eps'] = np.random.uniform(hpdict['epsMin'], hpdict['epsMax'])
        if hpdict['betaProvider'] == hpp.ConstantParameterProvider:
            hpdict['beta'] = np.random.uniform(hpdict['betaMin'], hpdict['betaMax'])
        print(hpdict)
        return hpdict

    def train(self, max_episodes : int = None, from_path:str=None)->Tuple[List[float], datetime, datetime]:
        if from_path is not None:
            hpdict = pickle.load(open(from_path, "rb"))
        else:
            hpdict = self.build_hyper_paramers()
        if max_episodes is not None:
            hpdict["max_episodes"] = max_episodes
        hpdict["seed"] = self.seed
        hyper_param_file = os.path.join(self.my_path, "hyper_parameters.pickle")
        pickle.dump(hpdict, open(hyper_param_file, "wb"))
        alphaProvider = hpp.ConstantParameterProvider(hpdict["alpha"])
        gammaProvider = hpp.ConstantParameterProvider(hpdict["gamma"])
        uniformityProvider = hpp.ConstantParameterProvider(hpdict["uniformity"])
        if hpdict["eps"] is None:
            provider = hpdict["epsProvider"]
            epsProvider = provider(hpdict["epsMax"], hpdict["epsMin"], hpdict["epsDecay"])
        else:
            epsProvider = hpp.ConstantParameterProvider(hpdict["eps"])
        if hpdict["beta"] is None:
            provider = hpdict["betaProvider"]
            betaProvider = provider(hpdict["betaMin"], hpdict["betaMax"], hpdict["betaGrowth"])
        else:
            betaProvider = hpp.ConstantParameterProvider(hpdict["beta"])
        arch = []
        for i in range(hpdict["num_layers"]):
            arch.append((hpdict["layer_types"][i], hpdict["layer_sizes"][i]))
            if i < hpdict["num_layers"] - 1:
                arch.append(("R", hpdict["layer_sizes"][i]))
        if hpdict["replay_buffer"] == PriorityReplayBuffer:
            replayBuffer = PriorityReplayBuffer(buffer_size=hpdict["buffer_size"], batch_size=hpdict["batch_size"],
                                                seed=hpdict["seed"], betaProvider=betaProvider, uniformity=uniformityProvider.get())
        else:
            replayBuffer = ReplayBuffer(buffer_size=hpdict["buffer_size"], batch_size=hpdict["batch_size"], seed=hpdict["seed"])

        env = UnityEnvironment(file_name="Banana.app", no_graphics=True)
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]
        state_size =  len(env_info.vector_observations[0])
        action_size = brain.vector_action_space_size
        print(hpdict["optimizer"], hpdict["lossFn"])
        if hpdict["agent"] == FixedTargetDQNAgent:
            agent = FixedTargetDQNAgent(state_size, action_size, epsProvider, gammaProvider, replayBuffer,
                                        hpdict["model"], hpdict["optimizer"], hpdict["lossFn"], 
                                        hpdict["update_every"], hpdict["update_target_every"], arch)
        else:
            agent = DoubleDQNAgent(state_size, action_size, epsProvider, gammaProvider, replayBuffer,
                                        hpdict["model"], hpdict["optimizer"], hpdict["lossFn"], 
                                        hpdict["update_every"], hpdict["update_target_every"], arch)
        
        scores = []
        best_scores =  None
        scores_window = deque(maxlen=hpdict["scores_window_len"])
        last_best_score = -np.inf
        now = datetime.now()
        for episode in range(hpdict["num_episodes"]):
            if (episode -1) % 100 == 0:
                print("Starting Episode number:", episode)
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            agent.prepare_for_new_episode()
            for t in range(hpdict["max_steps"]):
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
            current_score = np.mean(scores_window)
            if current_score > last_best_score:
                torch.save(agent.tmodel.state_dict(), os.path.join(self.my_path,"checkpoint.pth"))
                last_best_score = score
                best_scores = scores_window.copy()
                pickle.dump(scores, open(os.path.join(self.my_path, "best_scores.pickle"), "wb"))
                if current_score > hpdict["good_score_threshold"]:
                    print("solved environmnet in ", episode, " episodes with avg score of ", np.mean(scores_window))
                    break
        completed = datetime.now()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(best_scores)), best_scores)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()
        plt.savefig(os.path.join(self.my_path, "best_scores_plot.jpg"))
        completion_dict = {
            "completed" : completed,
            "started" : now
        }
        pickle.dump(completion_dict, open(os.path.join(self.my_path, "completion_dict.pickle")))
        return (best_scores, completed, now)


available_time = 48*3600
epoch = datetime.utcfromtimestamp(0)
OUTPUT_DIR="/Users/rkhona/learn/udacity/rl/projects2/outputs/navigation_project"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
while available_time > 0:
    i = (datetime.now() - epoch).total_seconds()
    trainer = Trainer(f"iter-{i}", OUTPUT_DIR, seed=i)
    scores, completed, start = trainer.train()
    print("Competed iteration in ", (completed - start).total_seconds()/3600, " hours")
    available_time -= (completed - start).total_seconds()