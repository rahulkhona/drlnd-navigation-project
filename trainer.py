import torch
import torch.nn.functional as F
import torch.optim as optim
import hyper_parameter_providers as hpp
from models import DuelingDQN2, DQN2
from agents import FixedTargetDQNAgent, DoubleDQNAgent
from replay_buffer import ReplayBuffer, PriorityReplayBuffer
from functools import partial
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random
import json
from typing import List, Tuple
from filenames import CHECKPOINT_FILE, SCORES_FILE, HYPER_PARAMETERS_FILE, COMPLETION_TIME_FILE, SCORES_IMAGE, ALL_SCORES_FILE
from string_to_function_mappers import hpp_mapper, loss_fn_mapper, optim_mapper, replay_buffer_mapper, agent_mapper, model_mapper
from hyper_parameter_providers import ConstantParameterProvider, LinearChangeParameterProvider, ExponentialChangeParameterProvider
from numpyencoder import NumpyEncoder


class Trainer():
    """Train a model. It randomly choses hyper parameters and it is designed to driven multiple times from
    an execution environment to try different parameters and generate different models
    """
    def __init__(self, iteration_name : str, output_path : str, seed:int=0x21):
        """Constructor

        Parameters:
            iteration_name (str) : Name of the iteration. All the output would be stored in the directory whose name is 
                same as iteration name
            output_path (str) : Top level directory where all the iterations are saved
            seed (int) : seed to initialize random number generator with
        """
        self.name = iteration_name
        self.output_path = output_path

        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        self.my_path = os.path.join(self.output_path, self.name)
        os.makedirs(self.my_path)
        random.seed(seed)
        self.seed = seed

    def build_hyper_paramers(self)->dict:
        """Generate hyper parameters
        """
        num_layers = np.random.randint(1, 4)
        epsMax = np.random.uniform(0, 1)
        num_episodes = np.random.randint(2000, 4000)
        betaMin=np.random.uniform(0.0001, 0.5)
        max_steps = 300
        alpha = np.random.uniform(0.0001, 0.20)
        gamma = np.random.uniform(0.5, 1)
        tau = np.random.uniform(.001, .05)
        hpdict = {
            "num_episodes" : num_episodes,
            "max_steps" : max_steps,
            "num_layers" : num_layers,
            "layer_sizes" : np.random.randint(10, 2048, size=num_layers).tolist(),
            "layer_types" : np.random.choice(["NL", "L"], size=num_layers).tolist(),
            "model" : np.random.choice([DQN2.__name__, DuelingDQN2.__name__]).tolist(),
            "replay_buffer" : np.random.choice([ReplayBuffer.__name__, PriorityReplayBuffer.__name__]).tolist(),
            "agent" : np.random.choice([FixedTargetDQNAgent.__name__, DoubleDQNAgent.__name__]).tolist(),
            "buffer_size" : np.random.randint(1000, 4000),
            "batch_size" : np.random.randint(16, 256),
            "update_every" : np.random.randint(2, 10),
            "update_target_every" : np.random.randint(10, 200),
            "scores_window_len" : 100,
            "good_score_threshold" : 13,
            "alpha" : alpha,
            "gamma" : gamma,
            "tau" : tau,
            "useTau": np.random.choice([0, 1]),
            "uniformity" : np.random.uniform(0, 1),
            "epsMax" :  epsMax,
            "epsMin" : np.random.uniform(0, epsMax),
            "epsDecay" : -1 / (np.random.uniform(0.01, 1) * num_episodes),
            "epsProvider" : np.random.choice([ExponentialChangeParameterProvider.__name__,
                                              LinearChangeParameterProvider.__name__,
                                              ConstantParameterProvider.__name__]).tolist(),
            "eps" : None,
            "betaMin" : betaMin,
            "betaMax" : 1,
            "betaGrowth" : 1 / (np.random.uniform(0.1, 1) * num_episodes * max_steps),
            "betaProvider": np.random.choice([LinearChangeParameterProvider.__name__,
                                              ExponentialChangeParameterProvider.__name__,
                                              ConstantParameterProvider.__name__]).tolist(),
            "beta" : None,
            "lossFn" : np.random.choice([F.mse_loss.__name__, F.l1_loss.__name__, F.smooth_l1_loss.__name__]).tolist(),
            "optimizer" : np.random.choice([optim.Adam.__name__, optim.RMSprop.__name__]).tolist()
        }

        if hpdict['epsProvider'] == 'ConstantParameterProvider':
            hpdict['eps'] = np.random.uniform(hpdict['epsMin'], hpdict['epsMax'])
        if hpdict['betaProvider'] == 'ConstantParameterProvider':
            hpdict['beta'] = np.random.uniform(hpdict['betaMin'], hpdict['betaMax'])
        print(hpdict)
        return hpdict

    def load_json(self, path:str):
        with open(path, "r") as jsonFile:
            return json.load(jsonFile)

    def save_json(self, arg, path:str):
        with open(path, "w") as jsonFile:
            json.dump(arg, jsonFile, cls=NumpyEncoder)

    def train(self, env, brain_name, state_size, action_size, num_episodes : int = None,
              from_path:str=None, avail_time:int=None, max_steps:int=None, hyp_params:dict=None)->Tuple[List[float], datetime, datetime]:
        """Train a model

        Note: Env details are passed in from the caller as env environment cannot be recreated after it is closed.

        Parameters:
            env : UnityEnvironment env
            brain_name : brain name returned by unity env
            state_size (int) : state _size returned by env reset that the caller used to determine our state size
            action_size (int) : state _size returned by env reset that the caller used to determine our action size
            max_episodes (int) : default is pick generate value, but override with this
            from_path (str) : default is None. Use hyper parameters from json file saved in this parameter rather than generating
        Returns:
            a tuple containing list of best cores, completion time and start time
        """
        if hyp_params is not None:
            hpdict = hyp_params
        elif from_path is not None:
            hpdict = self.load_json(from_path)
        else:
            hpdict = self.build_hyper_paramers()
        if num_episodes is not None:
            hpdict["num_episodes"] = num_episodes
        if max_steps is not None:
            hpdict['max_steps'] = max_steps
        hpdict["seed"] = self.seed
        alphaProvider = hpp.ConstantParameterProvider(hpdict["alpha"])
        gammaProvider = hpp.ConstantParameterProvider(hpdict["gamma"])
        uniformityProvider = hpp.ConstantParameterProvider(hpdict["uniformity"])
        tauProvider = hpp.ConstantParameterProvider(hpdict["tau"])
        if hpdict["eps"] is None:
            provider = hpp_mapper[hpdict["epsProvider"]]
            epsProvider = provider(hpdict["epsMax"], hpdict["epsMin"], hpdict["epsDecay"])
        else:
            epsProvider = hpp.ConstantParameterProvider(hpdict["eps"])
        if hpdict["beta"] is None:
            provider = hpp_mapper[hpdict["betaProvider"]]
            betaProvider = provider(hpdict["betaMin"], hpdict["betaMax"], hpdict["betaGrowth"])
        else:
            betaProvider = hpp.ConstantParameterProvider(hpdict["beta"])
        arch = []
        for i in range(hpdict["num_layers"]):
            arch.append((hpdict["layer_types"][i], hpdict["layer_sizes"][i]))
            if i < hpdict["num_layers"] - 1:
                arch.append(("R", hpdict["layer_sizes"][i]))
        if hpdict["replay_buffer"] == PriorityReplayBuffer.__name__:
            replayBuffer = PriorityReplayBuffer(buffer_size=hpdict["buffer_size"], batch_size=hpdict["batch_size"],
                                                seed=hpdict["seed"], betaProvider=betaProvider, uniformity=uniformityProvider.get())
        else:
            replayBuffer = ReplayBuffer(buffer_size=hpdict["buffer_size"], batch_size=hpdict["batch_size"], seed=hpdict["seed"])

        hpdict['state_size'] = state_size
        hpdict['action_size'] = action_size
        hyper_param_file = os.path.join(self.my_path, HYPER_PARAMETERS_FILE)
        self.save_json(hpdict, hyper_param_file)
        optimizer = partial(optim_mapper[hpdict["optimizer"]], lr=hpdict['alpha'])
        agentClass = agent_mapper[hpdict['agent']]
        agent = agentClass(state_size, action_size, epsProvider, gammaProvider, replayBuffer,
                           model_mapper[hpdict["model"]], optimizer, loss_fn_mapper[hpdict["lossFn"]], 
                           hpdict["update_every"], hpdict["update_target_every"], arch, hpdict['useTau'], hpdict['tau'])
        
        scores = []
        best_scores =  None
        scores_window = deque(maxlen=hpdict["scores_window_len"])
        last_best_score = -np.inf
        started = datetime.now()
        for episode in range(1, hpdict["num_episodes"] + 1):
            if (episode - 1) % 100 == 0:
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
            if len(scores_window) >= hpdict['scores_window_len'] and current_score > last_best_score:
                torch.save(agent.lmodel.state_dict(), os.path.join(self.my_path, CHECKPOINT_FILE))
                last_best_score = score
                best_scores = scores_window.copy()
                print("New best score ", current_score)
                scores_dict = {
                    "best_scores" : list(best_scores),
                    "scores_till_now" : scores
                }
                self.save_json(scores_dict, os.path.join(self.my_path, SCORES_FILE))
                if current_score > hpdict["good_score_threshold"]:
                    print("solved environmnet in ", episode, " episodes with avg score of ", np.mean(scores_window))
                    break
            if avail_time is not None and (datetime.now() - started).total_seconds() > avail_time:
                break
        completed = datetime.now()
        self.save_json(scores, os.path.join(self.my_path, ALL_SCORES_FILE))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(best_scores)), best_scores)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        #plt.show()
        plt.savefig(os.path.join(self.my_path, SCORES_IMAGE))
        completion_dict = {
            "time_taken" : (completed - started).total_seconds(),
        }
        self.save_json(completion_dict, os.path.join(self.my_path, COMPLETION_TIME_FILE))
        return (best_scores, completed, started)
