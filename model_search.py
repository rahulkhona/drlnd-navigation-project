from trainer import Trainer
from datetime import datetime
from filenames import HYPER_PARAMETERS_FILE, CHECKPOINT_FILE, SCORES_FILE, ALL_SCORES_FILE
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Tuple, List
import time
import multiprocessing as mp
from unityagents import UnityEnvironment

### main script to drive training.

def sort_models_by_performance(outputdir:str)->Tuple[List[str], List[float]]:
    entries = os.listdir(outputdir)
    avg_scores = []
    paths = []
    for entry in entries:
        path = os.path.join(outputdir, entry)
        if not os.path.isdir(path):
            continue
        scores_path = os.path.join(path, SCORES_FILE)
        if not os.path.isfile(scores_path):
            continue
        scores_dict = json.load(open(scores_path, "r"))
        scores = scores_dict['best_scores']
        all_scores = scores_dict['scores_till_now']
        mean_score = np.mean(scores)
        avg_scores.append(mean_score)
        paths.append(path)
    
    indices = np.argsort(np.array(avg_scores))
    sorted_scores = np.array(avg_scores)[indices]
    sorted_paths = np.array(paths)[indices]
    return (sorted_paths[::-1], sorted_scores[::-1])


def find_best_model(outputdir:str)->Tuple[str, List[float], float, List[float]]:
    """Go through output dir and find the best model and its hyper parameters 

    Returns:
        Tuple containing path to best model, its best 100 consecutive scores, mean of best scores, all scores till
        the capture of the output

    """
    entries = os.listdir(outputdir)
    best = -np.inf
    best_scores = None
    best_path = None
    for entry in entries:
        path = os.path.join(outputdir, entry)
        if not os.path.isdir(path):
            continue
        scores_path = os.path.join(path, SCORES_FILE)
        if not os.path.isfile(scores_path):
            continue
        print(" checking ", scores_path)
        scores_dict = json.load(open(scores_path, "r"))
        scores = scores_dict['best_scores']
        #print(scores)
        all_scores = scores_dict['scores_till_now']
        mean_score = np.mean(scores)
        if mean_score > best:
            best = mean_score
            best_scores = scores
            best_path = path
    return best_path, best_scores, best, all_scores


def plot_model_scores(path:str):
    all_scores = json.load(open(os.path.join(path, ALL_SCORES_FILE)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(all_scores)), all_scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


def train(outputdir:str=None, max_training_hours=48, max_episodes:int=-1, max_steps:int=None):
    """ Training loop that generates different models and hyper parameters and trains
    them

    Parameters:
        outputir (str) : path were outputs should write model and hyper paramaters
        max_training_hours (int) : maximum hours the training loop should be allowed default 48 hrs
        max_episodes  (int) : Maximum number of episodes to run, default is random choice
        max_steps  (int) : Maximum number of steps per episodes to run, default is defined by trainer script

    """
    env = UnityEnvironment(file_name="Banana.app", no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size =  len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    available_time = max_training_hours*3600
    epoch = datetime.utcfromtimestamp(0)
    OUTPUT_DIR=outputdir if outputdir is not None else "/Users/rkhona/learn/udacity/rl/projects2/outputs/navigation_project"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    while available_time > 0:
        i = (datetime.now() - epoch).total_seconds()
        i = int(i)
        trainer = Trainer(f"iter-{i}", OUTPUT_DIR, seed=i)
        scores, completed, started = trainer.train(env, brain_name, state_size, action_size, num_episodes=max_episodes, avail_time=available_time, max_steps=max_steps)
        print("Competed iteration in ", (completed - started).total_seconds()/3600, " hours")
        available_time -= (completed - started).total_seconds()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and find the best model")
    parser.add_argument("results_dir", help="Directory which contains or will contain training outputs", type=str)
    parser.add_argument("--train", help="train new models before finding the best one", action='store_true')
    parser.add_argument("--max_training_hours", help="Maximum hours training session should run", type=int, default=48)
    parser.add_argument("--max_episodes", help="maximum episodes to run", type=int, default=None)
    parser.add_argument("--max_steps", help="maximum steps per episodes to run", type=int, default=None)
    parser.add_argument("--sorted", help="return list of models sorted by their performance", action="store_true")
    parser.add_argument("--plot", help="scores of a given model", type=str, default=None)

    args = parser.parse_args()
    outputdir = args.results_dir
    do_train = args.train
    hours=args.max_training_hours
    max_episodes = args.max_episodes
    max_steps = args.max_steps
    if do_train:
        train(outputdir, hours, max_episodes, max_steps)
    if args.sorted:
        paths, scores = sort_models_by_performance(outputdir)
        print(list(zip(paths, scores)))
    else:
        path, best_scores, best, all_scores = find_best_model(outputdir)
        print("best model path", path, " best mean score", best, "best scores ", best_scores)
        plot_model_scores(path)
