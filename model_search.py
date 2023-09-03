from trainer import Trainer
from datetime import datetime
from filenames import HYPER_PARAMETERS_FILE, CHECKPOINT_FILE, SCORES_FILE
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Tuple, List

### main script to drive training.

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
        print(scores)
        all_scores = scores_dict['scores_till_now']
        mean_score = np.mean(scores)
        if mean_score > best:
            best = mean_score
            best_scores = scores
            best_path = path
    return best_path, best_scores, best, all_scores


def train(outputdir, max_training_hours=48, max_episodes:int=-1, max_steps:int=None):
    """ Training loop that generates different models and hyper parameters and trains
    them

    Parameters:
        outputir (str) : path were outputs should write model and hyper paramaters
        max_training_hours (int) : maximum hours the training loop should be allowed default 48 hrs
        max_episodes  (int) : Maximum number of episodes to run, default is random choice
        max_steps  (int) : Maximum number of steps per episodes to run, default is defined by trainer script

    """
    available_time = max_training_hours*3600
    epoch = datetime.utcfromtimestamp(0)
    OUTPUT_DIR="/Users/rkhona/learn/udacity/rl/projects2/outputs/navigation_project"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    while available_time > 0:
        i = (datetime.now() - epoch).total_seconds()
        trainer = Trainer(f"iter-{i}", OUTPUT_DIR, seed=i)
        scores, completed, start = trainer.train(num_episodes=max_episodes, avail_time=available_time, max_steps=max_steps)
        print("Competed iteration in ", (completed - start).total_seconds()/3600, " hours")
        available_time -= (completed - start).total_seconds()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and find the best model")
    parser.add_argument("results_dir", help="Directory which contains or will contain training outputs", type=str)
    parser.add_argument("--train", help="train new models before finding the best one", action='store_true')
    parser.add_argument("--max_training_hours", help="Maximum hours training session should run", type=int, default=48)
    parser.add_argument("--max_episodes", help="maximum episodes to run", type=int, default=None)
    parser.add_argument("--max_steps", help="maximum steps per episodes to run", type=int, default=None)
    args = parser.parse_args()
    outputdir = args.results_dir
    do_train = args.train
    hours=args.max_training_hours
    max_episodes = args.max_episodes
    max_steps = args.max_steps
    if do_train:
        train(outputdir, hours, max_episodes, max_steps)
    path, best_scores, best, all_scores = find_best_model(outputdir)
    print("best model path", path, " best mean score", best, "best scores ", best_scores)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(all_scores)), all_scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()