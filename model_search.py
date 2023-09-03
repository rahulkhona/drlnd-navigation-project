from trainer import Trainer
from datetime import datetime
from filenames import HYPER_PARAMETERS_FILE, CHECKPOINT_FILE, SCORES_FILE
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


### main script to drive training.

def find_best_model(outputdir:str):
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
        scores = pickle.load(open(scores_path, "rb"))
        mean_score = np.mean(scores)
        if mean_score > best:
            best = mean_score
            best_scores = scores
            best_path = path
    return best_path, best_scores, best


def train(outputdir, max_training_hours=48, max_episodes=-1):
    available_time = max_training_hours*3600
    epoch = datetime.utcfromtimestamp(0)
    OUTPUT_DIR="/Users/rkhona/learn/udacity/rl/projects2/outputs/navigation_project"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    while available_time > 0:
        i = (datetime.now() - epoch).total_seconds()
        trainer = Trainer(f"iter-{i}", OUTPUT_DIR, seed=i)
        if max_episodes == -1:
            scores, completed, start = trainer.train()
        else:
            scores, completed, start = trainer.train(max_episodes=max_episodes)
        print("Competed iteration in ", (completed - start).total_seconds()/3600, " hours")
        available_time -= (completed - start).total_seconds()

def test_args(dir, train):
    print(dir, train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and find the best model")
    parser.add_argument("results_dir", help="Directory which contains or will contain training outputs", type=str)
    parser.add_argument("--train", help="train new models before finding the best one", action='store_true')
    parser.add_argument("--max_training_hours", help="Maximum hours training session should run", type=int, default=48)
    parser.add_argument("--max_episodes", help="maximum episodes to run", type=int, default=-1)
    args = parser.parse_args()
    outputdir = args.results_dir
    do_train = args.train
    hours=args.max_training_hours
    max_episodes = args.max_episodes
    if do_train:
        train(outputdir, hours, max_episodes)
    path, scores, best = find_best_model(outputdir)
    print("best model path", path, " best mean score", best, "best scores ", scores)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()

