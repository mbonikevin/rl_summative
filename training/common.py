"""Shared helpers: building the TorchRL env, the network, evaluation and logging."""
import csv
import os

import numpy as np
import torch
from torch import nn
from torchrl.envs import Compose, GymEnv, RewardSum, StepCounter, TransformedEnv

import environment.custom_env  # noqa: F401
from environment.custom_env import KigaliFreightEnv

OBS_SIZE = 21
N_ACTIONS = 5


def make_env(seed=0):
    env = TransformedEnv(GymEnv("KigaliFreight-v0"), Compose(StepCounter(), RewardSum()))
    env.set_seed(seed)
    return env


def make_net(hidden=128):
    """Same network shape for every algorithm: observation -> one number per action."""
    return nn.Sequential(
        nn.Linear(OBS_SIZE, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, N_ACTIONS),
    )


def pick_action(net, obs, sample=False):
    """DQN acts greedily (best Q-value). Policy-gradient agents sample from their action probabilities."""
    with torch.no_grad():
        out = net(torch.as_tensor(obs, dtype=torch.float32))
    if sample:
        return int(torch.distributions.Categorical(logits=out).sample())
    return int(out.argmax())


def evaluate(net, episodes=30, hard=False, seed=10_000, sample=False):
    """Run the trained policy on unseen seeds. Returns mean reward and on-time rate."""
    torch.manual_seed(seed)
    env = KigaliFreightEnv(hard=hard)
    rewards, on_time, jobs, lengths = [], 0, 0, []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        n_jobs = len(env.pending)
        while not done:
            obs, r, term, trunc, info = env.step(pick_action(net, obs, sample))
            total += r
            done = term or trunc
        rewards.append(total)
        lengths.append(env.steps)
        on_time += info["on_time"]
        jobs += n_jobs
    return {
        "eval_reward": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "on_time_rate": on_time / jobs,
        "eval_length": float(np.mean(lengths)),
    }


def save_curve(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def append_result(row, path="logs/results.csv"):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            writer.writeheader()
        writer.writerow(row)
