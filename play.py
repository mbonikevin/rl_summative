"""Watch a trained agent drive the Kigali freight truck.

uv run play.py                     best agent overall
uv run play.py --algo ppo          best PPO run
uv run play.py --hard --episodes 5 harder unseen setting (rain + 3 tight deadlines)
"""
import argparse
import csv

import torch

from environment.custom_env import ACTION_NAMES, NAMES, KigaliFreightEnv
from training.common import make_net, pick_action


def best_run(algo=None, path="logs/results.csv"):
    """Pick the run with the highest evaluation reward from the sweep results."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if algo is None or r["algo"].lower() == algo]
    return max(rows, key=lambda r: float(r["eval_reward"]))


def load_agent(algo=None):
    row = best_run(algo)
    folder = "dqn" if row["algo"] == "DQN" else "pg"
    net = make_net()
    net.load_state_dict(torch.load(f"models/{folder}/{row['run']}.pt"))
    net.eval()
    return net, row


def recommend(net, obs, sample):
    """Simple API: observation (21 numbers) in, action name out."""
    return ACTION_NAMES[pick_action(net, obs, sample)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--seed", type=int, default=39)  # rush-hour start, good demo
    args = parser.parse_args()

    net, row = load_agent(args.algo)
    sample = row["algo"] != "DQN"
    torch.manual_seed(args.seed)
    print(f"Loaded {row['run']} ({row['algo']}), eval reward {float(row['eval_reward']):.1f}, "
          f"on-time rate {float(row['on_time_rate']):.0%}")

    env = KigaliFreightEnv(render_mode="human", hard=args.hard)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        jobs = ", ".join(f"{NAMES[c]} by {d // 60 % 24:02d}:{d % 60:02d}" for c, d in env.deadlines.items())
        print(f"\n=== Episode {ep + 1}: start {env.clock // 60:02d}:{env.clock % 60:02d}, "
              f"{'raining' if env.raining else 'dry'}, deliveries: {jobs}")
        env.render()
        done = False
        while not done:
            action = recommend(net, obs, sample)
            obs, reward, terminated, truncated, info = env.step(ACTION_NAMES.index(action))
            h, m = divmod(env.clock % 1440, 60)
            print(f"step {env.steps:3d} {h:02d}:{m:02d}  action {action:5s}  pos {env.pos}  "
                  f"reward {reward:+6.1f}  total {env.total_reward:+7.1f}  left {info['pending']}")
            done = terminated or truncated
        result = "all delivered" if terminated else "shift over"
        print(f"--- {result}: on time {info['on_time']}, late {info['late']}, total reward {env.total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
