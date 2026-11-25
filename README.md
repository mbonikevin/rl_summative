# Kigali Freight RL

RL summative project. A truck has to deliver goods around a small grid map of Kigali before the
customer deadlines. Rush hour (7-9 and 17-19) makes the city centre slow and rain slows all roads,
so the agent has to think about time, traffic and weather when picking a route. It's linked to my
capstone (truck ETA prediction in Kigali).

I trained DQN, REINFORCE, A2C and PPO on it using TorchRL.

![environment](assets/env_screenshot.png)

## How to run

```bash
uv sync
uv run main.py
```

This opens the pygame window with the best agent (PPO). Other options:

```bash
uv run play.py --algo dqn    # or reinforce, a2c, ppo
uv run play.py --hard        # harder setting the agent never saw
uv run python -m environment.rendering   # random agent, just to see the env
```

## Training

```bash
uv run python -m training.run_all     # all 41 runs, took me ~25 min
uv run python -m training.make_plots  # plots go to assets/, tables to logs/tables.md
uv run pytest
```

## The environment

- Actions: up, down, left, right, wait (10 min)
- Observation: 21 numbers. truck position, time, rain, rush hour, and for each of the 4 stops
  if it's pending, time left and direction to it
- Reward: -0.1 per minute, +20 on time delivery, +5 late, -1 for hitting a wall, +0.5 for getting
  closer to a stop, -10 for each stop not delivered at the end
- Each episode starts at the depot at a random time between 6:00 and 18:00, with 2 random stops
  and 30% chance of rain
- Ends when everything is delivered or the 4 hour shift (150 steps) runs out

Code is in `environment/` (env + pygame) and `training/` (one file per method type, plus the sweep
and plots). Trained models are in `models/` and results in `logs/`.
