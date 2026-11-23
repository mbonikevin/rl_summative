"""KigaliFreightEnv: a truck delivering goods across a simplified Kigali grid.

The truck starts at the depot with a few deliveries, each with a deadline.
Driving through the city centre is slow during rush hour and rain slows
every road, so the agent has to plan routes around time and traffic.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 10x10 map of Kigali (row = y, column = x)
# D depot, N Nyabugogo, R Remera, G Gikondo, K Kicukiro
# B busy city-centre roads, # blocked (hills / buildings), . normal road
MAP = [
    "D...#....",
    ".##.#.##.",
    "....N....",
    ".#..BB.#.",
    ".#.BBB.R.",
    "...BB....",
    ".##..#.#.",
    "G....#...",
    ".#.#...#.",
    "....K....",
]
MAP = [row + "." for row in MAP]  # pad to 10 columns
SIZE = 10

CUSTOMERS = ["N", "R", "G", "K"]  # possible delivery stops
NAMES = {"D": "Depot", "N": "Nyabugogo", "R": "Remera", "G": "Gikondo", "K": "Kicukiro"}

# actions
UP, DOWN, LEFT, RIGHT, DELIVER = 0, 1, 2, 3, 4
MOVES = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
ACTION_NAMES = ["up", "down", "left", "right", "deliver"]

SHIFT_MINUTES = 240  # a 4 hour shift
MAX_STEPS = 150


def find(letter):
    for y, row in enumerate(MAP):
        x = row.find(letter)
        if x != -1:
            return (x, y)


POSITIONS = {c: find(c) for c in ["D"] + CUSTOMERS}


def is_rush_hour(minute_of_day):
    hour = (minute_of_day // 60) % 24
    return 7 <= hour < 9 or 17 <= hour < 19


class KigaliFreightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode=None, hard=False):
        self.render_mode = render_mode
        self.hard = hard  # harder setting, only used to test generalization
        self.action_space = spaces.Discrete(5)
        # [x, y, time of day, raining, rush hour, then (pending, slack) for each customer]
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5 + 2 * len(CUSTOMERS),), dtype=np.float32)
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos = POSITIONS["D"]
        self.clock = int(rng.integers(6 * 60, 18 * 60))  # shift starts between 6:00 and 18:00
        self.start_clock = self.clock
        self.raining = rng.random() < (1.0 if self.hard else 0.3)

        n_jobs = 3 if self.hard else 2
        chosen = rng.choice(CUSTOMERS, size=n_jobs, replace=False)
        low, high = (40, 120) if self.hard else (60, 180)
        self.deadlines = {c: self.clock + int(rng.integers(low, high)) for c in chosen}
        self.pending = set(chosen)
        self.on_time = 0
        self.late = 0
        self.steps = 0
        self.last_reward = 0.0
        self.total_reward = 0.0
        return self._obs(), self._info()

    def _obs(self):
        obs = [
            self.pos[0] / (SIZE - 1),
            self.pos[1] / (SIZE - 1),
            (self.clock % 1440) / 1440,
            float(self.raining),
            float(is_rush_hour(self.clock)),
        ]
        for c in CUSTOMERS:
            if c in self.pending:
                slack = (self.deadlines[c] - self.clock) / SHIFT_MINUTES
                obs += [1.0, float(np.clip(slack, -1, 1))]
            else:
                obs += [0.0, 0.0]
        return np.array(obs, dtype=np.float32)

    def _info(self):
        return {"clock": self.clock, "pending": len(self.pending), "on_time": self.on_time, "late": self.late}

    def travel_minutes(self, cell):
        """Minutes needed to drive into a cell. This is the simple traffic model."""
        x, y = cell
        minutes = 3
        if MAP[y][x] == "B":
            minutes = 12 if is_rush_hour(self.clock) else 5
        if self.raining:
            minutes += 2
        return minutes

    def step(self, action):
        action = int(action)
        self.steps += 1
        reward = 0.0

        if action in MOVES:
            dx, dy = MOVES[action]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and MAP[ny][nx] != "#":
                minutes = self.travel_minutes((nx, ny))
                self.pos = (nx, ny)
            else:
                minutes = 1
                reward -= 1.0  # hit a wall or left the map
        else:  # DELIVER
            here = MAP[self.pos[1]][self.pos[0]]
            if here in self.pending:
                minutes = 5  # unloading
                self.pending.remove(here)
                if self.clock + minutes <= self.deadlines[here]:
                    reward += 20.0
                    self.on_time += 1
                else:
                    reward += 5.0
                    self.late += 1
            else:
                minutes = 1
                reward -= 1.0  # nothing to deliver here

        self.clock += minutes
        reward -= 0.1 * minutes  # every minute on the road costs fuel and time

        # weather can change during the shift
        if not self.hard and self.np_random.random() < 0.05:
            self.raining = not self.raining

        terminated = len(self.pending) == 0
        truncated = False
        if not terminated and (self.clock - self.start_clock >= SHIFT_MINUTES or self.steps >= MAX_STEPS):
            truncated = True
            reward -= 10.0 * len(self.pending)  # deliveries never made

        self.last_reward = reward
        self.total_reward += reward
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        if self.renderer is None:
            from environment.rendering import Renderer
            self.renderer = Renderer(self.render_mode)
        return self.renderer.draw(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


gym.register(id="KigaliFreight-v0", entry_point="environment.custom_env:KigaliFreightEnv")
