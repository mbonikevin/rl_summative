from collections import deque

from gymnasium.utils.env_checker import check_env

from environment.custom_env import MAP, POSITIONS, SIZE, KigaliFreightEnv


def test_gym_api():
    check_env(KigaliFreightEnv(), skip_render_check=True)


def test_every_stop_reachable_from_depot():
    seen, queue = {POSITIONS["D"]}, deque([POSITIONS["D"]])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and MAP[ny][nx] != "#" and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    assert all(p in seen for p in POSITIONS.values())


def test_deliver_at_stop_gives_reward():
    env = KigaliFreightEnv()
    env.reset(seed=0)
    stop = next(iter(env.pending))
    env.pos = POSITIONS[stop]
    _, reward, _, _, info = env.step(4)
    assert reward > 0 and stop not in env.pending


def test_episode_ends():
    env = KigaliFreightEnv()
    env.reset(seed=1)
    for _ in range(200):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            return
    assert False, "episode never ended"
