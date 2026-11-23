"""DQN on KigaliFreightEnv with TorchRL. Run: uv run python -m training.dqn_training"""
import sys
import time

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

from training.common import append_result, evaluate, make_env, make_net, save_curve

FRAMES = 300_000

# 11 configs (double DQN on by default): each changes the learning rate, discount, replay, batch, exploration or target update
CONFIGS = [
    dict(name="dqn_01_baseline", lr=1e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.01, double=True),
    dict(name="dqn_02_low_lr", lr=1e-4, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.01, double=True),
    dict(name="dqn_03_high_lr", lr=5e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.01, double=True),
    dict(name="dqn_04_low_gamma", lr=1e-3, gamma=0.90, buffer=20_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.01, double=True),
    dict(name="dqn_05_small_buffer", lr=1e-3, gamma=0.99, buffer=2_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.01, double=True),
    dict(name="dqn_06_big_batch", lr=1e-3, gamma=0.99, buffer=20_000, batch=256, eps_end=0.05, eps_frames=20_000, tau=0.01, double=True),
    dict(name="dqn_07_fast_explore", lr=1e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_frames=5_000, tau=0.01, double=True),
    dict(name="dqn_08_long_explore", lr=1e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.20, eps_frames=40_000, tau=0.01, double=True),
    dict(name="dqn_09_slow_target", lr=1e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.001, double=True),
    dict(name="dqn_10_fast_target", lr=1e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.1, double=True),
    dict(name="dqn_11_no_double", lr=1e-3, gamma=0.99, buffer=20_000, batch=64, eps_end=0.05, eps_frames=20_000, tau=0.01, double=False),
]


def train(cfg, frames=FRAMES, seed=0):
    torch.manual_seed(seed)
    env = make_env(seed)
    net = make_net()
    actor = QValueActor(net, in_keys=["observation"], spec=env.action_spec)
    explore = EGreedyModule(env.action_spec, eps_init=1.0, eps_end=cfg["eps_end"], annealing_num_steps=cfg["eps_frames"])
    policy = TensorDictSequential(actor, explore)

    collector = Collector(env, policy, frames_per_batch=100, total_frames=frames, init_random_frames=1_000)
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer"]), batch_size=cfg["batch"])
    loss_fn = DQNLoss(actor, action_space=env.action_spec, delay_value=True, double_dqn=cfg["double"])
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    target = SoftUpdate(loss_fn, tau=cfg["tau"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    curve, recent, frames_done, start = [], [], 0, time.time()
    for data in collector:
        frames_done += data.numel()
        buffer.extend(data.reshape(-1))
        done = data["next", "done"].squeeze(-1)
        recent += data["next", "episode_reward"][done].flatten().tolist()

        if len(buffer) < 1_000:
            continue
        for _ in range(10):  # 10 gradient steps per 100 frames
            loss = loss_fn(buffer.sample())["loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            target.step()
        explore.step(data.numel())

        if frames_done % 2_000 == 0 and recent:
            curve.append(dict(frames=frames_done, episode_reward=sum(recent[-20:]) / len(recent[-20:]),
                              loss=loss.item(), epsilon=explore.eps.item()))
    collector.shutdown()

    torch.save(net.state_dict(), f"models/dqn/{cfg['name']}.pt")
    save_curve(f"logs/curves/{cfg['name']}.csv", curve)
    result = dict(algo="DQN", run=cfg["name"], **{k: v for k, v in cfg.items() if k != "name"})
    result.update(evaluate(net))
    result.update({f"hard_{k}": v for k, v in evaluate(net, hard=True).items()})
    result["minutes"] = round((time.time() - start) / 60, 1)
    append_result(dict(algo=result["algo"], run=result["run"], eval_reward=result["eval_reward"],
                       on_time_rate=result["on_time_rate"], hard_eval_reward=result["hard_eval_reward"],
                       hard_on_time_rate=result["hard_on_time_rate"], minutes=result["minutes"]))
    print(result, flush=True)
    return result


if __name__ == "__main__":
    # optional: pass config numbers, e.g. "python -m training.dqn_training 1 2"
    picks = [int(i) - 1 for i in sys.argv[1:]] or range(len(CONFIGS))
    for i in picks:
        train(CONFIGS[i])
