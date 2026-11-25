"""Policy gradient methods (REINFORCE, A2C, PPO) on KigaliFreightEnv with TorchRL.

Run all:     uv run python -m training.pg_training
Run one:     uv run python -m training.pg_training ppo_01_baseline
"""
import sys
import time

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from training.common import append_result, evaluate, make_env, make_net, save_curve

FRAMES = 500_000

REINFORCE = [
    dict(name="reinforce_01_baseline", lr=1e-3, gamma=0.99, entropy=0.01, batch=2000),
    dict(name="reinforce_02_low_lr", lr=1e-4, gamma=0.99, entropy=0.01, batch=2000),
    dict(name="reinforce_03_high_lr", lr=5e-3, gamma=0.99, entropy=0.01, batch=2000),
    dict(name="reinforce_04_low_gamma", lr=1e-3, gamma=0.90, entropy=0.01, batch=2000),
    dict(name="reinforce_05_mid_gamma", lr=1e-3, gamma=0.95, entropy=0.01, batch=2000),
    dict(name="reinforce_06_no_entropy", lr=1e-3, gamma=0.99, entropy=0.0, batch=2000),
    dict(name="reinforce_07_high_entropy", lr=1e-3, gamma=0.99, entropy=0.05, batch=2000),
    dict(name="reinforce_08_small_batch", lr=1e-3, gamma=0.99, entropy=0.01, batch=500),
    dict(name="reinforce_09_big_batch", lr=1e-3, gamma=0.99, entropy=0.01, batch=5000),
    dict(name="reinforce_10_big_batch_high_lr", lr=3e-3, gamma=0.99, entropy=0.01, batch=5000),
]

A2C = [
    dict(name="a2c_01_baseline", lr=1e-3, gamma=0.99, lmbda=0.95, entropy=0.01, critic=0.5, batch=200),
    dict(name="a2c_02_low_lr", lr=1e-4, gamma=0.99, lmbda=0.95, entropy=0.01, critic=0.5, batch=200),
    dict(name="a2c_03_high_lr", lr=5e-3, gamma=0.99, lmbda=0.95, entropy=0.01, critic=0.5, batch=200),
    dict(name="a2c_04_low_gamma", lr=1e-3, gamma=0.90, lmbda=0.95, entropy=0.01, critic=0.5, batch=200),
    dict(name="a2c_05_td_advantage", lr=1e-3, gamma=0.99, lmbda=0.0, entropy=0.01, critic=0.5, batch=200),
    dict(name="a2c_06_mc_advantage", lr=1e-3, gamma=0.99, lmbda=1.0, entropy=0.01, critic=0.5, batch=200),
    dict(name="a2c_07_no_entropy", lr=1e-3, gamma=0.99, lmbda=0.95, entropy=0.0, critic=0.5, batch=200),
    dict(name="a2c_08_high_entropy", lr=1e-3, gamma=0.99, lmbda=0.95, entropy=0.05, critic=0.5, batch=200),
    dict(name="a2c_09_strong_critic", lr=1e-3, gamma=0.99, lmbda=0.95, entropy=0.01, critic=1.0, batch=200),
    dict(name="a2c_10_long_rollout", lr=1e-3, gamma=0.99, lmbda=0.95, entropy=0.01, critic=0.5, batch=1000),
]

PPO = [
    dict(name="ppo_01_baseline", lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.01, clip=0.2, epochs=4, batch=2000),
    dict(name="ppo_02_low_lr", lr=1e-4, gamma=0.99, lmbda=0.95, entropy=0.01, clip=0.2, epochs=4, batch=2000),
    dict(name="ppo_03_high_lr", lr=1e-3, gamma=0.99, lmbda=0.95, entropy=0.01, clip=0.2, epochs=4, batch=2000),
    dict(name="ppo_04_low_gamma", lr=3e-4, gamma=0.90, lmbda=0.95, entropy=0.01, clip=0.2, epochs=4, batch=2000),
    dict(name="ppo_05_small_clip", lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.01, clip=0.1, epochs=4, batch=2000),
    dict(name="ppo_06_big_clip", lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.01, clip=0.4, epochs=4, batch=2000),
    dict(name="ppo_07_no_entropy", lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.0, clip=0.2, epochs=4, batch=2000),
    dict(name="ppo_08_high_entropy", lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.05, clip=0.2, epochs=4, batch=2000),
    dict(name="ppo_09_more_epochs", lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.01, clip=0.2, epochs=10, batch=2000),
    dict(name="ppo_10_small_rollout", lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.01, clip=0.2, epochs=4, batch=500),
]


def make_actor_critic(env):
    net = make_net()
    actor = ProbabilisticActor(
        TensorDictModule(net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], distribution_class=OneHotCategorical,
        spec=env.action_spec, return_log_prob=True,
    )
    critic = ValueOperator(make_value_net(), in_keys=["observation"])
    return net, actor, critic


def make_value_net():
    net = make_net()
    net[-1] = torch.nn.Linear(net[-1].in_features, 1)  # value head
    return net


def finished_rewards(data):
    done = data["next", "done"].squeeze(-1)
    return data["next", "episode_reward"][done].flatten().tolist()


def train(cfg, frames=FRAMES, seed=0):
    algo = cfg["name"].split("_")[0]
    torch.manual_seed(seed)
    env = make_env(seed)
    net, actor, critic = make_actor_critic(env)
    collector = Collector(env, actor, frames_per_batch=cfg["batch"], total_frames=frames)

    if algo == "ppo":
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip"], entropy_coeff=cfg["entropy"], critic_coeff=0.5)
    elif algo == "a2c":
        loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg["entropy"], critic_coeff=cfg["critic"])
    if algo != "reinforce":
        advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=True)
        optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    else:
        optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    curve, recent, frames_done, start = [], [], 0, time.time()
    for data in collector:
        frames_done += data.numel()
        recent += finished_rewards(data)

        if algo == "reinforce":
            # reinforce: MC returns
            returns = reward2go(data["next", "reward"], data["next", "done"], gamma=cfg["gamma"])
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            dist = actor.get_dist(data)
            log_prob = dist.log_prob(data["action"])
            entropy = dist.entropy().mean()
            loss = -(log_prob * returns.squeeze(-1)).mean() - cfg["entropy"] * entropy
            optim.zero_grad()
            loss.backward()
            optim.step()
            critic_loss = 0.0
        else:
            epochs = cfg.get("epochs", 1)
            for _ in range(epochs):
                with torch.no_grad():
                    advantage(data)
                batches = data.reshape(-1).split(500) if algo == "ppo" else [data.reshape(-1)]
                for mb in batches:
                    out = loss_fn(mb)
                    loss = sum(v for k, v in out.items() if k.startswith("loss_"))  # no loss_entropy if coef is 0
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0)
                    optim.step()
            entropy = out["entropy"] if "entropy" in out.keys() else actor.get_dist(mb).entropy().mean()
            critic_loss = out["loss_critic"].item()

        if recent:
            curve.append(dict(frames=frames_done, episodes=len(recent), episode_reward=sum(recent[-20:]) / len(recent[-20:]),
                              loss=loss.item(), entropy=float(entropy.detach()), critic_loss=critic_loss))
    collector.shutdown()

    torch.save(net.state_dict(), f"models/pg/{cfg['name']}.pt")
    save_curve(f"logs/curves/{cfg['name']}.csv", curve)
    result = dict(algo=algo.upper(), run=cfg["name"])
    result.update(evaluate(net, sample=True))
    result.update({f"hard_{k}": v for k, v in evaluate(net, hard=True, sample=True).items()})
    result["minutes"] = round((time.time() - start) / 60, 1)
    append_result(dict(algo=result["algo"], run=result["run"], eval_reward=result["eval_reward"],
                       on_time_rate=result["on_time_rate"], hard_eval_reward=result["hard_eval_reward"],
                       hard_on_time_rate=result["hard_on_time_rate"], minutes=result["minutes"]))
    print(result, flush=True)
    return result


ALL = {c["name"]: c for c in REINFORCE + A2C + PPO}

if __name__ == "__main__":
    for name in sys.argv[1:] or ALL:
        train(ALL[name])
