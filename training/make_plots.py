"""Make the report figures and tables from logs/. Run: uv run python -m training.make_plots"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from training import dqn_training, pg_training

OUT = "assets"
ALGOS = ["DQN", "REINFORCE", "A2C", "PPO"]
COLORS = {"DQN": "#1f77b4", "REINFORCE": "#d62728", "A2C": "#2ca02c", "PPO": "#9467bd"}
os.makedirs(OUT, exist_ok=True)

results = pd.read_csv("logs/results.csv")
best = results.loc[results.groupby("algo")["eval_reward"].idxmax()].set_index("algo")


def curve(run):
    df = pd.read_csv(f"logs/curves/{run}.csv")
    df["smooth"] = df["episode_reward"].rolling(10, min_periods=1).mean()
    return df


# best run of each algo
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
for ax, algo in zip(axes.flat, ALGOS):
    df = curve(best.loc[algo, "run"])
    ax.plot(df["episodes"], df["episode_reward"], color=COLORS[algo], alpha=0.3, lw=0.8)
    ax.plot(df["episodes"], df["smooth"], color=COLORS[algo])
    ax.set_title(f"{algo}: {best.loc[algo, 'run']}")
    ax.grid(alpha=0.3)
fig.supxlabel("training episode")
fig.supylabel("episode reward (mean of last 20)")
fig.tight_layout()
fig.savefig(f"{OUT}/reward_curves.png", dpi=150)

# all runs
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
for ax, algo in zip(axes.flat, ALGOS):
    for run in results[results["algo"] == algo]["run"]:
        df = curve(run)
        ax.plot(df["episodes"], df["smooth"], label=run.split("_", 2)[2], lw=1)
    ax.set_title(algo)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
fig.supxlabel("training episode")
fig.supylabel("mean episode reward")
fig.tight_layout()
fig.savefig(f"{OUT}/sweep_curves.png", dpi=150)

# dqn loss and epsilon
df = pd.read_csv(f"logs/curves/{best.loc['DQN', 'run']}.csv")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["episodes"], df["loss"].rolling(10, min_periods=1).mean(), color=COLORS["DQN"], label="TD loss")
ax.set_xlabel("training episode")
ax.set_ylabel("TD loss (smoothed)")
ax2 = ax.twinx()
ax2.plot(df["episodes"], df["epsilon"], color="gray", ls="--", label="epsilon")
ax2.set_ylabel("epsilon")
ax.set_title(f"DQN objective: {best.loc['DQN', 'run']}")
fig.legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/dqn_loss.png", dpi=150)

# entropy (pg only)
plt.figure(figsize=(8, 4))
for algo in ["REINFORCE", "A2C", "PPO"]:
    df = curve(best.loc[algo, "run"])
    plt.plot(df["episodes"], df["entropy"].rolling(10, min_periods=1).mean(), label=algo, color=COLORS[algo])
plt.axhline(1.609, color="gray", ls=":", label="uniform random (ln 5)")
plt.xlabel("training episode")
plt.ylabel("policy entropy")
plt.title("Policy entropy, best run of each policy-gradient method")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/pg_entropy.png", dpi=150)

# normal vs hard setting
fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(ALGOS))
ax.bar([i - 0.2 for i in x], [best.loc[a, "on_time_rate"] for a in ALGOS], 0.4, label="normal (unseen seeds)")
ax.bar([i + 0.2 for i in x], [best.loc[a, "hard_on_time_rate"] for a in ALGOS], 0.4, label="hard (rain, 3 tight deadlines)")
ax.set_xticks(list(x), ALGOS)
ax.set_ylabel("on-time delivery rate")
ax.set_ylim(0, 1)
ax.set_title("Generalization of the best run of each method")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/generalization.png", dpi=150)


THRESHOLD = 20  # around here most deliveries are on time


def episodes_to(run, threshold=THRESHOLD):
    df = curve(run)
    hit = df[df["smooth"] >= threshold]
    return int(hit["episodes"].iloc[0]) if len(hit) else None


fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, algo in zip(axes, ALGOS):
    run = best.loc[algo, "run"]
    df = curve(run)
    ax.plot(df["episodes"], df["smooth"], color=COLORS[algo])
    ax.axhline(THRESHOLD, color="gray", ls=":")
    ep = episodes_to(run)
    if ep:
        ax.axvline(ep, color="black", ls="--")
        ax.text(ep, ax.get_ylim()[0] + 5, f" {ep} ep", fontsize=9)
    ax.set_title(algo)
    ax.set_xlabel("training episode")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("smoothed episode reward")
fig.tight_layout()
fig.savefig(f"{OUT}/convergence.png", dpi=150)


configs = {c["name"]: c for c in dqn_training.CONFIGS} | pg_training.ALL
lines = []
for algo in ALGOS:
    rows = []
    for _, r in results[results["algo"] == algo].sort_values("run").iterrows():
        cfg = {k: v for k, v in configs[r["run"]].items() if k != "name"}
        rows.append(dict(run=r["run"], **cfg, eval_reward=round(r["eval_reward"], 1),
                         on_time=f"{r['on_time_rate']:.0%}", hard_on_time=f"{r['hard_on_time_rate']:.0%}",
                         episodes_to_converge=episodes_to(r["run"])))
    lines += [f"\n## {algo}\n", pd.DataFrame(rows).to_markdown(index=False)]
summary = pd.DataFrame([dict(algo=a, best_run=best.loc[a, "run"], eval_reward=round(best.loc[a, "eval_reward"], 1),
                             on_time=f"{best.loc[a, 'on_time_rate']:.0%}",
                             hard_on_time=f"{best.loc[a, 'hard_on_time_rate']:.0%}",
                             episodes_to_converge=episodes_to(best.loc[a, "run"])) for a in ALGOS])
lines = ["# Summary\n", summary.to_markdown(index=False)] + lines
with open("logs/tables.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
