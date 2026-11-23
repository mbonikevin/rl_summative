"""Run every hyperparameter config for all four algorithms, several at a time.

Run: uv run python -m training.run_all
"""
from multiprocessing import Pool

import torch

from training import dqn_training, pg_training


def run(job):
    torch.set_num_threads(2)
    module, cfg = job
    try:
        module.train(cfg)
    except Exception as e:  # one failed run should not stop the whole sweep
        print("FAILED", cfg["name"], e, flush=True)


if __name__ == "__main__":
    jobs = [(pg_training, c) for c in pg_training.ALL.values()] + [(dqn_training, c) for c in dqn_training.CONFIGS]
    with Pool(9) as pool:
        pool.map(run, jobs, chunksize=1)
