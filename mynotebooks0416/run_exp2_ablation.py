import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from datetime import datetime

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from experiments.scripts.experiment2_rb import (
    run_exp2_ablation,
    save_results_exp2_ablation,
)
from experiments.utils import load_config, make_run_dir

cfg = load_config("exp2_ablation.yaml")

def run_experiment_and_save(cfg):
    out = run_exp2_ablation(cfg)

    run_dir = make_run_dir(exp_name="experiment2")
    save_path = save_results_exp2_ablation(
        cfg,
        out,
        run_dir,
    )

    print(f"Saved to: {save_path}")

run_experiment_and_save(cfg)