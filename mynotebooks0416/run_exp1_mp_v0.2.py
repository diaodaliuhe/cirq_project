import os
# 必须在任何 numpy/scipy/cirq 导入之前进行设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import copy
from pathlib import Path
from datetime import datetime

# Import necessary functions from experiment1_fid_refactored
from experiments.scripts.experiment1_fid import run_exp1_mp_and_save
from experiments.utils import load_config

cfg = load_config("exp1_fid.yaml")

# Running the experiment and saving artifacts
def run_experiment_and_save(cfg):
    # Generate a timestamped run directory
    run_dir = Path(f"results")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run the experiment and save the results, including the config, samples, and reports
    results, meta, paths = run_exp1_mp_and_save(
        cfg,  # config
        exp_name="experiment1",  # Experiment name
        base_results_dir=str(run_dir),  # Directory for results
        preview_max_per_tau=100,  # Max samples per tau_c to preview
        raw_mode="inplace",  # Raw data saving mode (copy, move, inplace)
    )

    # Print paths of saved files
    print(f"Experiment results and artifacts saved to {run_dir}")
    print(f"Results summary: {paths['summary']}")
    print(f"Samples preview file: {paths['samples_preview']}")
    # print(f"Report file: {paths['report']}")
    print(f"Raw data directory: {paths['raw_dir']}")

# Run the function
run_experiment_and_save(cfg)