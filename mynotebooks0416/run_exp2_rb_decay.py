import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from datetime import datetime

from experiments.scripts.experiment2_rb import (
    run_exp2_rb_decay,
    save_results_exp2_rb_decay,
)
from experiments.utils import load_config

cfg = load_config("exp2_rb.yaml")

def run_experiment_and_save(cfg):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("results") / "experiment2" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    parallel_mode = cfg.get("experiment", {}).get("parallel_mode", "serial")
    print(f"[INFO] exp2 parallel_mode = {parallel_mode}")

    bound_estimate_nseq = cfg.get("experiment", {}).get("bound_estimate_nseq", 30)
    print(f"[INFO] exp2 bound_estimate_nseq = {bound_estimate_nseq}")

    out = run_exp2_rb_decay(cfg)
    save_path = save_results_exp2_rb_decay(
        cfg,
        out,
        run_dir,
    )

    print(f"Saved to: {save_path}")
    print(f"Run directory: {run_dir}")

if __name__ == "__main__":
    run_experiment_and_save(cfg)