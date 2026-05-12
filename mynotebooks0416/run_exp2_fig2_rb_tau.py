import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from pathlib import Path
from datetime import datetime

from experiments.scripts.experiment2_rb import (
    run_exp2_rb_decay,
    save_results_exp2_rb_decay,
)
from experiments.utils import load_config


CFG_PATH = "exp2_fig2_rb_tau.yaml"


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    return obj


def make_run_dir(base_dir: str = "results/experiment2") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_experiment_and_save():
    cfg = load_config(CFG_PATH)

    run_dir = make_run_dir()
    out = run_exp2_rb_decay(cfg)

    result_txt = save_results_exp2_rb_decay(
        cfg,
        out,
        run_dir,
        results_filename="results_exp2_fig2_rb_tau.txt",
    )

    json_path = run_dir / "rb_tau_sweep.json"
    json_path.write_text(
        json.dumps(_to_builtin(out), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] run_dir   -> {run_dir}")
    print(f"[OK] result_tx -> {result_txt}")
    print(f"[OK] json      -> {json_path}")


if __name__ == "__main__":
    run_experiment_and_save()
