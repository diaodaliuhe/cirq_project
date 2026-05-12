import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy, time, tempfile
import numpy as np

from pathlib import Path
from datetime import datetime
import platform
import sys


from experiments.utils import load_config
from experiments import (
    run_exp1_experiment_serial,
    run_exp1_experiment_mp,
)

def set_env():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run_once(fn, cfg):
    t0 = time.perf_counter()
    results, meta = fn(cfg)
    dt = time.perf_counter() - t0
    return dt, results, meta

def compare_samples(meta_a, meta_b, tau_c_list, logger=print, atol=1e-12):
    # 假设 meta["samples"][tau]["fidelity"] 是 numpy array-like（list 或 memmap 都行）
    for tau in tau_c_list:
        fa = np.asarray(meta_a["samples"][tau]["fidelity"], dtype=np.float64)
        fb = np.asarray(meta_b["samples"][tau]["fidelity"], dtype=np.float64)
        ta = np.asarray(meta_a["samples"][tau]["trace_distance"], dtype=np.float64)
        tb = np.asarray(meta_b["samples"][tau]["trace_distance"], dtype=np.float64)

        # 完整性
        assert not np.isnan(fb).any(), f"NaN in mp fidelity @ tau={tau}"
        assert not np.isnan(tb).any(), f"NaN in mp trace_distance @ tau={tau}"

        # 点对点差异
        df = np.max(np.abs(fa - fb))
        dt = np.max(np.abs(ta - tb))
        logger(f"[tau={tau}] max|Δfid|={df:.3e}, max|Δtd|={dt:.3e}")
        if df > atol or dt > atol:
            logger("  WARNING: exceeds atol; consider using allclose with looser tol.")

if __name__ == "__main__":
    set_env()
    cfg0 = load_config("exp1_fid.yaml")
    # ==== log file in project root ====
    proj_root = Path(__file__).resolve().parents[1]   # .../cirq_project
    log_path = proj_root / f"bench_exp1_mp_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log_lines: list[str] = []

    def log(msg: str = ""):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[INFO] python={sys.version.split()[0]} platform={platform.platform()}")
    log(f"[INFO] OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')} "
        f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS')} "
        f"OPENBLAS_NUM_THREADS={os.getenv('OPENBLAS_NUM_THREADS')} "
        f"NUMEXPR_NUM_THREADS={os.getenv('NUMEXPR_NUM_THREADS')}")


    # ===== 小规模但“任务数足够多”的 benchmark 配置 =====
    base = copy.deepcopy(cfg0)

    # tau_c_list
    # base["experiment"]["tau_c_list"] = list(np.linspace(10.0, 200.0, 20))

    # n_samples 
    base["experiment"]["n_samples"] = 40

    # chunk_size
    fixed_chunk = 5
    base["experiment"]["chunk_size"] = fixed_chunk

    # --- baseline：mp + n_workers=1 ---
    cfg_base = copy.deepcopy(base)
    cfg_base["experiment"]["n_workers"] = 1
    cfg_base["experiment"]["scratch_dir"] = tempfile.mkdtemp(prefix="exp1_base_")

    t_base, r_base, m_base = run_once(run_exp1_experiment_mp, cfg_base)
    n_tasks = len(cfg_base["experiment"]["tau_c_list"]) * int(cfg_base["experiment"]["n_samples"])
    log(f"[BASE] mp(n_workers=1, chunk={fixed_chunk}) time={t_base:.3f}s, {n_tasks/t_base:.3f} samples/s")

    # --- chunk_size=5, sweep n_workers ---
    for nw in [4, 8, 16, 32, 48]:
        cfg = copy.deepcopy(base)
        cfg["experiment"]["n_workers"] = nw
        cfg["experiment"]["scratch_dir"] = tempfile.mkdtemp(prefix=f"exp1_nw{nw}_")

        t, r, m = run_once(run_exp1_experiment_mp, cfg)
        log(f"mp(n_workers={nw}, chunk={fixed_chunk}) time={t:.3f}s, speedup={t_base/t:.2f}x, {n_tasks/t:.3f} samples/s")

        # 正确性
        if nw == 48:
            compare_samples(m_base, m, cfg["experiment"]["tau_c_list"], logger=log, atol=1e-12)

    # ===== n_workers=48, sweep chunk_size =====
    nw_fixed = 48
    for ch in [1, 2, 5, 10, 20, 40]:
        cfg = copy.deepcopy(base)
        
        cfg["experiment"]["n_workers"] = nw_fixed
        cfg["experiment"]["chunk_size"] = ch
        cfg["experiment"]["scratch_dir"] = tempfile.mkdtemp(prefix=f"exp1_ch{ch}_")

        t, r, m = run_once(run_exp1_experiment_mp, cfg)
        log(f"mp(n_workers={nw_fixed}, chunk={ch}) time={t:.3f}s, {n_tasks/t:.3f} samples/s")
    
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {log_path}", flush=True)
