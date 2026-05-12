# -*- coding: utf-8 -*-
"""
Experiment 1: τ_c-dependence under segmented flux noise (fidelity / trace distance).

Design goals
- Deterministic & reproducible sweeps (seeded per (tau_idx, sample_idx)).
- Parallel-friendly raw sample persistence (memmap-backed).
- Decouple "run" from "post-processing":
    (1) run_exp1_experiment_*  -> returns (results, meta)
    (2) save_*                -> persist config/env/raw/summary/preview/report
    (3) post-processing       -> read raw manifests and re-load arrays later

Notes on multiprocessing
- For large circuits/arrays, prefer start_method="fork" on Linux to avoid pickling huge objects.
- Before importing heavy numeric stacks in your launcher script, set:
    OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1
  to reduce "fork + BLAS threads" memory blow-ups.
"""

from __future__ import annotations

# =============================================================================
# 0. Imports
# =============================================================================
_TRACE_CACHE: Dict[Tuple[str, Tuple[int, int]], np.memmap] = {}

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import copy
import json
import os
import platform
import shutil
import socket
import subprocess
import time
import secrets
import multiprocessing as mp
import queue

import numpy as np
from tqdm.auto import tqdm

import cirq
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project utilities
from cirq.noise.utils.metrics import fidelity1, trace_distance
from cirq.noise.utils.timed_circuit_context import assign_timed_circuit_context
from cirq.noise.utils.composite_noise_model import CompositeNoiseModel
from cirq.noise.utils.circuit_factory import build_circuit_from_config
from cirq.noise.utils.circuit_timing_cal import compute_timing_summary
from cirq.noise.utils.noise_builder import (
    _get,
    derive_rng,
    build_idle_from_yaml,
    build_ry_from_yaml,
    build_photon_decay_from_yaml,
    is_flux_enabled,
    get_flux_sigma_seed,
    get_flux_sampling_cfg,
)
from cirq.noise.models.segmented_flux_noise_model import SegmentedFluxNoiseModel
from cirq.noise.models.average_flux_noise_model import AverageFluxNoiseModel
from cirq.noise.utils.compilation_scheme import all_in_one_compile

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # optional dependency


# =============================================================================
# 1. Helper functions (formatting / config / environment / IO)[run side]
# =============================================================================


def _require(cfg: Dict, path: str, typ=None):
    """Fetch a value from nested dict with a strict 'must exist' check."""
    v = _get(cfg, path, None)
    if v is None:
        top = list(cfg.keys()) if isinstance(cfg, dict) else type(cfg).__name__
        raise KeyError(f"Config missing '{path}'. Top-level keys: {top}")
    if typ and not isinstance(v, typ):
        raise TypeError(f"Config '{path}' should be {typ}, got {type(v)}: {v!r}")
    return v


def _limit_blas_threads(force: bool = False) -> None:
    """
    Reduce oversubscription for "multiprocessing + BLAS threads".
    If force=False, keep user's explicit settings.
    """
    setter = (lambda k, v: os.environ.__setitem__(k, v)) if force else os.environ.setdefault
    setter("OMP_NUM_THREADS", "1")
    setter("MKL_NUM_THREADS", "1")
    setter("OPENBLAS_NUM_THREADS", "1")
    setter("NUMEXPR_NUM_THREADS", "1")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _git_info(repo_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Best-effort git metadata. If not in a git repo, returns empty dict.
    """
    repo_dir = repo_dir or Path(__file__).resolve().parents[3]
    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, cwd=str(repo_dir), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    if not head:
        return {}
    status = _run(["git", "status", "--porcelain"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "git_commit": head,
        "git_branch": branch,
        "git_dirty": bool(status),
    }

def _inject_circuit_info_to_meta(circ_cfg: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """
    动态注入电路信息到 meta 字典中。
    注入内容包括 circ_cfg 中的所有键值对，排除非电路参数。
    """
    # 只保留除非电路无关的配置项（如 write_txt）以外的所有键值对
    circuit_params = {k: v for k, v in circ_cfg.items() if k != "write_txt"}
    
    # 将电路参数注入到 meta 中
    meta["circuit_params"] = circuit_params

def _resolve_sample_stride(segs: int, sample_stride_segments: Optional[int]) -> int:
    stride = int(segs) if sample_stride_segments is None else int(sample_stride_segments)
    if stride <= 0:
        raise ValueError(f"sample_stride_segments must be positive, got {stride}")
    return stride

def _sample_delta_phis_local_correlated(
    *,
    qubits,
    segs: int,
    sigma: float,
    base_flux_seed: int,
    tau_idx: int,
    sample_idx: int,
):
    rng = derive_rng(int(base_flux_seed), int(tau_idx), int(sample_idx))
    return {
        seg: {q: float(rng.normal(0.0, float(sigma))) for q in qubits}
        for seg in range(int(segs))
    }

def _reflect_into_bounds(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    将 x 反射回 [low, high]。
    支持一次越过多个区间宽度，不需要 while 循环。
    """
    width = float(high) - float(low)
    if width <= 0:
        raise ValueError(f"Invalid bounds: low={low}, high={high}")

    z = (x - low) % (2.0 * width)
    return np.where(z <= width, low + z, high - (z - width))

def _fill_ar1_trace(
    out: np.ndarray,
    *,
    sigma: float,
    rho: float,
    rng: np.random.Generator,
    bound_low: Optional[float] = None,
    bound_high: Optional[float] = None,
) -> None:
    """
    out.shape == (n_blocks, n_qubits)
    """
    n_blocks, n_qubits = out.shape
    if n_blocks <= 0:
        return

    x0 = rng.normal(0.0, float(sigma), size=n_qubits)
    if bound_low is not None and bound_high is not None:
        x0 = _reflect_into_bounds(x0, bound_low, bound_high)
    out[0, :] = x0

    coeff = float(sigma) * float(np.sqrt(max(0.0, 1.0 - float(rho) ** 2)))
    for j in range(1, n_blocks):
        eps = rng.normal(0.0, 1.0, size=n_qubits)
        x = float(rho) * out[j - 1, :] + coeff * eps
        if bound_low is not None and bound_high is not None:
            x = _reflect_into_bounds(x, bound_low, bound_high)
        out[j, :] = x

def _build_global_trace_ar1_file(
    path: Union[str, Path],
    *,
    n_blocks: int,
    n_qubits: int,
    sigma: float,
    rho: float,
    base_flux_seed: int,
    tau_idx: int,
    bound_low: Optional[float] = None,
    bound_high: Optional[float] = None,
) -> Tuple[str, Tuple[int, int]]:
    path = str(path)
    mm = np.memmap(path, mode="w+", dtype=np.float64, shape=(int(n_blocks), int(n_qubits)))
    rng = derive_rng(int(base_flux_seed), int(tau_idx), 314159)
    _fill_ar1_trace(
        mm,
        sigma=float(sigma),
        rho=float(rho),
        rng=rng,
        bound_low=bound_low,
        bound_high=bound_high,
    )
    mm.flush()
    return path, (int(n_blocks), int(n_qubits))

def _slice_delta_phis_from_trace(
    trace_2d: np.ndarray,
    *,
    qubits,
    segs: int,
    sample_idx: int,
    sample_stride_segments: Optional[int],
):
    segs = int(segs)
    stride = _resolve_sample_stride(segs, sample_stride_segments)
    abs0 = int(sample_idx) * stride

    out = {}
    for local_seg in range(segs):
        row = trace_2d[abs0 + local_seg]
        out[local_seg] = {q: float(v) for q, v in zip(qubits, row)}
    return out

def _get_trace_memmap_cached(path: str, shape: Tuple[int, int]) -> np.memmap:
    key = (str(path), (int(shape[0]), int(shape[1])))
    hit = _TRACE_CACHE.get(key)
    if hit is not None:
        return hit
    mm = np.memmap(path, mode="r", dtype=np.float64, shape=shape)
    _TRACE_CACHE[key] = mm
    return mm

def _resolve_seed(raw: Any, *, default: Optional[int] = None, name: str = "seed") -> int:
    """
    解析 seed：
    - None -> default
    - "auto" -> 自动生成一个适合 Cirq / numpy RandomState 的 32-bit seed
    - int / 可转 int 的字符串 -> 对应整数（要求在 [0, 2**32 - 1]）
    """
    MAX_SEED = 2**32 - 1

    if raw is None:
        if default is None:
            raise ValueError(f"{name} is required but got None.")
        value = int(default)
    elif isinstance(raw, str):
        s = raw.strip().lower()
        if s == "auto":
            return int(secrets.randbelow(2**32))
        value = int(raw)
    else:
        try:
            value = int(raw)
        except Exception as e:
            raise ValueError(f"Invalid {name}: {raw!r}") from e

    if not (0 <= value <= MAX_SEED):
        raise ValueError(
            f"{name} must be in [0, {MAX_SEED}], got {value}."
        )
    return value

def _get_flux_ar1_bounds_cfg(noise_cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    fx = noise_cfg.get("flux_quasistatic", {}) or {}
    sampling = fx.get("sampling", {}) or {}
    bounds = sampling.get("bounds", {}) or {}

    enabled = bool(bounds.get("enabled", False))
    if not enabled:
        return None, None

    low = float(bounds["low"])
    high = float(bounds["high"])
    if not (low < high):
        raise ValueError(f"AR1 bounds must satisfy low < high, got low={low}, high={high}")
    return low, high

# =============================================================================
# 2. Parallel worker (top-level for pickling / fork)
# =============================================================================
def _prepare_one_tau(args: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        tau_idx, tau_c,
        c_num, qubits,
        timing_cfg, noise_cfg,
        n_samples, scratch_dir,
        flux_sampling_mode,
        flux_sample_stride_segments,
        flux_sigma, flux_sampling_rho,
        base_flux_seed,
        bound_low, bound_high,
    ) = args

    _limit_blas_threads(force=False)

    # 1) 算 segs
    ctx_tmp = assign_timed_circuit_context(
        c_num,
        t1=timing_cfg["t1"],
        t2=timing_cfg["t2"],
        tau_c=float(tau_c),
    )
    max_seg = max(info.segment_id for infos in ctx_tmp.timing_map.values() for info in infos)
    segs = int(max_seg + 1)

    # 2) raw 文件
    scratch_dir = Path(scratch_dir)
    fid_path = scratch_dir / f"fid_tau{tau_idx}.dat"
    td_path  = scratch_dir / f"td_tau{tau_idx}.dat"

    np.memmap(fid_path, mode="w+", dtype=np.float64, shape=(int(n_samples),))[:] = np.nan
    np.memmap(td_path,  mode="w+", dtype=np.float64, shape=(int(n_samples),))[:] = np.nan

    # 3) trace 文件（仅 global_trace_ar1）
    flux_trace_path = None
    flux_trace_shape = None

    if is_flux_enabled(noise_cfg) and str(flux_sampling_mode).lower() == "global_trace_ar1":
        stride = _resolve_sample_stride(segs, flux_sample_stride_segments)
        n_blocks = (int(n_samples) - 1) * stride + segs

        trace_path = scratch_dir / f"flux_trace_tau{tau_idx}.dat"
        flux_trace_path, flux_trace_shape = _build_global_trace_ar1_file(
            trace_path,
            n_blocks=int(n_blocks),
            n_qubits=len(qubits),
            sigma=float(flux_sigma),
            rho=float(flux_sampling_rho),
            base_flux_seed=int(base_flux_seed),
            tau_idx=int(tau_idx),
            bound_low=bound_low,
            bound_high=bound_high,
        )

    return {
        "tau_idx": int(tau_idx),
        "tau_c": float(tau_c),
        "segs": int(segs),
        "fidelity1_path": str(fid_path),
        "trace_distance_path": str(td_path),
        "dtype": "float64",
        "shape": [int(n_samples)],
        "flux_trace_path": flux_trace_path,
        "flux_trace_shape": list(flux_trace_shape) if flux_trace_shape is not None else None,
    }

# Per-process cache to avoid re-building ctx & fixed models for every chunk.
_TAU_CACHE: Dict[Tuple[int, float], Tuple[Any, int, List[cirq.NoiseModel]]] = {}


def _get_tau_cached(
    tau_idx: int,
    tau_c: float,
    c_num: cirq.Circuit,
    timing_cfg: Dict[str, Any],
    noise_cfg: Dict[str, Any],
) -> Tuple[Any, int, List[cirq.NoiseModel]]:
    key = (int(tau_idx), float(tau_c))
    hit = _TAU_CACHE.get(key)
    if hit is not None:
        return hit

    ctx = assign_timed_circuit_context(
        c_num,
        t1=timing_cfg["t1"],
        t2=timing_cfg["t2"],
        tau_c=float(tau_c),
    )
    max_seg = max(info.segment_id for infos in ctx.timing_map.values() for info in infos)
    segs = int(max_seg + 1)

    fixed_models: List[cirq.NoiseModel] = []
    idle = build_idle_from_yaml(timing_cfg, noise_cfg)
    if idle is not None:
        fixed_models.append(idle)

    ry = build_ry_from_yaml(timing_cfg, noise_cfg)
    if ry is not None:
        fixed_models.append(ry)

    ph = build_photon_decay_from_yaml(noise_cfg, timed_ctx=ctx)
    if ph is not None:
        fixed_models.append(ph)

    _TAU_CACHE[key] = (ctx, segs, fixed_models)
    return _TAU_CACHE[key]


def _run_tau_chunk(args: Tuple[Any, ...]) -> Tuple[int, float, int]:
    """
    One task = fixed tau_c + sample slice [k0, k1).
    Writes directly into memmap files to avoid returning huge arrays to the parent.
    """
    _limit_blas_threads(force=False)

    (
        tau_idx, tau_c, k0, k1,
        n_samples,
        c_num, qubits, rho_ideal,
        timing_cfg, noise_cfg,
        seed, base_flux_seed, flux_sigma,
        flux_sampling_mode,
        flux_sample_stride_segments,
        flux_trace_path,
        flux_trace_shape,
        fid_path, td_path,
    ) = args

    ctx, segs, fixed_models = _get_tau_cached(int(tau_idx), float(tau_c), c_num, timing_cfg, noise_cfg)

    # A deterministic simulator seed per chunk.
    sim_noisy = cirq.DensityMatrixSimulator(seed=(int(seed) ^ (int(tau_idx) * 1_000_003) ^ int(k0)))

    fid_mm = np.memmap(fid_path, mode="r+", dtype=np.float64, shape=(int(n_samples),))
    td_mm  = np.memmap(td_path,  mode="r+", dtype=np.float64, shape=(int(n_samples),))

    for k in range(int(k0), int(k1)):
        if str(flux_sampling_mode).lower() == "local_correlated":
            delta_phis = _sample_delta_phis_local_correlated(
                qubits=qubits,
                segs=int(segs),
                sigma=float(flux_sigma),
                base_flux_seed=int(base_flux_seed),
                tau_idx=int(tau_idx),
                sample_idx=int(k),
            )

        elif str(flux_sampling_mode).lower() == "global_trace_ar1":
            if flux_trace_path is None or flux_trace_shape is None:
                raise RuntimeError("global_trace_ar1 requires flux_trace_path and flux_trace_shape.")
            trace_mm = _get_trace_memmap_cached(str(flux_trace_path), tuple(flux_trace_shape))
            delta_phis = _slice_delta_phis_from_trace(
                trace_mm,
                qubits=qubits,
                segs=int(segs),
                sample_idx=int(k),
                sample_stride_segments=flux_sample_stride_segments,
            )

        else:
            raise ValueError(f"Unknown flux_sampling_mode: {flux_sampling_mode!r}")

#        rng = derive_rng(int(base_flux_seed), int(tau_idx), int(k))
#
#        delta_phis = {
#            seg: {q: rng.normal(0.0, float(flux_sigma)) for q in qubits}
#            for seg in range(int(segs))
#        }

        models = list(fixed_models)
        if is_flux_enabled(noise_cfg):
            models.append(SegmentedFluxNoiseModel(ctx.timing_map, delta_phis))

        composite_model = CompositeNoiseModel(models)
        rho_noisy = sim_noisy.simulate(c_num.with_noise(composite_model)).final_density_matrix

        fid_mm[k] = fidelity1(rho_ideal, rho_noisy)
        td_mm[k]  = trace_distance(rho_ideal, rho_noisy)

    return (int(tau_idx), float(tau_c), int(segs))


# =============================================================================
# 3. Experiment runner: multiprocessing version (new)
# =============================================================================

def run_exp1_experiment_mp(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run τ_c sweep in parallel, backed by memmap raw files.

    Returns
    - results: list of dict rows (mean/std per tau_c + average_flux baseline)
    - meta:    dict including timing summary, raw samples (memmap views), provenance fields
    """

    #---------------- 1. Initialization and loading config ----------------

    wall_start = datetime.now() # 真实时间
    t0 = time.perf_counter() # 高精度计时器

    # Reduce BLAS oversubscription early (best effort).
    _limit_blas_threads(force=False) # 线程限制

    exp_cfg    = _require(cfg, "experiment", dict)
    circ_cfg   = _require(cfg, "circuit", dict)
    timing_cfg = _require(cfg, "timing", dict)
    noise_cfg  = _require(cfg, "noise", dict)

    #---------------- 2. Preprocess circuit and get ideal state ----------------

    c_sym, c_raw, _ = build_circuit_from_config(circ_cfg)

    if circ_cfg.get("compile", True):
        c_num = all_in_one_compile(c_raw)
    else:
        c_num = c_raw

    qubits = sorted(c_num.all_qubits(), key=lambda q: getattr(q, "x", str(q)))

    timing_summary = compute_timing_summary(c_num, timing_cfg["t1"], timing_cfg["t2"])

    seed = _resolve_seed(exp_cfg.get("seed", 2025), default=2025, name="experiment.seed")
    rho_ideal = cirq.DensityMatrixSimulator(seed=seed).simulate(c_num).final_density_matrix

    #---------------- 3. 解析 sweep 参数：tau_c_list 与采样次数 n_samples ----------------

    tau_raw = _require(exp_cfg, "tau_c_list")
    if isinstance(tau_raw, (list, tuple)):
        tau_c_list = [float(x) for x in tau_raw]
    elif isinstance(tau_raw, str):
        tau_c_list = [float(s) for s in tau_raw.replace(",", " ").split()]
    else:
        raise TypeError(f"'experiment.tau_c_list' should be list/tuple/str, got {type(tau_raw)}")

    n_samples = int(exp_cfg["n_samples"])

    #---------------- 4. flux parameters and enabled noises ----------------

    flux_sigma, flux_seed_raw = get_flux_sigma_seed(noise_cfg)
    base_flux_seed = (
        seed if flux_seed_raw is None
        else _resolve_seed(flux_seed_raw, name="noise.flux_quasistatic.flux_seed")
    )
    
    flux_sampling_cfg = get_flux_sampling_cfg(noise_cfg)
    flux_sampling_mode = str(flux_sampling_cfg["mode"])
    flux_sample_stride_segments = flux_sampling_cfg["sample_stride_segments"]
    flux_sampling_rho = flux_sampling_cfg["rho"]
    flux_bound_low, flux_bound_high = _get_flux_ar1_bounds_cfg(noise_cfg)

    # Enabled noise model names (YAML enabled flags only)
    enabled_names: List[str] = []
    if bool(noise_cfg.get("idle", {}).get("enabled", True)):
        enabled_names.append("IdleNoiseModel")
    if bool(noise_cfg.get("ry_gate", {}).get("enabled", True)):
        enabled_names.append("RyGateNoiseModel")
    if bool(noise_cfg.get("photon_decay", {}).get("enabled", True)):
        enabled_names.append("PhotonDecayNoiseModel")
    if is_flux_enabled(noise_cfg):
        enabled_names.append("SegmentedFluxNoiseModel")

    #---------------- 5. parallel parameters ----------------

    # Parallel parameters
    n_workers  = int(exp_cfg.get("n_workers", max(1, (os.cpu_count() or 1) // 2)))
    chunk_size = int(exp_cfg.get("chunk_size", 200))
    start_method = str(exp_cfg.get("mp_start_method", "fork"))

    scratch_dir = Path(exp_cfg.get("scratch_dir") or "/tmp")
    _ensure_dir(scratch_dir)

    mp_ctx = __import__("multiprocessing").get_context(start_method)

    #---------------- 6. 为每个 tau 建两份 memmap ----------------
    tau_infos: List[Dict[str, Any]] = []

    prepare_tasks: List[Tuple[Any, ...]] = []
    for tau_idx, tau_c in enumerate(tau_c_list):
        prepare_tasks.append(
            (
                int(tau_idx), float(tau_c),
                c_num, qubits,
                timing_cfg, noise_cfg,
                int(n_samples), str(scratch_dir),
                flux_sampling_mode,
                flux_sample_stride_segments,
                float(flux_sigma), flux_sampling_rho,
                int(base_flux_seed),
                flux_bound_low, flux_bound_high,
            )
        )

    prepare_workers = min(len(tau_c_list), os.cpu_count() or len(tau_c_list))

    with ProcessPoolExecutor(max_workers=prepare_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(_prepare_one_tau, t) for t in prepare_tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="prepare taus", unit="tau"):
            tau_infos.append(fut.result())

    tau_infos.sort(key=lambda d: int(d["tau_idx"]))
#    # Prepare memmap files per tau (raw binary float64 arrays).
#    tau_infos: List[Dict[str, Any]] = []
#    for tau_idx, tau_c in enumerate(tau_c_list):
#        fid_path = scratch_dir / f"fid_tau{tau_idx}.dat"
#        td_path  = scratch_dir / f"td_tau{tau_idx}.dat"
#        np.memmap(fid_path, mode="w+", dtype=np.float64, shape=(n_samples,))[:] = np.nan
#        np.memmap(td_path,  mode="w+", dtype=np.float64, shape=(n_samples,))[:] = np.nan
#        tau_infos.append(
#            {
#                "tau_idx": int(tau_idx),
#                "tau_c": float(tau_c),
#               "fidelity1_path": str(fid_path),
#                "trace_distance_path": str(td_path),
#                "dtype": "float64",
#                "shape": [int(n_samples)],
#           }
#        )

    #---------------- 7. 构造 tasks：tau × chunks ----------------

    # Task list (tau × chunks)
    tasks: List[Tuple[Any, ...]] = []

    for info in tau_infos:
        tau_idx = int(info["tau_idx"])
        tau_c = float(info["tau_c"])
        fid_path = str(info["fidelity1_path"])
        td_path  = str(info["trace_distance_path"])

        flux_trace_path = info.get("flux_trace_path")
        flux_trace_shape = tuple(info["flux_trace_shape"]) if info.get("flux_trace_shape") is not None else None

        for k0 in range(0, n_samples, chunk_size):
            k1 = min(n_samples, k0 + chunk_size)
            tasks.append(
                (
                    tau_idx, tau_c, k0, k1,
                    n_samples,
                    c_num, qubits, rho_ideal,
                    timing_cfg, noise_cfg,
                    seed, base_flux_seed, flux_sigma,
                    flux_sampling_mode,
                    flux_sample_stride_segments,
                    flux_trace_path,
                    flux_trace_shape,
                    fid_path, td_path,
                )
            )
#        for k0 in range(0, n_samples, chunk_size):
#            k1 = min(n_samples, k0 + chunk_size)
#            tasks.append(
#               (
#                   tau_idx, tau_c, k0, k1,
#                    n_samples,
#                    c_num, qubits, rho_ideal,
#                    timing_cfg, noise_cfg,
#                    seed, base_flux_seed, flux_sigma,
#                    fid_path, td_path,
#                )
#            )
#
#    tau_segs: Dict[int, Optional[int]] = {int(info["tau_idx"]): None for info in tau_infos}

    #---------------- 8. 并行执行：ProcessPoolExecutor + as_completed + tqdm ----------------

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(_run_tau_chunk, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="tau×chunks", unit="chunk"):
            _ = fut.result()
#    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as ex:
#        futs = [ex.submit(_run_tau_chunk, t) for t in tasks]
#        for fut in tqdm(as_completed(futs), total=len(futs), desc="tau×chunks", unit="chunk"):
#            tau_idx, _tau_c, segs = fut.result()
#            if tau_segs[tau_idx] is None:
#                tau_segs[tau_idx] = int(segs)

    #---------------- 9. 汇总阶段：读回 memmap，算 mean/std，放入 meta["samples"] ----------------

    # Aggregate means/stds and expose memmap views in meta["samples"].
    results: List[Dict[str, Any]] = []
    samples: Dict[Any, Dict[str, Any]] = {}

    for info in tau_infos:
        tau_idx = int(info["tau_idx"])
        tau_c = float(info["tau_c"])
        fid_mm = np.memmap(info["fidelity1_path"], mode="r", dtype=np.float64, shape=(n_samples,))
        td_mm  = np.memmap(info["trace_distance_path"], mode="r", dtype=np.float64, shape=(n_samples,))

        if np.isnan(fid_mm).any() or np.isnan(td_mm).any():
            raise RuntimeError(f"NaN detected in memmap for tau_c={tau_c}. Some chunks may have failed.")

        samples[tau_c] = {"fidelity1": fid_mm, "trace_distance": td_mm}
        results.append(
            {
                "tau_c": tau_c,
                "segs": int(info["segs"]),
                "fidelity1_mean": float(np.mean(fid_mm)),
                "fidelity1_std": float(np.std(fid_mm)),
                "trace_distance_mean": float(np.mean(td_mm)),
                "trace_distance_std": float(np.std(td_mm)),
            }
        )

    results.sort(key=lambda r: float(r["tau_c"]))

    #---------------- 10. average_flux baseline ----------------

    # Average-flux baseline (single shot; treated as a reference line).
    tau_c_for_avg = float(exp_cfg.get("tau_c_for_average", tau_c_list[0]))
    ctx_avg = assign_timed_circuit_context(
        c_num,
        t1=timing_cfg["t1"],
        t2=timing_cfg["t2"],
        tau_c=tau_c_for_avg,
    )

    models: List[cirq.NoiseModel] = []
    idle = build_idle_from_yaml(timing_cfg, noise_cfg)
    if idle is not None:
        models.append(idle)
    ry = build_ry_from_yaml(timing_cfg, noise_cfg)
    if ry is not None:
        models.append(ry)
    ph = build_photon_decay_from_yaml(noise_cfg, timed_ctx=ctx_avg)
    if ph is not None:
        models.append(ph)
    if is_flux_enabled(noise_cfg):
        models.append(AverageFluxNoiseModel(flux_sigma))

    composite_model = CompositeNoiseModel(models)
    sim_noisy = cirq.DensityMatrixSimulator(seed=int(seed) ^ 0xA5A5A5A5)
    rho_noisy = sim_noisy.simulate(c_num.with_noise(composite_model)).final_density_matrix

    fid_avg = float(fidelity1(rho_ideal, rho_noisy))
    td_avg  = float(trace_distance(rho_ideal, rho_noisy))

    samples["average_flux"] = {"fidelity1": [fid_avg], "trace_distance": [td_avg]}
    results.append(
        {
            "tau_c": "average_flux",
            "segs": None,
            "fidelity1_mean": fid_avg,
            "fidelity1_std": 0.0,
            "trace_distance_mean": td_avg,
            "trace_distance_std": 0.0,
        }
    )

    #---------------- 11. meta ----------------

    runtime_sec = float(time.perf_counter() - t0)
    wall_end = datetime.now()

    cfg_effective = copy.deepcopy(cfg)
    cfg_effective.setdefault("experiment", {})["seed"] = int(seed)

    fx = cfg_effective.setdefault("noise", {}).setdefault("flux_quasistatic", {})
    fx["flux_seed"] = int(base_flux_seed)

    meta: Dict[str, Any] = {
        "exp_name": exp_cfg.get("name", "exp1"),
        "title": exp_cfg.get("title", ""),
        "seed": seed,
        "tau_c_list": tau_c_list,
        "n_samples": n_samples,
        "sigma": float(flux_sigma),
        "flux_sampling_mode": flux_sampling_mode,
        "flux_sample_stride_segments": flux_sample_stride_segments,
        "flux_sampling_rho": flux_sampling_rho,
        "noise_models": enabled_names,
        "timing_summary": timing_summary,
        # "circuit_name": circ_cfg.get("type", "unknown"),
        # "n_layers": circ_cfg.get("n_layers"),
        "metrics": cfg.get("metrics", ["fidelity1", "trace_distance"]),
        "samples": samples,
        "raw_memmap_files": tau_infos,
        "scratch_dir": str(scratch_dir),
        "n_workers": int(n_workers),
        "chunk_size": int(chunk_size),
        "mp_start_method": start_method,
        "run_started_at": wall_start.strftime("%Y-%m-%d %H:%M:%S"),
        "run_finished_at": wall_end.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": runtime_sec,
        "blas_threads_env": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS"),
        },
        # Provenance: keep the effective config for saving to config.yaml.
        "cfg": cfg_effective,
        "flux_seed": int(base_flux_seed),
        # Optional circuit texts for archival.
        "circuit_sym_text": str(c_sym),
        "circuit_compiled_text": str(c_num),
    }

    _inject_circuit_info_to_meta(circ_cfg, meta)

    return results, meta


# =============================================================================
# 3. Experiment runner: serial version (kept as-is logically)
# =============================================================================

def run_exp1_experiment_serial(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Serial reference implementation (useful for correctness checks).
    """
    wall_start = datetime.now()
    t0 = time.perf_counter()

    exp_cfg    = _require(cfg, "experiment", dict)
    circ_cfg   = _require(cfg, "circuit", dict)
    timing_cfg = _require(cfg, "timing", dict)
    noise_cfg  = _require(cfg, "noise", dict)

    c_sym, c_raw, _ = build_circuit_from_config(circ_cfg)
    c_num = all_in_one_compile(c_raw)

    qubits = sorted(c_num.all_qubits(), key=lambda q: getattr(q, "x", str(q)))

    timing_summary = compute_timing_summary(c_num, timing_cfg["t1"], timing_cfg["t2"])

    seed = int(exp_cfg.get("seed", 2025))
    sim_ideal = cirq.DensityMatrixSimulator(seed=seed)
    rho_ideal = sim_ideal.simulate(c_num).final_density_matrix

    tau_raw = _require(exp_cfg, "tau_c_list")
    if isinstance(tau_raw, (list, tuple)):
        tau_c_list = [float(x) for x in tau_raw]
    elif isinstance(tau_raw, str):
        tau_c_list = [float(s) for s in tau_raw.replace(",", " ").split()]
    else:
        raise TypeError(f"'experiment.tau_c_list' should be list/tuple/str, got {type(tau_raw)}")

    n_samples = int(exp_cfg["n_samples"])
    flux_sigma, flux_seed = get_flux_sigma_seed(noise_cfg)

    base_flux_seed = int(seed) if flux_seed is None else (int(seed) ^ int(flux_seed))

    enabled_names: List[str] = []
    if bool(noise_cfg.get("idle", {}).get("enabled", True)):
        enabled_names.append("IdleNoiseModel")
    if bool(noise_cfg.get("ry_gate", {}).get("enabled", True)):
        enabled_names.append("RyGateNoiseModel")
    if bool(noise_cfg.get("photon_decay", {}).get("enabled", True)):
        enabled_names.append("PhotonDecayNoiseModel")
    if is_flux_enabled(noise_cfg):
        enabled_names.append("SegmentedFluxNoiseModel")

    results: List[Dict[str, Any]] = []
    samples: Dict[Any, Dict[str, Any]] = {}

    for tau_idx, tau_c in enumerate(tqdm(tau_c_list, desc="tau sweep (serial)", unit="tau")):
        ctx = assign_timed_circuit_context(c_num, t1=timing_cfg["t1"], t2=timing_cfg["t2"], tau_c=float(tau_c))
        max_seg = max(info.segment_id for infos in ctx.timing_map.values() for info in infos)
        segs = int(max_seg + 1)

        fixed_models: List[cirq.NoiseModel] = []
        idle = build_idle_from_yaml(timing_cfg, noise_cfg)
        if idle is not None:
            fixed_models.append(idle)
        ry = build_ry_from_yaml(timing_cfg, noise_cfg)
        if ry is not None:
            fixed_models.append(ry)
        ph = build_photon_decay_from_yaml(noise_cfg, timed_ctx=ctx)
        if ph is not None:
            fixed_models.append(ph)

        fids: List[float] = []
        tds: List[float] = []

        sim_noisy = cirq.DensityMatrixSimulator(seed=(int(seed) ^ (int(tau_idx) * 1_000_003)))

        for k in range(n_samples):
            rng = derive_rng(int(base_flux_seed), int(tau_idx), int(k))

            delta_phis = {
                seg: {q: rng.normal(0.0, float(flux_sigma)) for q in qubits}
                for seg in range(segs)
            }

            models = list(fixed_models)
            if is_flux_enabled(noise_cfg):
                models.append(SegmentedFluxNoiseModel(ctx.timing_map, delta_phis))

            composite_model = CompositeNoiseModel(models)
            rho_noisy = sim_noisy.simulate(c_num.with_noise(composite_model)).final_density_matrix

            fids.append(float(fidelity1(rho_ideal, rho_noisy)))
            tds.append(float(trace_distance(rho_ideal, rho_noisy)))

        samples[float(tau_c)] = {"fidelity1": fids, "trace_distance": tds}

        results.append(
            {
                "tau_c": float(tau_c),
                "segs": segs,
                "fidelity1_mean": float(np.mean(fids)),
                "fidelity1_std": float(np.std(fids)),
                "trace_distance_mean": float(np.mean(tds)),
                "trace_distance_std": float(np.std(tds)),
            }
        )

    results.sort(key=lambda r: float(r["tau_c"]))

    # Average-flux baseline
    tau_c_for_avg = float(exp_cfg.get("tau_c_for_average", tau_c_list[0]))
    ctx_avg = assign_timed_circuit_context(c_num, t1=timing_cfg["t1"], t2=timing_cfg["t2"], tau_c=tau_c_for_avg)

    models: List[cirq.NoiseModel] = []
    idle = build_idle_from_yaml(timing_cfg, noise_cfg)
    if idle is not None:
        models.append(idle)
    ry = build_ry_from_yaml(timing_cfg, noise_cfg)
    if ry is not None:
        models.append(ry)
    ph = build_photon_decay_from_yaml(noise_cfg, timed_ctx=ctx_avg)
    if ph is not None:
        models.append(ph)
    if is_flux_enabled(noise_cfg):
        models.append(AverageFluxNoiseModel(flux_sigma))

    composite_model = CompositeNoiseModel(models)
    sim_noisy = cirq.DensityMatrixSimulator(seed=int(seed) ^ 0xA5A5A5A5)
    rho_noisy = sim_noisy.simulate(c_num.with_noise(composite_model)).final_density_matrix

    fid_avg = float(fidelity1(rho_ideal, rho_noisy))
    td_avg  = float(trace_distance(rho_ideal, rho_noisy))

    samples["average_flux"] = {"fidelity1": [fid_avg], "trace_distance": [td_avg]}
    results.append(
        {
            "tau_c": "average_flux",
            "segs": None,
            "fidelity1_mean": fid_avg,
            "fidelity1_std": 0.0,
            "trace_distance_mean": td_avg,
            "trace_distance_std": 0.0,
        }
    )

    runtime_sec = float(time.perf_counter() - t0)
    wall_end = datetime.now()

    meta: Dict[str, Any] = {
        "exp_name": exp_cfg.get("name", "exp1"),
        "title": exp_cfg.get("title", ""),
        "seed": seed,
        "tau_c_list": tau_c_list,
        "n_samples": n_samples,
        "sigma": float(flux_sigma),
        "noise_models": enabled_names,
        "timing_summary": timing_summary,
        # "circuit_name": circ_cfg.get("type", "unknown"),
        # "n_layers": circ_cfg.get("n_layers"),
        "metrics": cfg.get("metrics", ["fidelity1", "trace_distance"]),
        "samples": samples,
        "run_started_at": wall_start.strftime("%Y-%m-%d %H:%M:%S"),
        "run_finished_at": wall_end.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": runtime_sec,
        "cfg": copy.deepcopy(cfg),
        "circuit_sym_text": str(c_sym),
        "circuit_compiled_text": str(c_num),
    }

    _inject_circuit_info_to_meta(circ_cfg, meta)

    return results, meta


# =============================================================================
# 4. Helper functions (?? / ?? / ?? / ??)[show side]
# =============================================================================


def _format_seconds(sec: float) -> str:
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _safe_json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _safe_yaml_dump(obj: Any, path: Path) -> None:
    if yaml is None:
        # Fallback: JSON with .yaml suffix; still human-readable enough.
        _safe_json_dump(obj, path)
        return
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True) + "\n", encoding="utf-8")


def collect_environment_info(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Capture run environment (for provenance / reproducibility).
    """
    info: Dict[str, Any] = {
        "time": _now_str(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": os.path.realpath(os.sys.executable),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "env_threads": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS"),
        },
    }
    info.update(_git_info())
    if extra:
        info.update(extra)
    return info


def _pick_indices(n: int, m: int) -> np.ndarray:
    """Deterministic uniform subsampling: pick m indices from [0, n-1]."""
    if m <= 0 or n <= 0:
        return np.array([], dtype=np.int64)
    if n <= m:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=m, dtype=np.int64)


# =============================================================================
# 5. Persistence layer: config/env/raw/summary/preview/report
# =============================================================================

def write_samples_preview_tsv(
    out_path: Path,
    samples: Dict[Any, Dict[str, Any]],
    *,
    chunk_lines: int = 10000,
    max_per_tau: Optional[int] = 100,
) -> None:
    """
    Write a small TSV preview of raw samples:
      tau_c<TAB>metric<TAB>value

    - max_per_tau: per metric per tau_c; None => write all.
    - Works for list / ndarray / memmap.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("tau_c\tmetric\tvalue\n")

        def _write_metric(tau_key: Any, metric: str, arr: Any) -> None:
            if arr is None:
                return
            try:
                n = len(arr)
            except Exception:
                arr = list(arr)
                n = len(arr)

            idx = np.arange(n, dtype=np.int64) if max_per_tau is None else _pick_indices(n, int(max_per_tau))
            buf: List[str] = []
            for j, k in enumerate(idx):
                buf.append(f"{tau_key}\t{metric}\t{float(arr[int(k)]):.10f}\n")
                if (j + 1) % chunk_lines == 0:
                    f.writelines(buf)
                    buf.clear()
            if buf:
                f.writelines(buf)

        for tau_key, d in samples.items():
            _write_metric(tau_key, "fidelity1", d.get("fidelity1", []))
            _write_metric(tau_key, "trace_distance", d.get("trace_distance", []))


def write_summary_table_txt(
    out_path: Path,
    results: List[Dict[str, Any]],
    meta: Dict[str, Any],
    *,
    extra_header_lines: Optional[List[str]] = None,
) -> None:
    """
    Human-readable summary (means/stds). Suitable for quick glance in terminal.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = meta.get("timing_summary", {}) or {}

    header: List[str] = []
    header.append(f"Experiment: {meta.get('exp_name', '-')}")
    header.append(f"Title: {meta.get('title', '-')}")
    header.append(f"Time: {_now_str()}\n")

    header.append(f"Run started at: {meta.get('run_started_at', '-')}")
    header.append(f"Run finished at: {meta.get('run_finished_at', '-')}")
    if "runtime_sec" in meta:
        header.append(f"Runtime: {_format_seconds(meta['runtime_sec'])}  ({meta['runtime_sec']:.3f} s)\n")
    else:
        header.append("Runtime: -\n")

    header.append("[Meta]")
    header.append(f"Circuit: {meta.get('circuit_params', {}).get('type', '-')}")
    if meta.get('circuit_params'):
        header.append(f"Circuit Parameters: {meta['circuit_params']}")
    header.append(f"Qubits: {ts.get('n_qubits', '-')}")
    header.append(f"Depth (moments): {ts.get('depth', '-')}")
    header.append(f"1q ops: {ts.get('n_ops_1q', '-')}, 2q ops: {ts.get('n_ops_2q', '-')}, others: {ts.get('n_ops_other', '-')}")
    header.append(f"Total duration (ns): {ts.get('total_duration_ns', '-')}")
    header.append(f"Noise models: {', '.join(meta.get('noise_models', [])) or '-'}")
    header.append(f"Samples per τ_c: {meta.get('n_samples', '-')}")
    header.append(f"τ_c list: {meta.get('tau_c_list', '-')}")
    header.append(f"Seed: {meta.get('seed', '-')}")
    header.append(f"MP: n_workers={meta.get('n_workers', '-')}, chunk_size={meta.get('chunk_size', '-')}, start={meta.get('mp_start_method', '-')}")
    header.append("")

    if extra_header_lines:
        header.append("[Extra]")
        header.extend(extra_header_lines)
        header.append("")

    header.append("[Summary Results]")

    cols = ["tau_c (ns)", "segs", "mean fidelity1", "std", "mean trace_distance", "std"]
    widths = [max(len(c), 16) for c in cols]
    if results:
        widths[0] = max(widths[0], max(len(str(r.get("tau_c", ""))) for r in results))
        widths[1] = max(widths[1], max(len(str(r.get("segs", "-"))) for r in results))

    header_row = " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)

    lines: List[str] = ["\n".join(header), header_row, sep]
    for r in results:
        tau_val = r.get("tau_c", "-")
        tau_str = f"{tau_val:<{widths[0]}}"
        segs_val = r.get("segs", None)
        segs_str = f"{('-' if segs_val is None else segs_val):<{widths[1]}}"

        lines.append(
            " | ".join(
                [
                    tau_str,
                    segs_str,
                    f"{float(r.get('fidelity1_mean', float('nan'))):<{widths[2]}.10f}",
                    f"{float(r.get('fidelity1_std', float('nan'))):<{widths[3]}.10f}",
                    f"{float(r.get('trace_distance_mean', float('nan'))):<{widths[4]}.10f}",
                    f"{float(r.get('trace_distance_std', float('nan'))):<{widths[5]}.10f}",
                ]
            )
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# def write_default_report_txt(out_path: Path, results: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
#     """
#     A minimal, general-purpose report: no hard-coded hypotheses, only descriptive stats.
#     """
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     numeric_rows = [r for r in results if isinstance(r.get("tau_c"), (int, float))]

#     def _extreme(key: str, fn=max):
#         if not numeric_rows:
#             return None
#         return fn(numeric_rows, key=lambda r: float(r.get(key, float("nan"))))

#     best = _extreme("fidelity1_mean", max)
#     worst = _extreme("fidelity1_mean", min)

#     avg = next((r for r in results if r.get("tau_c") == "average_flux"), None)

#     lines: List[str] = []
#     lines.append(f"Experiment: {meta.get('exp_name', '-')}")
#     lines.append(f"Title: {meta.get('title', '-')}")
#     lines.append(f"Generated at: {_now_str()}")
#     lines.append("")
#     lines.append("Overview")
#     lines.append(f"- τ_c points: {len(numeric_rows)} (+ average_flux={avg is not None})")
#     lines.append(f"- Samples per τ_c: {meta.get('n_samples', '-')}")
#     lines.append(f"- Circuit: {meta.get('circuit_name', '-')}, duration(ns)={meta.get('timing_summary', {}).get('total_duration_ns', '-')}")
#     lines.append(f"- Noise models: {', '.join(meta.get('noise_models', [])) or '-'}")
#     lines.append("")

#     if best:
#         lines.append("Key numbers")
#         lines.append(f"- Best fidelity1_mean: tau_c={best['tau_c']}  mean={best['fidelity1_mean']:.6f}  std={best['fidelity1_std']:.6f}")
#         lines.append(f"- Worst fidelity1_mean: tau_c={worst['tau_c']} mean={worst['fidelity1_mean']:.6f}  std={worst['fidelity1_std']:.6f}")
#         if avg:
#             lines.append(f"- Average-flux baseline: mean={avg['fidelity1_mean']:.6f}, td={avg['trace_distance_mean']:.6f}")
#         lines.append("")

#     lines.append("Raw data")
#     lines.append("- See raw/raw_manifest.json for file mapping (tau_idx -> tau_c -> raw .dat files).")
#     lines.append("- Use post-processing scripts to load memmaps and compute derived diagnostics.")
#     lines.append("")

#     out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def save_exp1_run_artifacts(
    results: List[Dict[str, Any]],
    meta: Dict[str, Any],
    run_dir: Union[str, Path],
    *,
    preview_max_per_tau: int = 100,
    write_raw: bool = True,
    keep_scratch_files: bool = False,
    raw_mode: str = "copy",
) -> Dict[str, Path]:
    """
    Persist a run into a directory structure:

    run_dir/
      config.yaml
      environment.json
      summary.txt
      samples_preview.tsv
      report.txt
      raw/
        raw_manifest.json
        fid_tau{idx}.dat
        td_tau{idx}.dat
        (optional: average_flux.json)

    Returns a dict of paths.
    """
    run_dir = _ensure_dir(run_dir)
    raw_dir = _ensure_dir(run_dir / "raw")

    raw_mode = str(raw_mode).lower()
    if raw_mode not in ("copy", "move", "inplace"):
        raise ValueError(f"raw_mode must be one of ('copy', 'move', 'inplace'), got {raw_mode!r}")

    # 1) Config & environment
    cfg_obj = meta.get("cfg", None)
    if cfg_obj is not None:
        _safe_yaml_dump(cfg_obj, run_dir / "config.yaml")

    env_info = collect_environment_info(extra={"experiment": meta.get("exp_name")})
    _safe_json_dump(env_info, run_dir / "environment.json")

    # 2) Raw samples
    raw_manifest: Dict[str, Any] = {"format": "exp1-raw-memmap-v1", "dtype": "float64", "items": []}

    if write_raw and isinstance(meta.get("raw_memmap_files"), list):
        for info in meta["raw_memmap_files"]:
            tau_idx = int(info["tau_idx"])
            tau_c = float(info["tau_c"])

            src_fid = Path(str(info["fidelity1_path"]))
            src_td  = Path(str(info["trace_distance_path"]))

            dst_fid = raw_dir / f"fid_tau{tau_idx}.dat"
            dst_td  = raw_dir / f"td_tau{tau_idx}.dat"
            # Transfer raw files into run_dir/raw.
            # - copy:    copy from scratch_dir to run_dir/raw
            # - move:    move (delete from scratch_dir)
            # - inplace: assume sources are already under run_dir/raw (or rename into canonical names)
            same_fid = src_fid.resolve() == dst_fid.resolve()
            same_td  = src_td.resolve() == dst_td.resolve()

            # If src==dst, copying would raise SameFileError; treat that as inplace.
            eff_mode = "inplace" if (raw_mode == "copy" and same_fid and same_td) else raw_mode

            if eff_mode == "inplace":
                if not same_fid:
                    shutil.move(str(src_fid), str(dst_fid))
                if not same_td:
                    shutil.move(str(src_td), str(dst_td))

            elif eff_mode == "move":
                shutil.move(str(src_fid), str(dst_fid))
                shutil.move(str(src_td), str(dst_td))

            else:  # copy
                if not same_fid:
                    shutil.copy2(src_fid, dst_fid)
                if not same_td:
                    shutil.copy2(src_td, dst_td)

                # Best-effort cleanup only when sources live inside meta['scratch_dir'] and are distinct from destinations.
                if not keep_scratch_files:
                    try:
                        scratch = Path(str(meta.get("scratch_dir", ""))).resolve()
                        if scratch and scratch != raw_dir.resolve():
                            for src, dst in ((src_fid, dst_fid), (src_td, dst_td)):
                                if src.resolve() != dst.resolve() and scratch in src.resolve().parents:
                                    src.unlink(missing_ok=True)
                    except Exception:
                        pass

            raw_manifest["items"].append(
                {
                    "tau_idx": tau_idx,
                    "tau_c": tau_c,
                    "fidelity1_file": dst_fid.name,
                    "trace_distance_file": dst_td.name,
                    "shape": info.get("shape", [int(meta.get("n_samples", 0))]),
                }
            )

            if not keep_scratch_files:
                # Do not delete by default: scratch might be shared. Only delete if you own it.
                pass

        _safe_json_dump(raw_manifest, raw_dir / "raw_manifest.json")
    else:
        # Serial mode: write arrays as .npy for each tau
        if write_raw:
            items = []
            for tau_key, d in (meta.get("samples") or {}).items():
                if tau_key == "average_flux":
                    continue
                tau_idx = len(items)
                fid = np.asarray(d.get("fidelity1", []), dtype=np.float64)
                td  = np.asarray(d.get("trace_distance", []), dtype=np.float64)
                fid_file = raw_dir / f"fid_tau{tau_idx}.npy"
                td_file  = raw_dir / f"td_tau{tau_idx}.npy"
                np.save(fid_file, fid)
                np.save(td_file, td)
                items.append({"tau_idx": tau_idx, "tau_c": float(tau_key), "fidelity1_file": fid_file.name, "trace_distance_file": td_file.name, "format": "npy"})
            raw_manifest["format"] = "exp1-raw-npy-v1"
            raw_manifest["items"] = items
            _safe_json_dump(raw_manifest, raw_dir / "raw_manifest.json")

    # average_flux baseline
    if "average_flux" in (meta.get("samples") or {}):
        _safe_json_dump(meta["samples"]["average_flux"], raw_dir / "average_flux.json")

    # 3) Summary & preview & report
    write_summary_table_txt(run_dir / "summary.txt", results, meta)
    write_samples_preview_tsv(run_dir / "samples_preview.tsv", meta.get("samples", {}), max_per_tau=preview_max_per_tau)
    # write_default_report_txt(run_dir / "report.txt", results, meta)

    return {
        "config": run_dir / "config.yaml",
        "environment": run_dir / "environment.json",
        "summary": run_dir / "summary.txt",
        "samples_preview": run_dir / "samples_preview.tsv",
        # "report": run_dir / "report.txt",
        "raw_manifest": raw_dir / "raw_manifest.json",
        "raw_dir": raw_dir,
    }


# =============================================================================
# Legacy-compatible save APIs (thin wrappers)
# =============================================================================

# def save_results_exp1(
#     results: List[Dict[str, Any]],
#     meta: Dict[str, Any],
#     run_dir: Union[str, Path],
#     circuit_text: Optional[str] = None,
#     results_filename: str = "results.txt",
#     samples_filename: str = "samples.tsv",
#     circuit_filename: str = "circuit.txt",
#     extra_header_lines: Optional[List[str]] = None,
#     write_samples: bool = True,
#     samples_write_chunk_lines: int = 10000,
#     max_samples_per_tau: Optional[int] = None,
# ) -> Dict[str, Path]:
#     """
#     Backward-compatible writer:
#     - results.txt: aligned table with headers
#     - samples.tsv: (possibly downsampled) raw sample lines
#     - circuit.txt: optional circuit dump
#     """
#     run_dir = _ensure_dir(run_dir)

#     # results table
#     tmp_meta = dict(meta)
#     write_summary_table_txt(run_dir / results_filename, results, tmp_meta, extra_header_lines=extra_header_lines)

#     # samples full/preview
#     samples_path = run_dir / samples_filename
#     if write_samples:
#         # For legacy compatibility, we keep the old schema: tau_c / metric / value.
#         # Here metric names are fidelity1/trace_distance.
#         write_samples_preview_tsv(
#             samples_path,
#             meta.get("samples", {}),
#             chunk_lines=samples_write_chunk_lines,
#             max_per_tau=max_samples_per_tau,
#         )

#     # circuit dump
#     circuit_path: Optional[Path] = None
#     if circuit_text is not None:
#         circuit_path = run_dir / circuit_filename
#         circuit_path.write_text(circuit_text, encoding="utf-8")

#     return {
#         "results": run_dir / results_filename,
#         "samples": samples_path,
#         "circuit": circuit_path,
#     }


# def save_results_exp1_mp(
#     results: List[Dict[str, Any]],
#     meta: Dict[str, Any],
#     run_dir: Union[str, Path],
#     circuit_text: Optional[str] = None,
#     results_filename: str = "results.txt",
#     samples_filename: str = "samples.tsv",
#     circuit_filename: str = "circuit.txt",
#     write_samples: bool = True,
#     samples_write_chunk_lines: int = 10000,
# ) -> Dict[str, Path]:
#     """
#     Legacy mp writer: append parallelism information into the header section.
#     """
#     mp_lines: List[str] = []
#     mp_lines.append("[Parallel]")
#     mp_lines.append(f"n_workers: {meta.get('n_workers', '-')}")
#     mp_lines.append(f"chunk_size: {meta.get('chunk_size', '-')}")
#     mp_lines.append(f"scratch_dir: {meta.get('scratch_dir', '-')}")
#     mp_lines.append(f"mp_start_method: {meta.get('mp_start_method', '-')}")
#     env = meta.get("blas_threads_env", None)
#     mp_lines.append(f"BLAS/OMP env: {env if isinstance(env, dict) else '-'}")

#     return save_results_exp1(
#         results,
#         meta,
#         run_dir,
#         circuit_text=circuit_text,
#         results_filename=results_filename,
#         samples_filename=samples_filename,
#         circuit_filename=circuit_filename,
#         extra_header_lines=mp_lines,
#         write_samples=write_samples,
#         samples_write_chunk_lines=samples_write_chunk_lines,
#         max_samples_per_tau=100,
#     )


# =============================================================================
# Plotting (kept)
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_fidelity_vs_tau_logx(results: List[Dict[str, Any]], title: Optional[str] = None, savepath: Optional[Union[str, Path]] = None):
    tau: List[float] = []
    fid_mean: List[float] = []
    fid_std: List[float] = []

    avg_fid: Optional[float] = None

    for r in results:
        if r.get("tau_c") == "average_flux":
            avg_fid = float(r.get("fidelity1_mean", float("nan")))
            continue
        tau.append(float(r["tau_c"]))
        fid_mean.append(float(r["fidelity1_mean"]))
        fid_std.append(float(r["fidelity1_std"]))

    tau_arr = np.array(tau, dtype=float)
    fid_mean_arr = np.array(fid_mean, dtype=float)
    fid_std_arr = np.array(fid_std, dtype=float)

    idx = np.argsort(tau_arr)
    tau_arr, fid_mean_arr, fid_std_arr = tau_arr[idx], fid_mean_arr[idx], fid_std_arr[idx]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        tau_arr,
        fid_mean_arr,
        yerr=fid_std_arr,
        fmt="o-",
        capsize=3,
        lw=1.8,
        markersize=5,
        label="Segmented flux noise",
    )

    if avg_fid is not None:
        ax.axhline(
            avg_fid,
            color="gray",
            linestyle="--",
            lw=1.5,
            label="Average flux noise",
        )

    ax.set_xscale("log")
    ax.set_xticks(tau_arr)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    ax.minorticks_off()
    ax.set_xlim(float(tau_arr.min()) / 1.2, float(tau_arr.max()) * 1.2)

    ax.set_xlabel(r"$\tau_c$ (ns)")
    ax.set_ylabel("Fidelity1")

    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(Path(savepath), dpi=300)
    plt.show()

# =============================================================================
# Convenience: timestamped run directory + one-call mp run & save
# =============================================================================

def make_timestamped_run_dir(
    exp_name: str = "experiment1",
    base_results_dir: str | Path | None = None,
    ts: str | None = None,
) -> Path:
    """Create results/<exp_name>/<YYYYMMDD-HHMMSS>/ under the project root.

    Notes
    - If base_results_dir is None, we derive project_root from this file's location.
    - The returned directory is created (parents=True).
    """
    if base_results_dir is None:
        # This file typically lives at: <project_root>/experiments/src/experiments/scripts/...
        # parents[4] should be <project_root>.
        try:
            project_root = Path(__file__).resolve().parents[4]
        except Exception:
            project_root = Path.cwd()
        base_results_dir = project_root / "results"

    base_results_dir = Path(base_results_dir)
    base_dir = _ensure_dir(base_results_dir / exp_name)

    ts_str = ts or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _ensure_dir(base_dir / ts_str)
    return run_dir


def run_exp1_mp_and_save(
    cfg: Dict[str, Any],
    *,
    exp_name: str = "experiment1",
    base_results_dir: str | Path | None = None,
    preview_max_per_tau: int = 100,
    raw_mode: str = "inplace",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Path]]:
    """Run Exp1 (mp) and persist artifacts into a timestamped folder.

    This is the recommended entrypoint for large runs:
    1) create a timestamped run_dir
    2) set experiment.scratch_dir -> run_dir/raw (so memmap raw files are produced in-place)
    3) run multiprocessing sweep
    4) save config/env/raw/preview/report into run_dir

    Returns (results, meta, paths).
    """
    cfg_eff = copy.deepcopy(cfg)
    run_dir = make_timestamped_run_dir(exp_name=exp_name, base_results_dir=base_results_dir)

    exp_cfg = cfg_eff.setdefault("experiment", {})
    # Force memmap raw files to land under the timestamped run directory.
    exp_cfg["scratch_dir"] = str(run_dir / "raw")

    results, meta = run_exp1_experiment_mp(cfg_eff)

    # Persist artifacts. Since scratch_dir == run_dir/raw, use raw_mode='inplace' by default.
    paths = save_exp1_run_artifacts(
        results,
        meta,
        run_dir,
        preview_max_per_tau=preview_max_per_tau,
        write_raw=True,
        keep_scratch_files=True,
        raw_mode=raw_mode,
    )
    paths["run_dir"] = Path(run_dir)

    return results, meta, paths
