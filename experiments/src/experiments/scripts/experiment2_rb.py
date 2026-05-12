# experiments/experiment2_rb.py

import cirq, numpy as np
from typing import Dict, List, Tuple, Optional, Iterable, Any
from tqdm.auto import tqdm

import copy
import math
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

# utils
from cirq.noise.utils.timed_circuit_context import assign_timed_circuit_context
from cirq.noise.utils.composite_noise_model import CompositeNoiseModel
from cirq.noise.utils.randomized_benchmarking import (
    _survival_prob_from_dm,
    FitResult,
    fit_rb_decay,
    build_rb1_circuit,
    DEFAULT_RB_GATE_SET,
    clifford_1q_gate_set,
    fill_rb_gate_samplers,
    _run_rb_sweep_custom,
    build_1q_clifford_rb_circuit_compiled,
)
from cirq.noise.utils.metrics import fidelity

# builder（延迟导入模型）
from cirq.noise.utils.noise_builder import (
    _get,
    eval_number,
    make_noise_model
)

from cirq.noise.models.segmented_flux_noise_model import SegmentedFluxNoiseModel

from cirq.noise.utils.compilation_scheme import all_in_one_compile, rb_1q_weak_compile_blockwise

from concurrent.futures import ProcessPoolExecutor, as_completed

def fix_matrix_invalid_values(matrix, label="Matrix", replace_value=0.0):
    """
    检查并修复矩阵中的 NaN 或 inf 值，将其替换为指定的合理值。
    """
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)) or np.any(matrix < 0):
        print(f"[WARN] {label} contains NaN, inf or negative values. Replacing with {replace_value}. Matrix:\n{matrix}")
        matrix = np.nan_to_num(matrix, nan=replace_value, posinf=replace_value, neginf=replace_value)
        matrix[matrix < 0] = replace_value  # 修复负值
    return matrix

def _choose_seed_and_set(cfg: Dict, seed: Optional[int]) -> int:
    """
    返回本次使用的随机种子：
      1) 如果显式传入 seed -> 用它
      2) 否则用 cfg['experiment'].get('seed')（若存在）
      3) 否则自动生成一个 32-bit 随机种子
    同时把最终使用的种子写回 cfg['experiment']['seed']，便于保存到结果文件。
    """
    if seed is not None:
        used = int(seed)
    else:
        exp = cfg.get("experiment", {})
        if "seed" in exp and exp["seed"] is not None:
            used = int(exp["seed"])
        else:
            import secrets
            used = secrets.randbits(32)  # numpy/cirq 都可接受的 int
    cfg.setdefault("experiment", {})["seed"] = int(used)
    return int(used)

def _apply_noise_overrides(noise_cfg: Dict, overrides: Dict[str, bool] | None) -> Dict:
    """
        根据 overrides 覆写 noise_cfg，返回新的 cfg，不修改原 cfg，若 overrides 为空则直接返回原 cfg。
        Args:
            noise_cfg: 原始噪声配置字典
            overrides: 覆写字典，如 {"use_idle": False, "use_ry": True} 等
        Returns: 新的噪声配置字典
    """
    if not overrides:
        return noise_cfg
    cfg = copy.deepcopy(noise_cfg)
    keymap = {
        "use_idle": "idle",
        "use_ry": "ry_gate",
        "use_photon_decay": "photon_decay",
        "use_segmented_flux": "flux_quasistatic",
    }
    for k, v in overrides.items():
        sect = keymap.get(k)
        if sect is None:
            continue
        cfg.setdefault(sect, {})
        if v is False:
            cfg[sect]["enabled"] = False
    return cfg

def _noise_params_snapshot(noise_cfg: Dict) -> Dict[str, Dict[str, float | int | None]]:
    """收集并解析启用中的噪声模型参数（与 builder 默认值对齐）。"""
    out: Dict[str, Dict[str, float | int | None]] = {}

    idle = _get(noise_cfg, "idle", {})
    if idle and idle.get("enabled", True):
        out["Idle"] = {
            "T1":       eval_number(idle.get("T1", 30e-6)),
            "Tphi":     eval_number(idle.get("Tphi", 60e-6)),
            "duration": eval_number(idle.get("duration", 40e-9)),
            "scale":    eval_number(idle.get("scale", 1.0)),
        }

    ry = _get(noise_cfg, "ry_gate", {})
    if ry and ry.get("enabled", True):
        out["RyGate"] = {
            "T1":       eval_number(ry.get("T1", 30e-6)),
            "Tphi":     eval_number(ry.get("Tphi", 60e-6)),
            "p_axis":   eval_number(ry.get("p_axis", 1e-4)),
            "p_plane":  eval_number(ry.get("p_plane", 5e-4)),
            "scale":    eval_number(ry.get("scale", 1.0)),
        }

    pd = _get(noise_cfg, "photon_decay", {})
    if pd and pd.get("enabled", True):
        out["PhotonDecay"] = {
            "chi":      eval_number(pd.get("chi", np.pi * (-2.6e6))),
            "alpha_0":  eval_number(pd.get("alpha_0", np.sqrt(0.8))),
            "kappa":    eval_number(pd.get("kappa", 4e6)),
            "tau_m":    eval_number(pd.get("tau_m", -590e-9)),
            "tau_g":    eval_number(pd.get("tau_g", 10e-9)),
            "scale":    eval_number(pd.get("scale", 1.0)),
        }

    fx = _get(noise_cfg, "flux_quasistatic", {})
    if fx and fx.get("enabled", True):
        out["FluxSegmented"] = {
            "sigma":    eval_number(fx.get("sigma", 0.05)),
            "seed":     fx.get("seed", None),
        }

    return out

def _resolve_n_qubits(circ_cfg: Dict) -> int:
    if "n_qubits" in circ_cfg:
        return int(circ_cfg["n_qubits"])
    if "num_qubits" in circ_cfg:
        return int(circ_cfg["num_qubits"])
    return 1

def _get_rb_idle_params(cfg: Dict) -> Tuple[float, float]:
    """
    从 cfg["circuit"] 读取 RB idle 插入参数。
    返回:
        idle_insert_prob, idle_duration_ns
    """
    circ_cfg = cfg.get("circuit", {})
    idle_insert_prob = float(circ_cfg.get("idle_insert_prob", 0.0))
    idle_duration_ns = float(circ_cfg.get("idle_duration_ns", 20.0))
    return idle_insert_prob, idle_duration_ns

def _is_standard_1q_clifford_gate_set(rb_cfg: Dict) -> bool:
    """
    判断当前 rb.gate_set 是否表示“标准 1Q Clifford RB”。
    """
    gate_set_cfg = rb_cfg.get("gate_set", None)

    if gate_set_cfg is None:
        return True

    if isinstance(gate_set_cfg, str):
        alias = gate_set_cfg.strip().lower()
        return alias in (
            "clifford_1q",
            "1q_clifford",
            "single_qubit_clifford",
        )

    return False

def _get_rb_compiler(rb_cfg: Dict):
    """
    根据 rb.compile_mode 返回 compiler callable 或 None。
    """
    mode = str(rb_cfg.get("compile_mode", "weak")).lower()

    if mode == "none":
        return None
    if mode == "weak":
        return rb_1q_weak_compile_blockwise
    if mode == "all":
        return all_in_one_compile

    raise ValueError(f"Unknown RB compile_mode: {mode}")

def _compile_rb_circuit(c0: cirq.Circuit, cfg: Dict) -> cirq.Circuit:
    """
    RB 线路编译策略：
      - weak: 保留随机 Clifford 结构，优先用于 RB
      - none: 不编译
      - all: 仅调试时使用，不推荐作为正式 RB
    """
    rb_cfg = cfg.get("rb", {})
    mode = str(rb_cfg.get("compile_mode", "weak")).lower()

    if mode == "none":
        return c0
    if mode == "weak":
        return rb_1q_weak_compile_blockwise(c0)
    if mode == "all":
        return all_in_one_compile(c0)

    raise ValueError(f"Unknown RB compile_mode: {mode}")

def _make_condition_cfg(cfg: Dict, cond: Dict[str, Any]) -> Dict:
    cfg2 = copy.deepcopy(cfg)

    # ---- timing / tau_c ----
    cfg2.setdefault("timing", {})
    if "tau_c" in cond and cond["tau_c"] is not None:
        cfg2["timing"]["tau_c"] = float(cond["tau_c"])

    # ---- RB circuit idle insertion ----
    cfg2.setdefault("circuit", {})
    if "idle_insert_prob" in cond:
        cfg2["circuit"]["idle_insert_prob"] = float(cond["idle_insert_prob"])
    if "idle_duration_ns" in cond:
        cfg2["circuit"]["idle_duration_ns"] = float(cond["idle_duration_ns"])

    # ---- flux sampling mode ----
    cfg2.setdefault("noise", {})
    cfg2["noise"].setdefault("flux_quasistatic", {})
    fx = cfg2["noise"]["flux_quasistatic"]
    fx["enabled"] = True
    fx.setdefault("sampling", {})
    samp = fx["sampling"]

    if "flux_mode" in cond and cond["flux_mode"] is not None:
        samp["mode"] = str(cond["flux_mode"])

    if "rho" in cond:
        if cond["rho"] is None:
            samp.pop("rho", None)
        else:
            samp["rho"] = float(cond["rho"])

    if "sample_stride_segments" in cond:
        samp["sample_stride_segments"] = cond["sample_stride_segments"]

    if "bounds" in cond and cond["bounds"] is not None:
        samp["bounds"] = copy.deepcopy(cond["bounds"])

    return cfg2

def _resolve_rb_gate_set(rb_cfg: Dict):
    gate_set_cfg = rb_cfg.get("gate_set", None)

    # 默认：1Q Clifford
    if gate_set_cfg is None:
        return clifford_1q_gate_set()

    # 允许字符串别名
    if isinstance(gate_set_cfg, str):
        alias = gate_set_cfg.strip().lower()

        if alias in ("clifford_1q", "1q_clifford", "single_qubit_clifford"):
            return clifford_1q_gate_set()

        raise ValueError(f"Unknown RB gate_set alias: {gate_set_cfg}")

    # 允许显式 sampler 配置列表
    return fill_rb_gate_samplers(gate_set_cfg)

def _build_rb_circuit_for_exp2(
    qs,
    m: int,
    cfg: Dict,
    seed_seq: int,
):
    """
    统一的 Exp2 RB 电路构造入口。

    逻辑：
      1) 默认 / 推荐：标准 1Q Clifford RB
         - 使用 build_1q_clifford_rb_circuit_compiled
         - 支持从 YAML 解析 idle_insert_prob / idle_duration_ns
         - 支持 compile_mode = none / weak / all

      2) 自定义 rb.gate_set:
         - 保留 generic 功能
         - 当前仅支持“无随机 idle 插入”的 generic 路径
         - 若你以后真要对 custom gate_set 也做随机 idle，再单独扩展
    """
    rb_cfg = cfg.get("rb", {})
    idle_insert_prob, idle_duration_ns = _get_rb_idle_params(cfg)
    compiler = _get_rb_compiler(rb_cfg)

    # ----- 默认 / 推荐路径：标准 1Q Clifford RB -----
    if _is_standard_1q_clifford_gate_set(rb_cfg):
        abstract_c, c, meta_rb = build_1q_clifford_rb_circuit_compiled(
            m=int(m),
            qubit=qs[0],
            seed=int(seed_seq),
            measure=False,
            compiler=compiler,
            idle_insert_prob=float(idle_insert_prob),
            idle_duration_ns=float(idle_duration_ns),
        )
        return abstract_c, c, meta_rb

    # ----- generic gate_set 路径：保留功能，但先做最稳妥版本 -----
    gate_set = _resolve_rb_gate_set(rb_cfg)

    c0 = build_rb1_circuit(
        qs,
        depth=int(m),
        gate_set=gate_set,
        seed=int(seed_seq),
        measure=False,
    )

    # 这里不要默默忽略 idle，不然最容易出假实验
    if idle_insert_prob > 0.0:
        raise NotImplementedError(
            "Random idle insertion is currently supported only for "
            "standard 1Q Clifford RB. For custom rb.gate_set, please "
            "set idle_insert_prob = 0.0 for now."
        )

    if compiler is None:
        c = c0
    else:
        c = compiler(c0)

    meta_rb = {
        "gate_set_mode": "custom",
        "idle_insert_prob": float(idle_insert_prob),
        "idle_duration_ns": float(idle_duration_ns),
    }
    return c0, c, meta_rb

def _estimate_op_duration_ns(op: cirq.Operation, t1_ns: float, t2_ns: float) -> float:
    gate = getattr(op, "gate", None)

    # 显式 wait
    if isinstance(gate, cirq.WaitGate):
        dur = gate.duration
        try:
            return float(dur.total_nanos())
        except Exception:
            pass
        try:
            return float(cirq.Duration(dur).total_nanos())
        except Exception:
            return 0.0

    # 测量默认不计入 RB 主体时长
    if isinstance(gate, cirq.MeasurementGate):
        return 0.0

    # 两比特门
    if len(op.qubits) == 2:
        return float(t2_ns)

    # 单比特门
    if len(op.qubits) == 1:
        # 这里保持与你当前 exp2 假设一致：1Q native gate 统一记 t1
        return float(t1_ns)

    return 0.0


def _estimate_circuit_total_duration_ns(c: cirq.Circuit, timing_cfg: Dict[str, Any]) -> float:
    t1_ns = float(timing_cfg.get("t1", 20.0))
    t2_ns = float(timing_cfg.get("t2", 40.0))

    total = 0.0
    for moment in c:
        if not moment.operations:
            continue
        moment_dur = 0.0
        for op in moment.operations:
            moment_dur = max(moment_dur, _estimate_op_duration_ns(op, t1_ns, t2_ns))
        total += moment_dur
    return float(total)


def _prepare_seq_seeds_by_m(m_list: List[int], n_seq: int, seed: int) -> Dict[int, List[int]]:
    rng = np.random.RandomState(int(seed))
    out: Dict[int, List[int]] = {}
    for m in m_list:
        out[int(m)] = [int(rng.randint(2**31)) for _ in range(int(n_seq))]
    return out


def _prepare_m_duration_upper_bounds(
    cfg: Dict,
    m_list: List[int],
    seq_seeds_by_m: Dict[int, List[int]],
    n_seq_for_bound: int = 30,
) -> Dict[int, float]:
    """
    对每个 m，预先估计一个“该 m 下线路总时长上界”。

    优化版：
      - 只抽前 n_seq_for_bound 条共享 sequence seeds 来估上界
      - 加 tqdm 与简短日志，避免长时间静默
    """
    circ_cfg = cfg["circuit"]
    n_qubits = _resolve_n_qubits(circ_cfg)
    if n_qubits != 1:
        raise ValueError(f"Current exp2 grid runner only supports 1Q RB, got n_qubits={n_qubits}")

    qs = cirq.LineQubit.range(n_qubits)

    out: Dict[int, float] = {}
    m_iter = tqdm(m_list, desc="prepare duration upper bounds", unit="m")

    for m in m_iter:
        max_dur = 0.0

        all_seeds = seq_seeds_by_m[int(m)]
        use_n = min(int(n_seq_for_bound), len(all_seeds))
        seeds = all_seeds[:use_n]

        print(
            f"[prep] m={m}, estimate_nseq={use_n}",
            flush=True,
        )

        for seed_seq in seeds:
            _abstract_c, c, _meta_rb = _build_rb_circuit_for_exp2(
                qs=qs,
                m=int(m),
                cfg=cfg,
                seed_seq=int(seed_seq),
            )
            dur_ns = _estimate_circuit_total_duration_ns(c, cfg["timing"])
            if dur_ns > max_dur:
                max_dur = dur_ns

        out[int(m)] = float(max_dur)

        print(
            f"[prep] m={m} done, duration_upper_ns={max_dur:.1f}",
            flush=True,
        )

    return out


def _first_saturated_tau_idx(
    tau_c_list: List[float],
    duration_upper_ns: float,
) -> Optional[int]:
    """
    返回首个满足 tau_c >= duration_upper_ns 的下标。
    若不存在则返回 None。
    """
    for i, tau in enumerate(tau_c_list):
        if float(tau) >= float(duration_upper_ns):
            return int(i)
    return None


def _run_one_rb1_point_worker(
    cfg_cond: Dict,
    m: int,
    seq_seeds: List[int],
    seed_base: int,
    noise_overrides: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    单个 (tau_c, m) worker。
    返回这个点上所有 seq 的 survival/fidelity 样本及统计量。
    """
    exp = cfg_cond["experiment"]
    circ_cfg = cfg_cond["circuit"]
    timing_cfg = cfg_cond["timing"]
    noise_cfg = cfg_cond["noise"]

    metric = str(exp.get("metric", "survival")).lower()
    if metric not in ("survival", "fidelity"):
        raise ValueError(f"Unsupported RB metric: {metric}")

    n_qubits = _resolve_n_qubits(circ_cfg)
    if n_qubits != 1:
        raise ValueError(f"Current exp2 grid runner only supports 1Q RB, got n_qubits={n_qubits}")

    qs = cirq.LineQubit.range(n_qubits)
    ncfg_eff = _apply_noise_overrides(noise_cfg, noise_overrides)

    vals = []
    probe_rows = []

    log_cfg = cfg_cond.get("logging", {}).get("fid_probe", {}) if isinstance(cfg_cond.get("logging"), dict) else {}
    probe_on = bool(log_cfg.get("enabled", False))
    probe_k = int(log_cfg.get("per_m", 0)) if probe_on else 0

    for seq_id, seed_seq in enumerate(seq_seeds):
        _abstract_c, c, meta_rb = _build_rb_circuit_for_exp2(
            qs=qs,
            m=int(m),
            cfg=cfg_cond,
            seed_seq=int(seed_seq),
        )

        noise_model, _meta = make_noise_model(
            c,
            timing_cfg,
            ncfg_eff,
            seed_base=int(seed_base),
            tag_a=int(m),
            tag_b=int(seq_id),
        )

        sim_noisy = cirq.DensityMatrixSimulator()
        c_noisy = c.with_noise(noise_model)
        rho_noisy = sim_noisy.simulate(c_noisy).final_density_matrix

        if metric == "survival":
            y = _survival_prob_from_dm(rho_noisy)
        else:
            sim_ideal = cirq.DensityMatrixSimulator()
            rho_ideal = sim_ideal.simulate(c).final_density_matrix
            y = float(fidelity(rho_ideal, rho_noisy))

        vals.append(float(y))

        if probe_on and seq_id < probe_k:
            sim_ideal = cirq.DensityMatrixSimulator()
            rho_ideal = sim_ideal.simulate(c).final_density_matrix
            try:
                fid = float(fidelity(rho_ideal, rho_noisy))
            except Exception:
                fid = float("nan")
            P_surv = float(np.real(rho_noisy[0, 0]))
            probe_rows.append((
                int(m),
                int(seq_id),
                P_surv,
                fid,
                int(seed_seq),
                str(meta_rb.get("gate_set_mode", "standard_1q_clifford")),
            ))

    vals = np.asarray(vals, dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    sem = float(std / np.sqrt(max(1, len(vals))))

    tau_c = float(cfg_cond["timing"]["tau_c"])
    return {
        "tau_c": tau_c,
        "m": int(m),
        "metric": metric,
        "seq_vals": vals,
        "mean": mean,
        "std": std,
        "sem": sem,
        "probe_rows": probe_rows,
    }


def _assemble_rb_result_from_grid(
    cfg: Dict,
    point_results_by_m: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    从某个 tau_c 下所有 m 的 point results 组装成与 _run_one_rb1_sweep 相同风格的结果。
    """
    exp = cfg["experiment"]
    circ_cfg = cfg["circuit"]
    metric = str(exp.get("metric", "survival")).lower()

    m_list = list(exp["m_list"])
    n_qubits = _resolve_n_qubits(circ_cfg)

    seq_vals_all = []
    mean_y = []
    probe_rows_all = []

    for m in m_list:
        pr = point_results_by_m[int(m)]
        vals = np.asarray(pr["seq_vals"], dtype=float)
        seq_vals_all.append(vals)
        mean_y.append(float(pr["mean"]))
        probe_rows_all.extend(pr.get("probe_rows", []))

    seq_vals_all = np.asarray(seq_vals_all, dtype=float)
    obs_stats = _compute_obs_stats_from_seq_vals(seq_vals_all)

    y_std = [float(x) for x in obs_stats["sem"]]
    d = 2 ** n_qubits
    fit = fit_rb_decay(
        m_list,
        mean_y,
        d=d,
        fit_curve=str(exp.get("fit_curve", "exponential")),
        y_std=y_std,
        lock_B=True,
        add_m0=True,
    )

    log_cfg = cfg.get("logging", {}).get("fid_probe", {}) if isinstance(cfg.get("logging"), dict) else {}
    probe_info = None
    if probe_rows_all:
        probe_info = {
            "rows": probe_rows_all,
            "outfile": log_cfg.get("outfile", "fid_probe.csv"),
            "header": ["m", "seq_id", "P_survival", "fidelity_to_ideal", "seed_seq", "gate_set_mode"],
        }

    out = {
        "metric": metric,
        "m_list": m_list,
        "Ybar": mean_y,
        "Pbar": mean_y if metric == "survival" else None,
        "seq_vals": seq_vals_all,
        "obs_stats": {
            "mean": obs_stats["mean"],
            "std": obs_stats["std"],
            "sem": obs_stats["sem"],
        },
        "fit": {
            "model": fit.model,
            "params": fit.params,
            "param_stderr": fit.param_stderr,
        },
        "probe": probe_info,
    }

    if metric == "survival":
        out["EPC"] = fit.EPC
        out["EPC_stderr"] = fit.EPC_stderr
    else:
        out["EPC"] = fit.EPC
        out["EPC_stderr"] = fit.EPC_stderr
        out["note"] = "metric=fidelity: EPC is fit-derived only and should not be interpreted as standard RB EPC"

    out["diagnostics"] = _compute_basic_fit_diagnostics(out)
    return out


def run_exp2_rb_decay_parallel_grid(cfg: Dict, seed: Optional[int] = None) -> Dict:
    """
    标准 RB decay 的并行 grid 版本：
      - 任务粒度 = (tau_c, m)
      - 同一个 m 下所有 tau_c 共用同一批 sequence seeds
      - 若某个 m 在 tau_c >= total_duration_upper 后进入单 segment 区，则后续更大 tau_c 直接复用该结果
    """
    used_seed = _choose_seed_and_set(cfg, seed)
    exp = cfg["experiment"]

    conditions = list(exp.get("conditions", []))
    if not conditions:
        raise ValueError("experiment.conditions is empty for exp2 RB decay")

    m_list = [int(m) for m in exp["m_list"]]
    n_seq = int(exp["n_sequences"])
    n_workers = int(exp.get("n_workers", 1))

    # 1) 先保证 tau_c 单调升序；图三/图四都建议这样
    conditions = sorted(conditions, key=lambda d: float(d["tau_c"]))
    tau_c_list = [float(c["tau_c"]) for c in conditions]

    # 2) 为每个 m 预生成共享 sequence seeds
    seq_seeds_by_m = _prepare_seq_seeds_by_m(m_list, n_seq=n_seq, seed=used_seed)

    # 3) 估计每个 m 的“最大线路总时长”，用于 saturation skip
    bound_estimate_nseq = int(exp.get("bound_estimate_nseq", 30))

    print(
        f"[INFO] bound_estimate_nseq = {bound_estimate_nseq}",
        flush=True,
    )

    duration_upper_by_m = _prepare_m_duration_upper_bounds(
        cfg,
        m_list,
        seq_seeds_by_m,
        n_seq_for_bound=bound_estimate_nseq,
    )

    print("[INFO] duration upper bounds by m:", flush=True)
    for m in m_list:
        print(f"  m={m:<4d} -> duration_upper_ns={duration_upper_by_m[int(m)]:.1f}", flush=True)

    # 4) 先生成真正需要跑的 tasks
    tasks = []
    saturation_src_tau_idx_by_m: Dict[int, Optional[int]] = {}

    for m in m_list:
        sat_idx = _first_saturated_tau_idx(tau_c_list, duration_upper_by_m[int(m)])
        saturation_src_tau_idx_by_m[int(m)] = sat_idx

        last_idx_to_run = len(tau_c_list) - 1 if sat_idx is None else sat_idx
        for tau_idx in range(last_idx_to_run + 1):
            cond = copy.deepcopy(conditions[tau_idx])
            cfg_cond = _make_condition_cfg(cfg, cond)
            tasks.append({
                "tau_idx": int(tau_idx),
                "tau_c": float(cond["tau_c"]),
                "m": int(m),
                "cfg_cond": cfg_cond,
                "seq_seeds": seq_seeds_by_m[int(m)],
                "seed_base": int(used_seed),
            })

    print("[INFO] first saturation tau index by m:", flush=True)
    for m in m_list:
        sat_idx = saturation_src_tau_idx_by_m[int(m)]
        if sat_idx is None:
            print(f"  m={m:<4d} -> no saturation within tau grid", flush=True)
        else:
            print(
                f"  m={m:<4d} -> sat_idx={sat_idx}, tau_c={tau_c_list[sat_idx]}",
                flush=True,
            )

    # 5) 并行执行
    point_table: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = []
        for task in tasks:
            fut = ex.submit(
                _run_one_rb1_point_worker,
                task["cfg_cond"],
                task["m"],
                task["seq_seeds"],
                task["seed_base"],
                None,
            )
            futs.append((task, fut))

        for task, fut in tqdm(futs, desc="RB grid", unit="task"):
            pr = fut.result()
            point_table[(int(task["tau_idx"]), int(task["m"]))] = pr

    # 6) 对进入 saturation 的 m，把后续 tau 直接复制首个饱和值
    for m in m_list:
        sat_idx = saturation_src_tau_idx_by_m[int(m)]
        if sat_idx is None:
            continue
        src = point_table[(int(sat_idx), int(m))]
        for tau_idx in range(int(sat_idx) + 1, len(tau_c_list)):
            cloned = {
                "tau_c": float(tau_c_list[tau_idx]),
                "m": int(m),
                "metric": src["metric"],
                "seq_vals": np.asarray(src["seq_vals"], dtype=float).copy(),
                "mean": float(src["mean"]),
                "std": float(src["std"]),
                "sem": float(src["sem"]),
                "probe_rows": list(src.get("probe_rows", [])),
                "copied_from_tau_idx": int(sat_idx),
                "copied_from_tau_c": float(tau_c_list[sat_idx]),
            }
            point_table[(int(tau_idx), int(m))] = cloned

    # 7) 组装成原 run_exp2_rb_decay 的输出格式
    out = {
        "seed": used_seed,
        "metric": str(exp.get("metric", "survival")).lower(),
        "conditions": {},
    }

    for tau_idx, cond in enumerate(conditions):
        by_m = {int(m): point_table[(int(tau_idx), int(m))] for m in m_list}
        res = _assemble_rb_result_from_grid(_make_condition_cfg(cfg, cond), by_m)

        out["conditions"][str(cond["label"])] = {
            "condition": copy.deepcopy(cond),
            "result": res,
            "grid_meta": {
                "tau_idx": int(tau_idx),
                "tau_c": float(cond["tau_c"]),
            },
        }

    return out


def _run_one_rb1_sweep(
    cfg: Dict,
    seed: int,
    noise_overrides: Dict[str, bool] | None = None,
) -> Dict:
    """
    公共 RB sweep 核心函数：
      - 扫 m_list
      - 每个 m 跑 n_sequences 条随机 1Q Clifford RB 序列
      - 支持 metric = survival / fidelity
      - 可用于：
          1) 标准 RB decay 条件扫描
          2) 逐类消融分析
    """
    exp = cfg["experiment"]
    circ_cfg = cfg["circuit"]
    timing_cfg = cfg["timing"]
    noise_cfg = cfg["noise"]
    rb_cfg = cfg.get("rb", {})

    metric = str(exp.get("metric", "survival")).lower()
    if metric not in ("survival", "fidelity"):
        raise ValueError(f"Unsupported RB metric: {metric}")

    m_list = list(exp["m_list"])
    n_seq = int(exp["n_sequences"])

    n_qubits = _resolve_n_qubits(circ_cfg)
    if n_qubits != 1:
        raise ValueError(f"Current exp2 RB runner only supports 1Q RB, got n_qubits={n_qubits}")

    qs = cirq.LineQubit.range(n_qubits)

    rng_seq = np.random.RandomState(int(seed))

    # gate set

    # gate_set_cfg = rb_cfg.get("gate_set", DEFAULT_RB_GATE_SET)
    # gate_set = fill_rb_gate_samplers(gate_set_cfg)

    # probe logging
    log_cfg  = cfg.get("logging", {}).get("fid_probe", {}) if isinstance(cfg.get("logging"), dict) else {}
    probe_on = bool(log_cfg.get("enabled", False))
    probe_k  = int(log_cfg.get("per_m", 0)) if probe_on else 0
    probe_rows = []

    mean_y = []
    seq_vals_all = []

    desc_suffix = "none" if not noise_overrides else ",".join([k for k, v in noise_overrides.items() if v is False])
    m_iter = tqdm(m_list, desc=f"RB sweep | off={desc_suffix}", unit="m")

    for m in m_iter:
        vals = []
        seq_id_iter = tqdm(range(n_seq), desc=f"m={m}", leave=False, unit="seq")

        for seq_id in seq_id_iter:
            seed_seq = int(rng_seq.randint(2**31))

            # 1) build one RB circuit according to cfg
            abstract_c, c, meta_rb = _build_rb_circuit_for_exp2(
                qs=qs,
                m=int(m),
                cfg=cfg,
                seed_seq=seed_seq,
            )

            # # 2) sanity check: ideal recovery
            # if seq_id == 0:
            #     rho_ideal_check = cirq.DensityMatrixSimulator().simulate(c).final_density_matrix
            #     p0_ideal = float(np.real(rho_ideal_check[0, 0]))
            #     print(f"[DEBUG] ideal survival at m={m}: {p0_ideal:.6f}")

            # 3) build noise model
            ncfg_eff = _apply_noise_overrides(noise_cfg, noise_overrides)
            noise_model, _meta = make_noise_model(
                c,
                timing_cfg,
                ncfg_eff,
                seed_base=int(seed),
                tag_a=int(m),
                tag_b=int(seq_id),
            )

            # 4) noisy simulate
            sim_noisy = cirq.DensityMatrixSimulator()
            c_noisy = c.with_noise(noise_model)
            rho_noisy = sim_noisy.simulate(c_noisy).final_density_matrix

            # 5) chosen observable
            if metric == "survival":
                y = _survival_prob_from_dm(rho_noisy)
            else:
                sim_ideal = cirq.DensityMatrixSimulator()
                rho_ideal = sim_ideal.simulate(c).final_density_matrix
                y = float(fidelity(rho_ideal, rho_noisy))

            vals.append(float(y))

            # probe
            if probe_on and seq_id < probe_k:
                sim_ideal = cirq.DensityMatrixSimulator()
                rho_ideal = sim_ideal.simulate(c).final_density_matrix
                try:
                    fid = float(fidelity(rho_ideal, rho_noisy))
                except Exception:
                    fid = float("nan")
                P_surv = float(np.real(rho_noisy[0, 0]))
                probe_rows.append((
                    int(m),
                    int(seq_id),
                    P_surv,
                    fid,
                    int(seed_seq),
                    str(meta_rb.get("gate_set_mode", "standard_1q_clifford")),
                ))

        vals = np.asarray(vals, dtype=float)
        mean_y.append(float(np.mean(vals)))
        seq_vals_all.append(vals)

    # ---- fit ----
    d = 2 ** n_qubits
    y_std = [float(np.std(v, ddof=1) / np.sqrt(max(1, len(v)))) for v in seq_vals_all]

    fit = fit_rb_decay(
        m_list,
        mean_y,
        d=d,
        fit_curve=str(exp.get("fit_curve", "exponential")),
        y_std=y_std,
        lock_B=True,
        add_m0=True,
    )

    probe_info = None
    if probe_on:
        probe_info = {
            "rows": probe_rows,
            "outfile": log_cfg.get("outfile", "fid_probe.csv"),
            "header": ["m", "seq_id", "P_survival", "fidelity_to_ideal", "seed_seq", "gate_set_mode"],
        }

    obs_stats = _compute_obs_stats_from_seq_vals(np.asarray(seq_vals_all, dtype=float))

    out = {
        "metric": metric,
        "m_list": m_list,
        "Ybar": mean_y,
        "Pbar": mean_y if metric == "survival" else None,
        "seq_vals": np.asarray(seq_vals_all, dtype=float),

        # 新增：观测统计量
        "obs_stats": {
            "mean": obs_stats["mean"],
            "std": obs_stats["std"],
            "sem": obs_stats["sem"],
        },

        "fit": {
            "model": fit.model,
            "params": fit.params,
            "param_stderr": fit.param_stderr,
        },
        "probe": probe_info,
    }

    # EPC 只有 survival 才作为“标准 RB 含义”来解释
    if metric == "survival":
        out["EPC"] = fit.EPC
        out["EPC_stderr"] = fit.EPC_stderr
    else:
        out["EPC"] = fit.EPC
        out["EPC_stderr"] = fit.EPC_stderr
        out["note"] = "metric=fidelity: EPC is fit-derived only and should not be interpreted as standard RB EPC"
        
    out["diagnostics"] = _compute_basic_fit_diagnostics(out)

    return out

def run_exp2_rb_decay(cfg: Dict, seed: Optional[int] = None) -> Dict:
    """
    Exp2.1: 标准 RB decay 条件扫描

    默认优先走 grid 并行版本：
      - 任务粒度 = (tau_c, m)
      - 同一 m 下共享 sequence seeds
      - 对进入单-segment区的更大 tau_c 做自动跳过/复用

    若 experiment.parallel_mode 显式设为 "serial"，
    则退回旧版串行逻辑。
    """
    parallel_mode = str(cfg.get("experiment", {}).get("parallel_mode", "grid_tau_m")).lower()

    if parallel_mode == "serial":
        used_seed = _choose_seed_and_set(cfg, seed)
        exp = cfg["experiment"]

        conditions = list(exp.get("conditions", []))
        if not conditions:
            raise ValueError("experiment.conditions is empty for exp2 RB decay")

        out = {
            "seed": used_seed,
            "metric": str(exp.get("metric", "survival")).lower(),
            "conditions": {},
        }

        for cond in conditions:
            label = str(cond["label"])
            cfg_cond = _make_condition_cfg(cfg, cond)
            res = _run_one_rb1_sweep(cfg_cond, seed=used_seed, noise_overrides=None)

            out["conditions"][label] = {
                "condition": copy.deepcopy(cond),
                "result": res,
            }

        return out

    if parallel_mode == "grid_tau_m":
        return run_exp2_rb_decay_parallel_grid(cfg, seed=seed)

    raise ValueError(f"Unknown experiment.parallel_mode: {parallel_mode}")


def run_exp2_ablation(
        cfg: Dict,
        seed: Optional[int] = None,
        base_condition: Optional[Dict[str, Any]] = None,
        ablation_keys: Optional[List[str]] = None,
    ) -> Dict:
        """
        Exp2.2: RB 逐类消融
        先在一个 base_condition 上跑 all-on，再逐个关闭噪声模型，计算 ΔEPC / 占比。
        """
        used_seed = _choose_seed_and_set(cfg, seed)

        # 1) 优先级：显式传入 > yaml中的 base_condition > fallback
        if base_condition is None:
            base_condition = copy.deepcopy(cfg.get("base_condition"))

        if base_condition is None:
            base_condition = {
                "label": "default_condition",
                "tau_c": cfg.get("timing", {}).get("tau_c", 400.0),
                "flux_mode": cfg.get("noise", {}).get("flux_quasistatic", {}).get("sampling", {}).get("mode", "local_correlated"),
                "rho": cfg.get("noise", {}).get("flux_quasistatic", {}).get("sampling", {}).get("rho", None),
                "idle_insert_prob": cfg.get("circuit", {}).get("idle_insert_prob", 0.0),
                "idle_duration_ns": cfg.get("circuit", {}).get("idle_duration_ns", 20.0),
            }

        # 2) 优先级：显式传入 > yaml中的 experiment.ablation_keys > fallback
        if ablation_keys is None:
            ablation_keys = list(
                cfg.get("experiment", {}).get(
                    "ablation_keys",
                    ["use_idle", "use_photon_decay", "use_ry", "use_segmented_flux"]
                )
            )

        cfg_base = _make_condition_cfg(cfg, base_condition)

        # all-on
        res_all = _run_one_rb1_sweep(cfg_base, seed=used_seed, noise_overrides=None)

        # ablations
        ablations = {}
        for key in ablation_keys:
            res = _run_one_rb1_sweep(cfg_base, seed=used_seed, noise_overrides={key: False})
            ablations[key] = res

        # contribution via EPC
        r_all = res_all.get("EPC", None)
        deltas = {}
        for k, v in ablations.items():
            epc_i = v.get("EPC", None)
            if r_all is not None and epc_i is not None:
                deltas[k] = float(r_all - epc_i)
            else:
                deltas[k] = None

        S = sum(max(0.0, x) for x in deltas.values() if x is not None) or 1.0
        shares = {
            k: (max(0.0, v) / S if v is not None else 0.0)
            for k, v in deltas.items()
        }

        return {
            "seed": used_seed,
            "metric": cfg_base["experiment"].get("metric", "survival"),
            "condition": copy.deepcopy(base_condition),
            "all": res_all,
            "ablations": ablations,
            "deltas": deltas,
            "shares": shares,
        }


def _save_rb_fit_plot_single(
    rb_result: Dict,
    out_dir: Path,
    filename: str = "rb_fit.png",
) -> Path:
    """
    对单个 RB result 画拟合图。
    rb_result 应该是 _run_one_rb1_sweep(...) 的返回值。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    metric = rb_result.get("metric", "survival").lower()
    m_obs = np.asarray(rb_result["m_list"], dtype=float)
    y_obs = np.asarray(rb_result["Ybar"], dtype=float)
    fit   = rb_result["fit"]

    obs_stats = rb_result.get("obs_stats", {}) or {}
    y_sem = None
    if "sem" in obs_stats:
        y_sem = np.asarray(obs_stats["sem"], dtype=float)

    model = fit.get("model", "exponential")
    params = fit.get("params", {})
    stderr = fit.get("param_stderr", {})

    m_dense = np.linspace(0.0, float(max([0.0] + list(m_obs))), 200)

    if model == "exponential":
        A = params.get("A", 0.9)
        p = params.get("p", 0.99)
        B = params.get("B", 0.01)
        y_fit = A * p**m_dense + B

        A_se = stderr.get("A")
        p_se = stderr.get("p")
        B_se = stderr.get("B")

        sigma_terms = []
        if A_se is not None:
            sigma_terms.append((p**m_dense * A_se) ** 2)
        if p_se is not None and p > 0:
            sigma_terms.append((A * m_dense * p**(m_dense - 1) * p_se) ** 2)
        if B_se is not None:
            sigma_terms.append(np.full_like(m_dense, B_se**2))

        y_hat_obs = A * p**m_obs + B
        legend_str = f"A={A:.3f}, p={p:.5f}, B={B:.5f}"

    else:
        raise ValueError(f"Currently only exponential fit is supported in save plot helper, got {model}")

    resid = y_obs - y_hat_obs

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 6.2), dpi=140,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    obs_label = "Observed mean ± SEM" if metric == "survival" else "Observed mean ± SEM"
    ln, = ax1.plot(m_dense, y_fit, lw=2.0, label="Fit")

    if y_sem is not None and len(y_sem) == len(y_obs):
        ax1.errorbar(
            m_obs,
            y_obs,
            yerr=y_sem,
            fmt="o",
            markersize=4.5,
            capsize=3,
            linewidth=1.0,
            label=obs_label,
        )
    else:
        ax1.scatter(m_obs, y_obs, s=28, label="Observed mean")

    if sigma_terms:
        sigma_y = np.sqrt(np.sum(sigma_terms, axis=0))
        y_lo = y_fit - 1.96 * sigma_y
        y_hi = y_fit + 1.96 * sigma_y
        ax1.fill_between(m_dense, y_lo, y_hi, alpha=0.18, color=ln.get_color(), label="95% CI (approx)")

    ax1.set_xlabel("m (sequence depth)")
    ax1.set_ylabel("Mean survival probability" if metric == "survival" else "Mean fidelity")
    ax1.set_title(f"RB Fit: {model}")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(title=legend_str, loc="best")

    ax2.axhline(0.0, color="k", lw=1)

    if y_sem is not None and len(y_sem) == len(resid):
        ax2.errorbar(
            m_obs,
            resid,
            yerr=y_sem,
            fmt="o",
            markersize=4.0,
            capsize=3,
            linewidth=1.0,
        )
    else:
        ax2.scatter(m_obs, resid, s=22)

    ax2.set_xlabel("m")
    ax2.set_ylabel("resid")
    ax2.grid(True, linestyle="--", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(out_dir) / filename
    fig.tight_layout()
    fig.savefig(img_path, bbox_inches="tight")
    plt.close(fig)
    return img_path

def _compute_obs_stats_from_seq_vals(seq_vals: np.ndarray) -> Dict[str, np.ndarray]:
    """
    给定 shape = (n_m, n_seq) 的 seq_vals，计算每个 m 上的统计量。
    返回:
      mean, std, sem
    """
    arr = np.asarray(seq_vals, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"seq_vals must be 2D, got shape={arr.shape}")

    n_seq = arr.shape[1]
    mean = np.mean(arr, axis=1)
    if n_seq >= 2:
        std = np.std(arr, axis=1, ddof=1)
    else:
        std = np.zeros(arr.shape[0], dtype=float)
    sem = std / np.sqrt(max(1, n_seq))

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
    }

def _save_rb_raw_npz(
    res: Dict,
    npz_path: Path,
    extra_meta: Optional[Dict[str, Any]] = None,
):
    """
    保存单个 RB 结果的原始数据到 .npz
    主要保存：
      - m_list
      - Ybar / Pbar
      - 每个 m 下的所有 sequence 原始值 seq_vals
      - EPC / EPC_stderr
      - 少量 meta
    """
    payload = {
        "metric": np.asarray(str(res.get("metric", "survival"))),
        "m_list": np.asarray(res["m_list"], dtype=int),
        "Ybar": np.asarray(res["Ybar"], dtype=float),
        "seq_vals": np.asarray(res["seq_vals"], dtype=float),
        "EPC": np.asarray([res.get("EPC", np.nan)], dtype=float),
        "EPC_stderr": np.asarray([res.get("EPC_stderr", np.nan)], dtype=float),
    }

    obs_stats = res.get("obs_stats", None)
    if obs_stats is not None:
        if "mean" in obs_stats:
            payload["obs_mean"] = np.asarray(obs_stats["mean"], dtype=float)
        if "std" in obs_stats:
            payload["obs_std"] = np.asarray(obs_stats["std"], dtype=float)
        if "sem" in obs_stats:
            payload["obs_sem"] = np.asarray(obs_stats["sem"], dtype=float)

    if res.get("Pbar") is not None:
        payload["Pbar"] = np.asarray(res["Pbar"], dtype=float)

    fit = res.get("fit", {})
    payload["fit_model"] = np.asarray(str(fit.get("model", "exponential")))

    params = fit.get("params", {})
    for k, v in params.items():
        try:
            payload[f"fit_param__{k}"] = np.asarray([float(v)], dtype=float)
        except Exception:
            pass

    stderr = fit.get("param_stderr", {})
    for k, v in stderr.items():
        if v is None:
            continue
        try:
            payload[f"fit_stderr__{k}"] = np.asarray([float(v)], dtype=float)
        except Exception:
            pass

    if extra_meta:
        for k, v in extra_meta.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                payload[f"meta__{k}"] = np.asarray([v])
            elif v is None:
                payload[f"meta__{k}"] = np.asarray("None")
            else:
                payload[f"meta__{k}"] = np.asarray(str(v))

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)

def _fmt_num(x, digits: int = 6) -> str:
    if x is None:
        return "N/A"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return "N/A"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def _fmt_list_inline(xs, digits: int = 6) -> str:
    if xs is None:
        return "N/A"
    try:
        return "[" + ", ".join(_fmt_num(v, digits) for v in xs) + "]"
    except Exception:
        return str(xs)
    
def _fmt_fix(x, digits: int = 6) -> str:
        if x is None:
            return "N/A"
        try:
            x = float(x)
        except Exception:
            return str(x)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:.{digits}f}"

def _infer_n_free_params(fit: Dict[str, Any]) -> int:
    """
    尽量从 param_stderr 推断真正自由参数数 k；
    若推不出来，再退回 params 个数。
    """
    stderr = fit.get("param_stderr", {}) or {}
    if isinstance(stderr, dict) and len(stderr) > 0:
        k = sum(v is not None for v in stderr.values())
        if k > 0:
            return int(k)

    params = fit.get("params", {}) or {}
    return max(1, len(params))


def _predict_from_fit(model: str, params: Dict[str, Any], m_obs: np.ndarray) -> Optional[np.ndarray]:
    """
    根据 fit model + params 生成观测点上的预测值。
    这里只支持你当前最可能会用到的几类。
    """
    model = str(model).lower()

    try:
        if model in ("exponential", "single_exp", "single_exponential"):
            A = float(params.get("A", 0.5))
            p = float(params["p"])
            B = float(params.get("B", 0.5))
            return A * (p ** m_obs) + B

        if model in ("stretched_exponential", "stretched_exp"):
            A = float(params.get("A", 0.5))
            p = float(params["p"])
            beta = float(params["beta"])
            B = float(params.get("B", 0.5))
            return A * np.exp(-(1.0 - p) * (m_obs ** beta)) + B

        if model in ("double_exponential", "double_exp"):
            A1 = float(params["A1"])
            p1 = float(params["p1"])
            A2 = float(params["A2"])
            p2 = float(params["p2"])
            B = float(params.get("B", 0.5))
            return A1 * (p1 ** m_obs) + A2 * (p2 ** m_obs) + B

    except Exception:
        return None

    return None


def _compute_basic_fit_diagnostics(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    只基于当前已有 result 内容计算“基础 diagnostics”：
      - RSS
      - RMSE
      - max_abs_resid
      - AICc
    不要求你先改 fit 主逻辑，属于纯加法。
    """
    y_obs = res.get("Ybar", None)
    if y_obs is None:
        y_obs = res.get("Pbar", None)

    m_list = res.get("m_list", None)
    fit = res.get("fit", {}) or {}
    model = fit.get("model", None)
    params = fit.get("params", {}) or {}

    if y_obs is None or m_list is None or model is None:
        return {
            "rss": None,
            "rmse": None,
            "max_abs_resid": None,
            "aicc": None,
            "resid": None,
            "y_hat": None,
            "n_points": None,
            "n_free_params": None,
        }

    m_obs = np.asarray(m_list, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    y_hat = _predict_from_fit(model, params, m_obs)
    if y_hat is None or len(y_hat) != len(y_obs):
        return {
            "rss": None,
            "rmse": None,
            "max_abs_resid": None,
            "aicc": None,
            "resid": None,
            "y_hat": None,
            "n_points": len(y_obs),
            "n_free_params": None,
        }

    resid = y_obs - y_hat
    rss = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    max_abs_resid = float(np.max(np.abs(resid)))

    n = int(len(y_obs))
    k = int(_infer_n_free_params(fit))

    aicc = None
    if n > k + 1:
        # AICc = n ln(RSS/n) + 2k + 2k(k+1)/(n-k-1)
        rss_safe = max(rss, 1e-300)
        aic = n * math.log(rss_safe / n) + 2 * k
        aicc = float(aic + (2 * k * (k + 1)) / (n - k - 1))

    return {
        "rss": rss,
        "rmse": rmse,
        "max_abs_resid": max_abs_resid,
        "aicc": aicc,
        "resid": resid,
        "y_hat": y_hat,
        "n_points": n,
        "n_free_params": k,
    }


def _extract_delta_aicc_single_vs_stretched(res: Dict[str, Any]) -> Optional[float]:
    """
    先尝试从 result 中读取你以后可能补进去的模型比较结果；
    如果还没有，就返回 None。
    """
    # 约定 1：res["diagnostics"]["delta_aicc_single_vs_stretched"]
    diag = res.get("diagnostics", {}) or {}
    if "delta_aicc_single_vs_stretched" in diag:
        try:
            return float(diag["delta_aicc_single_vs_stretched"])
        except Exception:
            pass

    # 约定 2：res["model_compare"]["delta_aicc_single_vs_stretched"]
    mc = res.get("model_compare", {}) or {}
    if "delta_aicc_single_vs_stretched" in mc:
        try:
            return float(mc["delta_aicc_single_vs_stretched"])
        except Exception:
            pass

    return None

def save_results_exp2_rb_decay(
    cfg: Dict,
    out: Dict,
    out_path,
    results_filename: str = "result.txt",
):
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / results_filename

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    exp = cfg["experiment"]

    # 先把各 condition 的信息整理一遍，后面统一写 overview
    rows = []

    for label, block in out["conditions"].items():
        cond = block["condition"]
        res = block["result"]
        fit = res["fit"]

        # 原始数据照旧保存
        _save_rb_raw_npz(
            res,
            raw_dir / f"{label}.npz",
            extra_meta={
                "label": label,
                "tau_c": cond.get("tau_c"),
                "idle_insert_prob": cond.get("idle_insert_prob"),
                "idle_duration_ns": cond.get("idle_duration_ns"),
                "flux_mode": cond.get("flux_mode"),
                "rho": cond.get("rho"),
                "seed": out.get("seed"),
            },
        )

        # 基础 diagnostics：纯加法
        basic_diag = _compute_basic_fit_diagnostics(res)
        delta_aicc = _extract_delta_aicc_single_vs_stretched(res)

        rows.append({
            "label": label,
            "cond": cond,
            "res": res,
            "fit": fit,
            "diag": basic_diag,
            "delta_aicc_single_vs_stretched": delta_aicc,
        })

        # 图照旧保存
        _save_rb_fit_plot_single(
            res,
            out_dir=out_dir,
            filename=f"rb_fit_{label}.png",
        )

    with open(results_file, "w", encoding="utf-8") as f:
        sep = "=" * 110
        sub = "-" * 110

        # =========================
        # Header
        # =========================
        f.write(sep + "\n")
        f.write(f"Experiment : {exp['name']}\n")
        f.write(f"Title      : {exp['title']}\n")
        f.write(f"Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Metric     : {out.get('metric', 'survival')}\n")
        f.write(f"Seed       : {out.get('seed')}\n")
        f.write(sep + "\n\n")

        # =========================
        # Overview Table
        # =========================
        f.write("[Overview]\n")
        f.write(sub + "\n")
        header = (
            f"{'label':<20}"
            f"{'tau_c(ns)':>12}"
            f"{'idle_p':>10}"
            f"{'idle_ns':>10}"
            f"{'p':>12}"
            f"{'EPC':>12}"
            f"{'RMSE':>12}"
            f"{'max|r|':>12}"
            f"{'ΔAICc(S-SS)':>14}"
        )
        f.write(header + "\n")
        f.write(sub + "\n")

        for row in rows:
            label = row["label"]
            cond = row["cond"]
            res = row["res"]
            fit = row["fit"]
            diag = row["diag"]

            p_val = (fit.get("params", {}) or {}).get("p", None)

            line = (
                f"{label:<20}"
                f"{_fmt_num(cond.get('tau_c'), 3):>12}"
                f"{_fmt_num(cond.get('idle_insert_prob', 0.0), 3):>10}"
                f"{_fmt_num(cond.get('idle_duration_ns', 20.0), 3):>10}"
                f"{_fmt_num(p_val, 6):>12}"
                f"{_fmt_num(res.get('EPC'), 6):>12}"
                f"{_fmt_num(diag.get('rmse'), 6):>12}"
                f"{_fmt_num(diag.get('max_abs_resid'), 6):>12}"
                f"{_fmt_num(row.get('delta_aicc_single_vs_stretched'), 6):>14}"
            )
            f.write(line + "\n")

        f.write(sub + "\n")
        f.write("Notes:\n")
        f.write("  - EPC is kept for reference only; under time-correlated / non-Markovian noise it should not be over-interpreted.\n")
        f.write("  - ΔAICc(S-SS) means AICc(single exponential) - AICc(stretched exponential); currently N/A if stretched fit is not provided.\n")
        f.write("\n")

        # =========================
        # Detailed blocks
        # =========================
        for row in rows:
            label = row["label"]
            cond = row["cond"]
            res = row["res"]
            fit = row["fit"]
            diag = row["diag"]

            f.write(sep + "\n")
            f.write(f"[Condition] {label}\n")
            f.write(sep + "\n")

            # ---- Config ----
            f.write("Config\n")
            f.write(sub + "\n")
            f.write(f"  tau_c (ns)         : {_fmt_num(cond.get('tau_c'), 6)}\n")
            f.write(f"  flux_mode          : {cond.get('flux_mode', 'N/A')}\n")
            f.write(f"  rho                : {_fmt_num(cond.get('rho'), 6)}\n")
            f.write(f"  idle_insert_prob   : {_fmt_num(cond.get('idle_insert_prob', 0.0), 6)}\n")
            f.write(f"  idle_duration_ns   : {_fmt_num(cond.get('idle_duration_ns', 20.0), 6)}\n")
            f.write("\n")

            # ---- Fit summary ----
            f.write("Fit Summary\n")
            f.write(sub + "\n")
            f.write(f"  fit_model          : {fit.get('model', 'N/A')}\n")
            f.write(f"  fit_params         : {fit.get('params', {})}\n")
            f.write(f"  fit_param_stderr   : {fit.get('param_stderr', {})}\n")
            f.write(f"  EPC                : {_fmt_num(res.get('EPC'), 8)}\n")
            f.write(f"  EPC_stderr         : {_fmt_num(res.get('EPC_stderr'), 8)}\n")
            if "note" in res:
                f.write(f"  note               : {res['note']}\n")
            f.write("\n")

            # ---- Diagnostics ----
            f.write("Diagnostics\n")
            f.write(sub + "\n")
            f.write(f"  n_points           : {diag.get('n_points', 'N/A')}\n")
            f.write(f"  n_free_params      : {diag.get('n_free_params', 'N/A')}\n")
            f.write(f"  RSS                : {_fmt_num(diag.get('rss'), 8)}\n")
            f.write(f"  RMSE               : {_fmt_num(diag.get('rmse'), 8)}\n")
            f.write(f"  max_abs_resid      : {_fmt_num(diag.get('max_abs_resid'), 8)}\n")
            f.write(f"  AICc(single)       : {_fmt_num(diag.get('aicc'), 8)}\n")
            f.write(f"  ΔAICc(S-SS)        : {_fmt_num(row.get('delta_aicc_single_vs_stretched'), 8)}\n")
            f.write("\n")

            # ---- Data ----
            f.write("Data\n")
            f.write(sub + "\n")
            f.write(f"  m_list             : {_fmt_list_inline(res.get('m_list'), 0)}\n")
            f.write(f"  Ybar               : {_fmt_list_inline(res.get('Ybar'), 6)}\n")
            obs_stats = res.get("obs_stats", {}) or {}
            if "std" in obs_stats:
                f.write(f"  obs_std            : {_fmt_list_inline(obs_stats.get('std'), 6)}\n")
            if "sem" in obs_stats:
                f.write(f"  obs_sem            : {_fmt_list_inline(obs_stats.get('sem'), 6)}\n")

            if res.get("Pbar", None) is not None:
                f.write(f"  Pbar               : {_fmt_list_inline(res.get('Pbar'), 6)}\n")

            if "seq_vals" in res and res["seq_vals"] is not None:
                seq_vals = np.asarray(res["seq_vals"], dtype=float)
                f.write(f"  seq_vals.shape     : {tuple(seq_vals.shape)}\n")
            else:
                f.write("  seq_vals.shape     : N/A\n")

            f.write("\n")

            # ---- Files ----
            f.write("Artifacts\n")
            f.write(sub + "\n")
            f.write(f"  raw_npz            : {raw_dir / f'{label}.npz'}\n")
            f.write(f"  fit_plot           : {out_dir / f'rb_fit_{label}.png'}\n")
            f.write("\n")

        f.write(sep + "\n")
        f.write("[End of Report]\n")
        f.write(sep + "\n")

    return results_file

# def save_results_exp2_rb_decay(
#     cfg: Dict,
#     out: Dict,
#     out_path,
#     results_filename: str = "results_exp2_rb_decay.txt",
# ):
#     out_dir = Path(out_path)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     results_file = out_dir / results_filename

#     raw_dir = out_dir / "raw"
#     raw_dir.mkdir(parents=True, exist_ok=True)

#     exp = cfg["experiment"]

#     with open(results_file, "w", encoding="utf-8") as f:
#         f.write(f"Experiment: {exp['name']}\n")
#         f.write(f"Title: {exp['title']}\n")
#         f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#         f.write(f"Metric: {out.get('metric', 'survival')}\n")
#         f.write(f"Seed: {out.get('seed')}\n\n")

#         for label, block in out["conditions"].items():
#             cond = block["condition"]
#             res  = block["result"]
#             fit  = res["fit"]

#             _save_rb_raw_npz(
#                 res,
#                 raw_dir / f"{label}.npz",
#                 extra_meta={
#                     "label": label,
#                     "tau_c": cond.get("tau_c"),
#                     "idle_insert_prob": cond.get("idle_insert_prob"),
#                     "idle_duration_ns": cond.get("idle_duration_ns"),
#                     "flux_mode": cond.get("flux_mode"),
#                     "rho": cond.get("rho"),
#                     "seed": out.get("seed"),
#                 },
#             )

#             f.write(f"[Condition: {label}]\n")
#             f.write(f"tau_c = {cond.get('tau_c')}\n")
#             f.write(f"flux_mode = {cond.get('flux_mode')}\n")
#             f.write(f"rho = {cond.get('rho')}\n")
#             f.write(f"idle_insert_prob = {cond.get('idle_insert_prob', 0.0)}\n")
#             f.write(f"idle_duration_ns = {cond.get('idle_duration_ns', 20.0)}\n")
#             f.write(f"m_list = {res['m_list']}\n")
#             f.write(f"Ybar = {res['Ybar']}\n")
#             f.write(f"fit_model = {fit.get('model')}\n")
#             f.write(f"fit_params = {fit.get('params')}\n")
#             f.write(f"fit_param_stderr = {fit.get('param_stderr')}\n")
#             f.write(f"EPC = {res.get('EPC')}\n")
#             f.write(f"EPC_stderr = {res.get('EPC_stderr')}\n")
#             if "note" in res:
#                 f.write(f"note = {res['note']}\n")
#             f.write("\n")

#             _save_rb_fit_plot_single(
#                 res,
#                 out_dir=out_dir,
#                 filename=f"rb_fit_{label}.png",
#             )

#     return results_file

def save_results_exp2_ablation(
    cfg: Dict,
    out: Dict,
    out_path,
    results_filename: str = "result.txt",
):
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / results_filename

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    exp = cfg["experiment"]

    # ---------- all-on diagnostics ----------
    all_diag = out["all"].get("diagnostics", None)
    if all_diag is None:
        all_diag = _compute_basic_fit_diagnostics(out["all"])

    # ---------- ablation diagnostics ----------
    ablation_rows = []
    for k, v in out["ablations"].items():
        diag = v.get("diagnostics", None)
        if diag is None:
            diag = _compute_basic_fit_diagnostics(v)

        ablation_rows.append({
            "key": k,
            "res": v,
            "diag": diag,
            "delta_epc": out["deltas"].get(k),
            "share": out["shares"].get(k),
        })

    with open(results_file, "w", encoding="utf-8") as f:
        sep = "=" * 110
        sub = "-" * 110

        # ==================================================
        # Header
        # ==================================================
        f.write(sep + "\n")
        f.write(f"Experiment : {exp['name']}\n")
        f.write(f"Title      : {exp['title']}\n")
        f.write(f"Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Metric     : {out.get('metric', 'survival')}\n")
        f.write(f"Seed       : {out.get('seed')}\n")
        f.write(f"Condition  : {out.get('condition')}\n")
        f.write(sep + "\n\n")

        # ==================================================
        # Save raw first
        # ==================================================
        _save_rb_raw_npz(
            out["all"],
            raw_dir / "all_on.npz",
            extra_meta={
                "seed": out.get("seed"),
                "condition": str(out.get("condition")),
                "kind": "all_on",
            },
        )

        for row in ablation_rows:
            _save_rb_raw_npz(
                row["res"],
                raw_dir / f"ablation_{row['key']}.npz",
                extra_meta={
                    "seed": out.get("seed"),
                    "condition": str(out.get("condition")),
                    "kind": "ablation",
                    "ablation_key": row["key"],
                },
            )

        # ==================================================
        # Overview
        # ==================================================
        f.write("[Overview]\n")
        f.write(sub + "\n")
        header = (
            f"{'case':<24}"
            f"{'p':>12}"
            f"{'EPC':>12}"
            f"{'RMSE':>12}"
            f"{'max|r|':>12}"
            f"{'ΔEPC':>12}"
            f"{'share(%)':>12}"
        )
        f.write(header + "\n")
        f.write(sub + "\n")

        p_all = (out["all"].get("fit", {}).get("params", {}) or {}).get("p", None)
        f.write(
            f"{'all_on':<24}"
            f"{_fmt_num(p_all, 6):>12}"
            f"{_fmt_num(out['all'].get('EPC'), 6):>12}"
            f"{_fmt_num(all_diag.get('rmse'), 6):>12}"
            f"{_fmt_num(all_diag.get('max_abs_resid'), 6):>12}"
            f"{'--':>12}"
            f"{'--':>12}\n"
        )

        for row in ablation_rows:
            key = row["key"]
            res = row["res"]
            diag = row["diag"]
            p_val = (res.get("fit", {}).get("params", {}) or {}).get("p", None)
            share_pct = None if row["share"] is None else 100.0 * float(row["share"])

            f.write(
                f"{('off_' + key):<24}"
                f"{_fmt_num(p_val, 6):>12}"
                f"{_fmt_num(res.get('EPC'), 6):>12}"
                f"{_fmt_num(diag.get('rmse'), 6):>12}"
                f"{_fmt_num(diag.get('max_abs_resid'), 6):>12}"
                f"{_fmt_num(row.get('delta_epc'), 6):>12}"
                f"{_fmt_num(share_pct, 4):>12}\n"
            )

        f.write(sub + "\n")
        f.write("Notes:\n")
        f.write("  - ΔEPC = EPC(all_on) - EPC(off_i)\n")
        f.write("  - share(%) is normalized from positive ΔEPC only\n")
        f.write("  - RMSE / max|r| measure how far the mean RB decay deviates from the fitted single-exponential reference\n")
        f.write("\n")

        # ==================================================
        # All-on block
        # ==================================================
        f.write(sep + "\n")
        f.write("[All-on]\n")
        f.write(sep + "\n")

        f.write("Fit Summary\n")
        f.write(sub + "\n")
        f.write(f"  fit_model          : {out['all'].get('fit', {}).get('model', 'N/A')}\n")
        f.write(f"  fit_params         : {out['all'].get('fit', {}).get('params', {})}\n")
        f.write(f"  fit_param_stderr   : {out['all'].get('fit', {}).get('param_stderr', {})}\n")
        f.write(f"  EPC                : {_fmt_num(out['all'].get('EPC'), 8)}\n")
        f.write(f"  EPC_stderr         : {_fmt_num(out['all'].get('EPC_stderr'), 8)}\n")
        f.write("\n")

        f.write("Diagnostics\n")
        f.write(sub + "\n")
        f.write(f"  RSS                : {_fmt_num(all_diag.get('rss'), 8)}\n")
        f.write(f"  RMSE               : {_fmt_num(all_diag.get('rmse'), 8)}\n")
        f.write(f"  max_abs_resid      : {_fmt_num(all_diag.get('max_abs_resid'), 8)}\n")
        f.write(f"  AICc(single)       : {_fmt_num(all_diag.get('aicc'), 8)}\n")
        f.write("\n")

        f.write("Data\n")
        f.write(sub + "\n")
        f.write(f"  m_list             : {_fmt_list_inline(out['all'].get('m_list'), 0)}\n")
        f.write(f"  Ybar               : {_fmt_list_inline(out['all'].get('Ybar'), 6)}\n")
        obs_stats = out["all"].get("obs_stats", {}) or {}
        if "std" in obs_stats:
            f.write(f"  obs_std            : {_fmt_list_inline(obs_stats.get('std'), 6)}\n")
        if "sem" in obs_stats:
            f.write(f"  obs_sem            : {_fmt_list_inline(obs_stats.get('sem'), 6)}\n")
        seq_vals = np.asarray(out["all"].get("seq_vals"), dtype=float)
        f.write(f"  seq_vals.shape     : {tuple(seq_vals.shape)}\n")
        f.write("\n")

        f.write("Artifacts\n")
        f.write(sub + "\n")
        f.write(f"  raw_npz            : {raw_dir / 'all_on.npz'}\n")
        f.write(f"  fit_plot           : {out_dir / 'rb_fit_all_on.png'}\n")
        f.write("\n")

        _save_rb_fit_plot_single(
            out["all"],
            out_dir=out_dir,
            filename="rb_fit_all_on.png",
        )

        # ==================================================
        # Ablation blocks
        # ==================================================
        for row in ablation_rows:
            key = row["key"]
            res = row["res"]
            diag = row["diag"]

            f.write(sep + "\n")
            f.write(f"[Ablation] off_{key}\n")
            f.write(sep + "\n")

            f.write("Fit Summary\n")
            f.write(sub + "\n")
            f.write(f"  fit_model          : {res.get('fit', {}).get('model', 'N/A')}\n")
            f.write(f"  fit_params         : {res.get('fit', {}).get('params', {})}\n")
            f.write(f"  fit_param_stderr   : {res.get('fit', {}).get('param_stderr', {})}\n")
            f.write(f"  EPC                : {_fmt_num(res.get('EPC'), 8)}\n")
            f.write(f"  EPC_stderr         : {_fmt_num(res.get('EPC_stderr'), 8)}\n")
            f.write(f"  ΔEPC               : {_fmt_num(row.get('delta_epc'), 8)}\n")
            share_pct = None if row["share"] is None else 100.0 * float(row["share"])
            f.write(f"  share(%)           : {_fmt_num(share_pct, 6)}\n")
            f.write("\n")

            f.write("Diagnostics\n")
            f.write(sub + "\n")
            f.write(f"  RSS                : {_fmt_num(diag.get('rss'), 8)}\n")
            f.write(f"  RMSE               : {_fmt_num(diag.get('rmse'), 8)}\n")
            f.write(f"  max_abs_resid      : {_fmt_num(diag.get('max_abs_resid'), 8)}\n")
            f.write(f"  AICc(single)       : {_fmt_num(diag.get('aicc'), 8)}\n")
            f.write("\n")

            f.write("Data\n")
            f.write(sub + "\n")
            f.write(f"  m_list             : {_fmt_list_inline(res.get('m_list'), 0)}\n")
            f.write(f"  Ybar               : {_fmt_list_inline(res.get('Ybar'), 6)}\n")
            obs_stats = res.get("obs_stats", {}) or {}
            if "std" in obs_stats:
                f.write(f"  obs_std            : {_fmt_list_inline(obs_stats.get('std'), 6)}\n")
            if "sem" in obs_stats:
                f.write(f"  obs_sem            : {_fmt_list_inline(obs_stats.get('sem'), 6)}\n")
            seq_vals = np.asarray(res.get("seq_vals"), dtype=float)
            f.write(f"  seq_vals.shape     : {tuple(seq_vals.shape)}\n")
            f.write("\n")

            f.write("Artifacts\n")
            f.write(sub + "\n")
            f.write(f"  raw_npz            : {raw_dir / f'ablation_{key}.npz'}\n")
            f.write(f"  fit_plot           : {out_dir / f'rb_fit_ablation_{key}.png'}\n")
            f.write("\n")

            _save_rb_fit_plot_single(
                res,
                out_dir=out_dir,
                filename=f"rb_fit_ablation_{key}.png",
            )

        f.write(sep + "\n")
        f.write("[End of Report]\n")
        f.write(sep + "\n")

    return results_file

def _load_rb_result_from_npz(npz_path: Path) -> Dict[str, Any]:
    arr = np.load(npz_path, allow_pickle=False)

    fit_params = {}
    fit_stderr = {}

    for k in arr.files:
        if k.startswith("fit_param__"):
            fit_params[k.replace("fit_param__", "")] = float(arr[k][0])
        elif k.startswith("fit_stderr__"):
            fit_stderr[k.replace("fit_stderr__", "")] = float(arr[k][0])

    res = {
        "metric": str(arr["metric"].item()) if "metric" in arr else "survival",
        "m_list": arr["m_list"].astype(int).tolist() if "m_list" in arr else [],
        "Ybar": arr["Ybar"].astype(float).tolist() if "Ybar" in arr else [],
        "fit": {
            "model": str(arr["fit_model"].item()) if "fit_model" in arr else "exponential",
            "params": fit_params,
            "param_stderr": fit_stderr,
        },
        "EPC": float(arr["EPC"][0]) if "EPC" in arr else None,
        "EPC_stderr": float(arr["EPC_stderr"][0]) if "EPC_stderr" in arr else None,
    }

    if "Pbar" in arr:
        res["Pbar"] = arr["Pbar"].astype(float).tolist()
    else:
        res["Pbar"] = None

    if "seq_vals" in arr:
        res["seq_vals"] = arr["seq_vals"].astype(float)
    else:
        res["seq_vals"] = None

    obs_stats = {}
    if "obs_mean" in arr:
        obs_stats["mean"] = arr["obs_mean"].astype(float)
    if "obs_std" in arr:
        obs_stats["std"] = arr["obs_std"].astype(float)
    if "obs_sem" in arr:
        obs_stats["sem"] = arr["obs_sem"].astype(float)
    if obs_stats:
        res["obs_stats"] = obs_stats

    meta = {}
    for k in arr.files:
        if k.startswith("meta__"):
            key = k.replace("meta__", "")
            v = arr[k]
            if isinstance(v, np.ndarray):
                if v.shape == ():
                    meta[key] = v.item()
                elif v.size == 1:
                    meta[key] = v.reshape(-1)[0].item()
                else:
                    meta[key] = v.tolist()
            else:
                meta[key] = v
    res["meta"] = meta

    res["diagnostics"] = _compute_basic_fit_diagnostics(res)

    return res


def regenerate_exp2_ablation_report_from_raw(
    raw_dir,
    out_dir=None,
    title: Optional[str] = None,
    results_filename: str = "result_pretty.txt",
):
    """
    基于已保存的 raw/*.npz 重新生成一个更整洁的 ablation 报告。
    不重跑仿真。
    """
    raw_dir = Path(raw_dir)
    if out_dir is None:
        out_dir = raw_dir.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / results_filename

    all_path = raw_dir / "all_on.npz"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing raw file: {all_path}")

    all_res = _load_rb_result_from_npz(all_path)

    ablation_paths = sorted(raw_dir.glob("ablation_*.npz"))
    if not ablation_paths:
        raise FileNotFoundError(f"No ablation_*.npz found under {raw_dir}")

    ablation_rows = []
    for p in ablation_paths:
        res = _load_rb_result_from_npz(p)
        key = p.stem.replace("ablation_", "")
        ablation_rows.append({
            "key": key,
            "res": res,
            "diag": res["diagnostics"],
        })

    r_all = all_res.get("EPC", None)
    deltas = {}
    for row in ablation_rows:
        epc_i = row["res"].get("EPC", None)
        if r_all is not None and epc_i is not None:
            deltas[row["key"]] = float(r_all - epc_i)
        else:
            deltas[row["key"]] = None

    S = sum(max(0.0, x) for x in deltas.values() if x is not None) or 1.0
    shares = {
        k: (max(0.0, v) / S if v is not None else 0.0)
        for k, v in deltas.items()
    }

    used_seed = all_res.get("meta", {}).get("seed", "N/A")
    condition = all_res.get("meta", {}).get("condition", "N/A")
    metric = all_res.get("metric", "survival")
    if title is None:
        title = "Exp2.2 ablation report regenerated from raw data"

    sep = "=" * 118
    sub = "-" * 118

    with open(results_file, "w", encoding="utf-8") as f:
        f.write(sep + "\n")
        f.write(f"Title      : {title}\n")
        f.write(f"Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Metric     : {metric}\n")
        f.write(f"Seed       : {used_seed}\n")
        f.write(f"Condition  : {condition}\n")
        f.write(sep + "\n\n")

        f.write("[Overview]\n")
        f.write(sub + "\n")
        header = (
            f"{'case':<26} | "
            f"{'p':>10} | "
            f"{'EPC':>10} | "
            f"{'RMSE':>10} | "
            f"{'max|r|':>10} | "
            f"{'ΔEPC':>10} | "
            f"{'share(%)':>10}"
        )
        f.write(header + "\n")
        f.write(sub + "\n")

        p_all = (all_res.get("fit", {}).get("params", {}) or {}).get("p", None)
        diag_all = all_res["diagnostics"]
        f.write(
            f"{'all_on':<26} | "
            f"{_fmt_fix(p_all, 6):>10} | "
            f"{_fmt_fix(all_res.get('EPC'), 6):>10} | "
            f"{_fmt_fix(diag_all.get('rmse'), 6):>10} | "
            f"{_fmt_fix(diag_all.get('max_abs_resid'), 6):>10} | "
            f"{'--':>10} | "
            f"{'--':>10}\n"
        )

        for row in ablation_rows:
            key = row["key"]
            res = row["res"]
            diag = row["diag"]
            p_val = (res.get("fit", {}).get("params", {}) or {}).get("p", None)
            share_pct = None if shares[key] is None else 100.0 * float(shares[key])

            f.write(
                f"{('off_' + key):<26} | "
                f"{_fmt_fix(p_val, 6):>10} | "
                f"{_fmt_fix(res.get('EPC'), 6):>10} | "
                f"{_fmt_fix(diag.get('rmse'), 6):>10} | "
                f"{_fmt_fix(diag.get('max_abs_resid'), 6):>10} | "
                f"{_fmt_fix(deltas[key], 6):>10} | "
                f"{_fmt_fix(share_pct, 4):>10}\n"
            )

        f.write(sub + "\n")
        f.write("Notes:\n")
        f.write("  - ΔEPC = EPC(all_on) - EPC(off_i)\n")
        f.write("  - share(%) is normalized from positive ΔEPC only\n")
        f.write("  - RMSE / max|r| are the diagnostics of single-exponential adequacy\n")
        f.write("\n")

        def _write_one_block(name: str, res: Dict[str, Any]):
            diag = res["diagnostics"]
            f.write(sep + "\n")
            f.write(f"[{name}]\n")
            f.write(sep + "\n")

            f.write("Fit Summary\n")
            f.write(sub + "\n")
            f.write(f"  fit_model          : {res.get('fit', {}).get('model', 'N/A')}\n")
            f.write(f"  fit_params         : {res.get('fit', {}).get('params', {})}\n")
            f.write(f"  fit_param_stderr   : {res.get('fit', {}).get('param_stderr', {})}\n")
            f.write(f"  EPC                : {_fmt_num(res.get('EPC'), 8)}\n")
            f.write(f"  EPC_stderr         : {_fmt_num(res.get('EPC_stderr'), 8)}\n")
            f.write("\n")

            f.write("Diagnostics\n")
            f.write(sub + "\n")
            f.write(f"  RSS                : {_fmt_num(diag.get('rss'), 8)}\n")
            f.write(f"  RMSE               : {_fmt_num(diag.get('rmse'), 8)}\n")
            f.write(f"  max_abs_resid      : {_fmt_num(diag.get('max_abs_resid'), 8)}\n")
            f.write(f"  AICc(single)       : {_fmt_num(diag.get('aicc'), 8)}\n")
            f.write("\n")

            f.write("Data\n")
            f.write(sub + "\n")
            f.write(f"  m_list             : {_fmt_list_inline(res.get('m_list'), 0)}\n")
            f.write(f"  Ybar               : {_fmt_list_inline(res.get('Ybar'), 6)}\n")
            obs_stats = res.get("obs_stats", {}) or {}
            if "std" in obs_stats:
                f.write(f"  obs_std            : {_fmt_list_inline(obs_stats.get('std'), 6)}\n")
            if "sem" in obs_stats:
                f.write(f"  obs_sem            : {_fmt_list_inline(obs_stats.get('sem'), 6)}\n")
            if res.get("seq_vals", None) is not None:
                f.write(f"  seq_vals.shape     : {tuple(np.asarray(res['seq_vals']).shape)}\n")
            f.write("\n")

        _write_one_block("All-on", all_res)
        for row in ablation_rows:
            _write_one_block(f"Ablation off_{row['key']}", row["res"])

        f.write(sep + "\n")
        f.write("[End of Report]\n")
        f.write(sep + "\n")

    return results_file



def run_exp2_experiment(cfg: Dict, seed: Optional[int] = None) -> Dict:
    """
        实验二主函数：运行全噪声与逐类消融的 RB 扫描，计算边际贡献与占比。
        Args:
            cfg: 实验配置字典
            seed: 随机种子（可选，若传入则覆盖 cfg 中的 seed）
        Returns: 结果字典，包括 "all"（全噪声）和 "ablations"（逐类消融）两部分
    """
    used_seed = _choose_seed_and_set(cfg, seed)

    # 1) 全噪声
    res_all = _run_one_rb1_sweep(cfg, seed = used_seed)

    # 2) 逐类消融
    ablations = {}
    for key in ["use_idle", "use_photon_decay", "use_ry", "use_segmented_flux"]:
    # for key in ["use_idle", "use_ry", "use_photon_decay"]:
        res = _run_one_rb1_sweep(cfg, seed = used_seed, noise_overrides={key: False})
        ablations[key] = res

    # 3) 边际贡献 + 占比
    r_all = res_all.get("EPC")
    deltas = {}
    for k, v in ablations.items():
        epc_i = v.get("EPC")
        if epc_i is not None and r_all is not None:
            deltas[k] = r_all - epc_i
        else:
            deltas[k] = None  # 或者直接跳过，或设置为 0.0
    S = sum(max(0.0, x) for x in deltas.values() if x is not None) or 1.0
    shares = {
        k: max(0.0, v) / S if v is not None else 0.0
        for k, v in deltas.items()
    }

    return {
        "seed": used_seed,
        "all": res_all,
        "ablations": ablations,
        "deltas": deltas,
        "shares": shares,
    }

def _save_rb_fit_plot1(out: Dict, out_dir: Path, filename: str = "rb_fit.png") -> Path:
    """
        保存 RB 拟合图，返回图片路径。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    allres = out["all"]
    m_obs = np.asarray(allres["m_list"], dtype=float)
    y_obs = np.asarray(allres["Pbar"], dtype=float)
    fit   = allres["fit"]
    metric = out.get("metric", "fidelity").lower() 

    model = fit.get("model", "exponential")
    params = fit.get("params", {})
    stderr = fit.get("param_stderr", {})

    # 拟合曲线采样
    m0 = 0.0
    m_max = float(max([0.0] + list(m_obs)))
    m_dense = np.linspace(m0, m_max, 200)

    # === 构造模型函数和置信区间计算 ===
    def compute_y_and_ci(model_name: str):
        if model_name == "exponential":
            A = params.get("A", 0.9)
            p = params.get("p", 0.99)
            B = params.get("B", 0.01)
            y_fit = A * p**m_dense + B
            A_se = stderr.get("A"); p_se = stderr.get("p"); B_se = stderr.get("B")
            sigma_terms = []
            if A_se: sigma_terms.append((p**m_dense * A_se)**2)
            if p_se and p > 0: sigma_terms.append((A * m_dense * p**(m_dense - 1) * p_se)**2)
            if B_se: sigma_terms.append(B_se**2)
            return y_fit, sigma_terms, f"A={A:.3f}, p={p:.5f}, B={B:.5f}"

        elif model_name == "double_exponential":
            A1 = params.get("A1", 0.5)
            p1 = params.get("p1", 0.99)
            A2 = params.get("A2", 0.3)
            p2 = params.get("p2", 0.95)
            B = params.get("B", 0.01)
            y_fit = A1 * p1**m_dense + A2 * p2**m_dense + B
            return y_fit, [], f"A1={A1:.2f}, p1={p1:.3f}, A2={A2:.2f}, p2={p2:.3f}, B={B:.3f}"

        elif model_name == "stretched_exponential":
            A = params.get("A", 0.9)
            p = params.get("p", 0.99)
            beta = params.get("beta", 1.0)
            B = params.get("B", 0.01)
            y_fit = A * np.exp(-(1 - p) * m_dense**beta) + B
            return y_fit, [], f"A={A:.2f}, p={p:.3f}, β={beta:.2f}, B={B:.3f}"

        else:
            raise ValueError(f"Unsupported model: {model}")

    y_fit, sigma_terms, legend_str = compute_y_and_ci(model)

    # 拟合值 & 残差
    if model == "exponential":
        y_hat_obs = params["A"] * params["p"]**m_obs + params["B"]
    elif model == "double_exponential":
        y_hat_obs = (
            params["A1"] * params["p1"]**m_obs +
            params["A2"] * params["p2"]**m_obs +
            params["B"]
        )
    elif model == "stretched_exponential":
        y_hat_obs = (
            params["A"] * np.exp(-(1 - params["p"]) * m_obs**params["beta"]) +
            params["B"]
        )
    else:
        y_hat_obs = np.zeros_like(m_obs)

    resid = y_obs - y_hat_obs

    # 开始画图
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 6.2), dpi=140,
        gridspec_kw={'height_ratios':[3,1]}
    )

    obs_label = "Observed F(m)" if metric == "fidelity" else "Observed P(m)"
    ax1.scatter(m_obs, y_obs, s=28, label=obs_label)
    ln, = ax1.plot(m_dense, y_fit, lw=2.0, label=f"Fit: {model}")
    ax1.set_xlabel("m (sequence depth)")
    if metric == "fidelity":
        ax1.set_ylabel("Mean fidelity F(m)")
    else:
        ax1.set_ylabel("Mean survival probability P(m)")
    ax1.set_title(f"RB Fit: {model}")
    ax1.grid(True, linestyle="--", alpha=0.3)

    if sigma_terms:
        sigma_y = np.sqrt(np.sum(sigma_terms, axis=0))
        y_lo = y_fit - 1.96 * sigma_y
        y_hi = y_fit + 1.96 * sigma_y
        ax1.fill_between(
            m_dense, y_lo, y_hi,
            alpha=0.18,
            color=ln.get_color(),
            label="95% CI (approx)",
            zorder=1
        )

    ax1.legend(title=legend_str, loc="best")

    # 残差图
    ax2.axhline(0.0, color="k", lw=1)
    ax2.scatter(m_obs, resid, s=22)
    ax2.set_xlabel("m")
    ax2.set_ylabel("resid")
    ax2.grid(True, linestyle="--", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(out_dir) / filename
    fig.tight_layout()
    fig.savefig(img_path, bbox_inches="tight")
    plt.close(fig)
    return img_path

def _fmt_flo(x, digits=6):
    return f"{x:.{digits}f}" if isinstance(x, (float, int)) and x is not None else "N/A"

def save_results_exp2(cfg: Dict, out: Dict, out_path, results_filename: str = "results.txt"):
    """
        保存实验二结果到文本文件，并生成拟合图。
    """
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / results_filename

    exp = cfg["experiment"]; circuit = cfg["circuit"]
    timing = cfg["timing"];   noise   = cfg["noise"]
    metric = exp.get("metric", "fidelity").lower()
    
    with open(results_file, "w", encoding="utf-8") as f:
        # ======= HEADER =======
        f.write(f"Experiment: {exp['name']}\n")
        f.write(f"Title: {exp['title']}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ======= ALL-ON =======
        f.write("[Results: All-on]\n")
        fit = out["all"]["fit"]
        f.write(f"Fit model: {fit.get('model', 'exponential')}\n")
        f.write("Fitted parameters:\n")
        params = fit.get("params", {})
        stderr = fit.get("param_stderr", {})
        for k, v in params.items():
            f.write(f"    {k} = {v:.6f}")
            if stderr.get(k) is not None:
                f.write(f" ± {stderr[k]:.6f}")
            f.write("\n")

        # 输出 EPC（如果有）
        EPC = out["all"].get("EPC")
        EPC_se = out["all"].get("EPC_stderr")
        if EPC is not None:
            f.write(f"    EPC = {EPC:.8f}")
            if EPC_se is not None:
                f.write(f" ± {EPC_se:.8f}")
            f.write("\n")
        f.write("\n")


        if metric == "fidelity":
            f.write("Mean fidelities (to ideal state):\n")
        else:
            f.write("Mean survival probabilities:\n")
        # Use F̄ or P̄ symbol for average, depending on metric
        avg_symbol = "F\u0304" if metric == "fidelity" else "P\u0304"
        for m, val in zip(out["all"]["m_list"], out["all"]["Pbar"]):
            f.write(f"    m={m:<3d}  {avg_symbol}={val:.6f}\n")
        f.write("\n")

        # ======= ABLATIONS =======
        f.write("[Results: Ablations]\n")
        f.write(f"{'Noise component':<18}{'EPC(-i)':<16}{'Δr':<16}{'Contribution(%)':<16}\n")
        f.write("-" * 66 + "\n")
        for k, v in out["ablations"].items():
            epc_i = v["EPC"]
            delta = out["deltas"][k]
            contrib = 100 * out["shares"][k]
            f.write(f"{k:<18}{_fmt_flo(epc_i):<16}{_fmt_flo(delta):<16}{_fmt_flo(contrib, 2):<16}\n")
        f.write("-" * 66 + "\n")
        sum_delta = sum(x for x in out["deltas"].values() if x is not None)
        f.write(f"Sum Δr = {sum_delta:.6e}\n\n")

        # ======= META INFO =======
        f.write("[Meta Info]\n")
        f.write(f"Qubits: {circuit['n_qubits']}\n")
        gate_names = [g["name"] for g in cfg.get("rb_gate_set", [])] or ["Y90+", "Y90-", "CZ", "IDLE"]
        f.write(f"Gate set: {gate_names}\n")
        probs = [g.get("prob", 1.0) for g in cfg.get("rb_gate_set", [])] or [1.0]*len(gate_names)
        f.write(f"Sampling probs: {probs}\n")
        f.write(f"Metric: { 'Fidelity' if metric=='fidelity' else 'Survival probability' }\n")
        f.write(f"m_list: {exp['m_list']}\n")
        f.write(f"n_sequences: {exp['n_sequences']}\n")
        f.write(f"t1(ns): {timing['t1']}, t2(ns): {timing['t2']}, τ_c(ns): {timing['tau_c']}\n")
        sigma = _get(noise, "flux_quasistatic.sigma", "N/A")
        f.write(f"Flux noise σ: {sigma}\n")
        f.write(f"Simulator: DensityMatrixSimulator\n")
        f.write(f"Seed: {exp.get('seed', 'N/A')}\n")
        enabled = []
        if _get(noise, "idle.enabled", False):            enabled.append("Idle")
        if _get(noise, "ry_gate.enabled", False):         enabled.append("RyGate")
        if _get(noise, "photon_decay.enabled", False):    enabled.append("PhotonDecay")
        if _get(noise, "flux_quasistatic.enabled", False):enabled.append("FluxSegmented")
        f.write(f"Noise models: {', '.join(enabled) or 'None'}\n\n")

        # ======= NOTES =======
        f.write("[Notes]\n")
        f.write("- Fit model is chosen via cfg['experiment']['fit_curve'], options include:\n")
        f.write("  'exponential', 'double_exponential', 'stretched_exponential'.\n")
        f.write("- Parameters are estimated using SciPy's curve_fit with bounds and optional weights.\n")
        if "p" in fit.get("params", {}):
            f.write("- EPC = (d-1)/d * (1 - p). Uncertainty propagated from Var(p).\n")
        f.write("- Negative Δr are clamped to 0 when normalizing shares.\n")


        # === 新增：噪声参数细节 ===
        f.write("\n[Noise Parameters]\n")
        snap = _noise_params_snapshot(noise)
        if not snap:
            f.write("(none)\n\n")
        else:
            def _fmt(v):
                # 对 float 做紧凑格式化，其余原样
                return f"{v:.6g}" if isinstance(v, float) else str(v)
            # 固定顺序更好读
            for section in ["Idle", "RyGate", "PhotonDecay", "FluxSegmented"]:
                if section not in snap:
                    continue
                f.write(f"{section}:\n")
                for k, v in snap[section].items():
                    f.write(f"    {k}: {_fmt(v)}\n")
            f.write("\n")

    print(f"[OK] results -> {results_file}")

    # 生成并保存拟合图
    try:
        img_path = _save_rb_fit_plot1(out, out_dir, filename="rb_fit.png")
        print(f"[OK] plot   -> {img_path}")
    except Exception as e:
        print(f"[WARN] failed to save RB fit plot: {e}")

    # 保存 fid-probe CSV
    probe = out.get("all", {}).get("probe")
    if probe and probe.get("rows"):
        import csv
        csv_path = out_dir / probe.get("outfile", "fid_probe.csv")
        header = probe.get(
            "header",
            ["m","seq_id","P_survival","fidelity_to_ideal","rng_tag_a","rng_tag_b"]
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)          # 只写一次表头
            w.writerows(probe["rows"])  # 纯数据行（tuple/list）
        print(f"[OK] fid-probe -> {csv_path}")

# ==== Interleaved RB 实验函数 ==== #

def run_interleaved_rb_experiment(cfg: Dict, seed: Optional[int] = None) -> Dict:
    """
    针对 gate_set 中每个门分别跑一轮 interleaved RB（与 reference 比较）。
    
    Args:
        cfg: 实验配置字典，需包含 gate_set 与 depth_list
        seed: 随机种子（可选）

    Returns:
        包含 reference 与各 gate 的 EPC、差值等信息
    """

    used_seed = _choose_seed_and_set(cfg, seed)
    # random_gate_set: List[Dict[str, Any]] = fill_rb_gate_samplers(cfg.get("random_gate_set", DEFAULT_RB_GATE_SET))
    random_gate_set: List[Dict[str, Any]] = clifford_1q_gate_set()
    irb_gate_set: List[Dict[str, Any]] = fill_rb_gate_samplers(cfg.get("irb_gate_set", DEFAULT_RB_GATE_SET))
    qs_maximum = cirq.LineQubit.range(int(cfg["circuit"]["n_qubits"]))
    depth_list = list(cfg["experiment"]["m_list"])
    n_seq = int(cfg["experiment"]["n_sequences"])
    d = 2 ** len(qs_maximum)
    use_fid = bool(cfg["experiment"].get("use_fidelity", True))

    timing_cfg = cfg["timing"]
    noise_cfg = cfg["noise"]

    result = {"seed": used_seed, "reference": None, "interleaved": {}, "delta_epc": {}}

    # 1. Reference RB
    ref_res = _run_rb_sweep_custom(
        qubits=qs_maximum,
        gate_set=random_gate_set,
        timing_cfg=timing_cfg,
        noise_cfg=noise_cfg,
        interleaved_gate=None,
        depth_list=depth_list,
        n_seq=n_seq,
        d=d,
        seed=used_seed,
        use_fidelity=use_fid,
    )
    result["reference"] = ref_res

    # 2. Interleaved for each gate
    for g in irb_gate_set:
        gname = g.get("name", "UNK")
        arity = g.get("arity", 1)
        qs = cirq.LineQubit.range(arity)  # 按 arity 选择 qubits
        d = 2 ** len(qs)
        inter_res = _run_rb_sweep_custom(
            qubits=qs,
            gate_set=random_gate_set,
            timing_cfg=timing_cfg,
            noise_cfg=noise_cfg,
            interleaved_gate=g,
            depth_list=depth_list,
            n_seq=n_seq,
            d=d,
            seed=used_seed,
            use_fidelity=use_fid,
        )
        result["interleaved"][gname] = inter_res

        # 计算 delta EPC（参考 - 插入）
        ref_epc = ref_res.get("EPC")
        epc_i = inter_res.get("EPC")
        if ref_epc is not None and epc_i is not None:
            result["delta_epc"][gname] = epc_i - ref_epc
        else:
            result["delta_epc"][gname] = None

    return result

def save_results_interleaved_rb(cfg: Dict, out: Dict, out_path, results_filename: str = "results.txt"):
    """
        保存 interleaved RB 实验结果到文本文件。
    """
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / results_filename

    exp = cfg["experiment"]
    circuit = cfg["circuit"]
    timing = cfg["timing"]
    noise = cfg["noise"]
    metric = exp.get("metric", "fidelity").lower()

    with open(results_file, "w", encoding="utf-8") as f:
        # ======= HEADER =======
        f.write(f"Experiment: {exp['name']} (interleaved RB)\n")
        f.write(f"Title: Interleaved RB for Gate-Specific Error Characterization\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ======= REFERENCE FIT =======
        f.write("[Results]\n")
        fit = out["reference"]["fit"]
        f.write(f"Fit model: {fit.get('model', 'exponential')}\n")
        f.write("Fitted parameters:\n")
        params = fit.get("params", {})
        stderr = fit.get("param_stderr", {})
        for k, v in params.items():
            f.write(f"    {k} = {v:.6f}")
            if stderr.get(k) is not None:
                f.write(f" ± {stderr[k]:.6f}")
            f.write("\n")

        # 输出参考 EPC
        EPC = out["reference"].get("EPC")
        EPC_se = out["reference"].get("EPC_stderr")
        if EPC is not None:
            f.write(f"    EPC = {EPC:.8f}")
            if EPC_se is not None:
                f.write(f" ± {EPC_se:.8f}")
            f.write("\n")
        f.write("\n")

        # ======= PER-GATE RESULTS =======
        f.write("Interleaved gate results:\n")
        for name, res in out["interleaved"].items():
            epc = res.get("EPC", 0.0)
            delta = out["delta_epc"].get(name, 0.0)
            f.write(f"  - {name:<8}: EPC = {epc:.6f}, Δ = {delta:+.6f}\n")
        f.write("\n")

        # ======= META INFO =======
        f.write("[Meta Info]\n")
        f.write(f"Qubits: {circuit['n_qubits']}\n")
        gate_names = [g["name"] for g in cfg.get("rb_gate_set", [])]
        f.write(f"Gate set: {gate_names}\n")
        probs = [g.get("prob", 1.0) for g in cfg.get("rb_gate_set", [])]
        f.write(f"Sampling probs: {probs}\n")
        f.write(f"Metric: { 'Fidelity' if metric=='fidelity' else 'Survival probability' }\n")
        f.write(f"m_list: {exp['m_list']}\n")
        f.write(f"n_sequences: {exp['n_sequences']}\n")
        f.write(f"t1(ns): {timing['t1']}, t2(ns): {timing['t2']}, τ_c(ns): {timing['tau_c']}\n")
        sigma = _get(noise, "flux_quasistatic.sigma", "N/A")
        f.write(f"Flux noise σ: {sigma}\n")
        f.write(f"Simulator: DensityMatrixSimulator\n")
        f.write(f"Seed: {exp.get('seed', 'N/A')}\n")
        enabled = []
        if _get(noise, "idle.enabled", False):            enabled.append("Idle")
        if _get(noise, "ry_gate.enabled", False):         enabled.append("RyGate")
        if _get(noise, "photon_decay.enabled", False):    enabled.append("PhotonDecay")
        if _get(noise, "flux_quasistatic.enabled", False):enabled.append("FluxSegmented")
        f.write(f"Noise models: {', '.join(enabled) or 'None'}\n\n")

        # ======= NOTES =======
        # f.write("[Notes]\n")
        # f.write("- Fit model is chosen via cfg['experiment']['fit_curve'], options include:\n")
        # f.write("  'exponential', 'double_exponential', 'stretched_exponential'.\n")
        # f.write("- Parameters are estimated using SciPy's curve_fit with bounds and optional weights.\n")
        # f.write("- EPC = (d-1)/d * (1 - p). Uncertainty propagated from Var(p).\n\n")

        # ======= NOISE PARAMETERS =======
        f.write("[Noise Parameters]\n")
        snap = _noise_params_snapshot(noise)
        if not snap:
            f.write("(none)\n\n")
        else:
            def _fmt(v):
                return f"{v:.6g}" if isinstance(v, float) else str(v)
            for section in ["Idle", "RyGate", "PhotonDecay", "FluxSegmented"]:
                if section not in snap:
                    continue
                f.write(f"{section}:\n")
                for k, v in snap[section].items():
                    f.write(f"    {k}: {_fmt(v)}\n")
            f.write("\n")

    print(f"[OK] results -> {results_file}")


# if __name__ == "__main__":
#     # 只在你直接 `python experiment1_fid.py` 时执行，
#     # 被 notebook import 时不会执行
#     from experiments.utils import make_run_dir, load_config

#     # 1) 读取 YAML
#     cfg = load_config("exp2_rb.yaml")

#     # 2) 跑实验（返回结果 + 元数据）
#     results = run_exp2_experiment(cfg)

#     run_dir = make_run_dir(exp_name="experiment2")

#     # 3) 保存结果（带 header）

#     save_results_exp2(cfg, results, run_dir)

#     print("[DONE] Experiment1 finished.")