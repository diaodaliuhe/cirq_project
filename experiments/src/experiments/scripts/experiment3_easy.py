import cirq
import numpy as np
from typing import Dict, List, Tuple, Sequence, Any
from collections import Counter
from pathlib import Path
from datetime import datetime
import csv
import copy

from cirq.noise.utils.compilation_scheme import compile_to_Rz_Ry_CZ
from cirq.noise.models import IdleNoiseModel, PhotonDecayNoiseModel, RyGateNoiseModel, SegmentedFluxNoiseModel, sample_flux_noise_segments
from cirq.noise.utils.timed_circuit_context import assign_timed_circuit_context
from cirq.noise.utils.composite_noise_model import CompositeNoiseModel
from cirq.noise.utils.easy_gatelevel_model import identify_gate_name, estimate_circuit_fidelity
from cirq.noise.utils.metrics import fidelity, trace_distance
from cirq.noise.utils.noise_builder import make_noise_model

from cirq.noise.utils.noise_builder import (
    build_idle_from_yaml,
    build_ry_from_yaml,
    build_photon_decay_from_yaml,
    is_flux_enabled,
)

from supermarq.benchmarks import BitCode


def _gate_counts(c: cirq.Circuit) -> Dict[str, int]:
    cnt = Counter()
    for op in c.all_operations():
        name = identify_gate_name(getattr(op, "gate", None))
        if name not in ("IDLE", "MEAS"):
            cnt[name] += 1
    return dict(cnt)

def _as_list(x, default):
    if x is None:
        return [default]
    return x if isinstance(x, list) else [x]

def build_circuit_family(
    circ_cfg: Dict,
    *,
    rng: np.random.RandomState,
) -> List[Tuple[str, cirq.Circuit, Dict[str, int]]]:
    """
    返回 [(circuit_id, circuit_raw, meta)], 其中 meta 至少包含 {'n':..., 'rounds':...}

    支持配置：
      circuit:
        type: BitCode
        num_data_qubits: [3, 4, 5]  # 或单个整数
        num_rounds: [1, 2, 4]       # 或单个整数
        bit_state: [0,0,0,...]      # 可选；若缺省则对每个 n 用全 0 向量

        type: GHZ

    """
    ctype = (circ_cfg.get("type") or "BitCode").strip().lower()
    out: List[Tuple[str, cirq.Circuit, Dict[str, int]]] = []

    if ctype == "bitcode":
        n_list = _as_list(circ_cfg.get("num_data_qubits"), default=3)
        r_list = _as_list(circ_cfg.get("num_rounds"), default=1)
        fixed_bs = circ_cfg.get("bit_state")  # 若提供，按给定 bit_state（需与当前 n 长度一致）

        for n in n_list:
            n = int(n)
            for r in r_list:
                r = int(r)
                if fixed_bs is None:
                    bs = [0] * n
                else:
                    # 容错：长度不匹配时退回全 0
                    bs = list(fixed_bs)
                    if len(bs) != n:
                        bs = [0] * n
                bc = BitCode(num_data_qubits=n, num_rounds=r, bit_state=bs)
                c_raw = bc.circuit()
                cid = f"BitCode_n{n}_r{r}"
                out.append((cid, c_raw, {"n": n, "rounds": r}))
        return out
    
    if ctype == "merminbell":
        n_list = _as_list(circ_cfg.get("num_qubits"), default=3)
        for n in n_list:
            n = int(n)
            from supermarq.benchmarks.mermin_bell import MerminBell
            mb = MerminBell(num_qubits=n)
            c_raw = mb.circuit()
            cid = f"MerminBell_n{n}"
            out.append((cid, c_raw, {"n": n, "rounds": 1}))
        return out

    raise NotImplementedError(f"Unsupported circuit.type={circ_cfg.get('type')!r}")

def remove_measurements(circuit: cirq.Circuit) -> cirq.Circuit:
    return cirq.Circuit(
        op for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.MeasurementGate) and not isinstance(op.gate, cirq.ResetChannel)
    )


def run_exp3_experiment(cfg: Dict, noise_overrides: Dict[str,bool] | None = None) -> Dict[str, Any]:
    exp_cfg    = cfg["experiment"]
    circ_cfg   = cfg.get("circuit", {})
    timing_cfg = cfg.get("timing", {})
    noise_cfg  = cfg.get("noise", {})
    easy_list  = cfg.get("easy_noise", [])

    seed      = int(exp_cfg.get("seed", 2025))

    rng_master = np.random.RandomState(seed)

    # easy_noise → gate_errors（含 Y/Z 兜底：Y=avg(Y90±)，Z 若缺则用 Y）
    gate_errors: Dict[str, float] = {}
    for item in (easy_list or []):
        gate_errors[str(item["name"]).strip()] = float(item["epc"])
    if "Y" not in gate_errors:
        ys = [gate_errors.get("Y90+"), gate_errors.get("Y90-")]
        ys = [v for v in ys if v is not None]
        if ys:
            gate_errors["Y"] = float(np.mean(ys))
    if "Z" not in gate_errors and "Y" in gate_errors:
        gate_errors["Z"] = gate_errors["Y"]

    # 生成电路族（按 n × rounds）
    family = build_circuit_family(circ_cfg, rng=rng_master)

    per_circuit_results: List[Dict[str, Any]] = []

    for cid, c_raw, meta_c in family:
        # 1) 编译到 {Rz, Ry, CZ}
        c_num = compile_to_Rz_Ry_CZ(c_raw)
        c_num = remove_measurements(c_num)

        # 2) easy EPC 预测
        predicted_fid = float(
            estimate_circuit_fidelity(c_num, gate_errors, default_error=0.0, verbose=False)
        )

        model_idle = build_idle_from_yaml(noise_cfg.get("idle", {}))

        # 3) 理想态
        sim_ideal = cirq.DensityMatrixSimulator()
        rho_ideal = sim_ideal.simulate(c_num).final_density_matrix

        # 4) 真实噪声采样（每个样本都重新构建完整 noise model）
        fid_samples, td_samples = [], []


         # 4.3 组装“本次样本”的完整噪声模型（与 make_noise_model 思路一致）
        noise_model, _meta = make_noise_model(
            c_num, timing_cfg, noise_cfg = noise_cfg,
            seed_base = seed,
        )

        # 4.4 仿真（每个样本独立的随机种子）
        # noisy_circuit = c_num.with_noise(noise_model)
        sim_noisy = cirq.DensityMatrixSimulator(noise = noise_model)
        rho_noisy = sim_noisy.simulate(c_num).final_density_matrix
        # print(f"[Debug] Circuit {cid}: simulated noisy circuit:\n{noisy_circuit}")

        fid_samples.append(float(fidelity(rho_ideal, rho_noisy)))
        td_samples.append(float(trace_distance(rho_ideal, rho_noisy)))

        per_circuit_results.append({
            "circuit_id": cid,
            "n": meta_c["n"],
            "rounds": meta_c["rounds"],
            "predicted_fidelity": float(predicted_fid),
            "sim_fidelity_mean": float(np.mean(fid_samples)),
            "sim_fidelity_std":  float(np.std(fid_samples)),
            "sim_trace_mean":    float(np.mean(td_samples)),
            "sim_trace_std":     float(np.std(td_samples)),
            "fid_samples": [float(x) for x in fid_samples],
            "td_samples":  [float(x) for x in td_samples],
            "gate_counts": _gate_counts(c_num),
            "circuit_compiled": str(c_num),
        })

    return {
        "exp_name": exp_cfg.get("name", "exp3"),
        "title":    exp_cfg.get("title", ""),
        "seed":     seed,
        "timing":   timing_cfg,
        "easy_gate_errors": gate_errors,
        "num_circuits": len(per_circuit_results),
        "results": per_circuit_results,
    }

def _fmt_flo(x, nd=6):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _get(d: dict, dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def save_results_exp3(cfg: Dict, out: Dict, out_path, results_filename: str = "results.txt", csv_filename: str = "summary.csv"):
    """
    保存实验三（exp3）结果：
    - 文本：逐电路 (n, rounds) 列表，对比 easy 预测与真实仿真（含均值±方差）
    - CSV：同样的逐电路明细，便于后续画图或表格分析
    """
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / results_filename
    csv_file = out_dir / csv_filename

    exp = cfg["experiment"]
    timing = cfg.get("timing", {})
    noise = cfg.get("noise", {})
    easy_gate_errors = out.get("easy_gate_errors", {})
    results = out.get("results", [])

    # ---- 写文本结果 ----
    with open(results_file, "w", encoding="utf-8") as f:
        # ======= HEADER =======
        f.write(f"Experiment: {exp.get('name','exp3')}\n")
        f.write(f"Title: {exp.get('title','')}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ======= 全局配置 =======
        f.write("[Global Config]\n")
        f.write(f"Seed: {exp.get('seed','N/A')}\n")
        f.write(f"τ_c(ns): {timing.get('tau_c','N/A')}\n")
        enabled = []
        if _get(noise, "idle.enabled", False):            enabled.append("Idle")
        if _get(noise, "ry_gate.enabled", False):         enabled.append("RyGate")
        if _get(noise, "photon_decay.enabled", False):    enabled.append("PhotonDecay")
        if _get(noise, "flux_quasistatic.enabled", False):enabled.append("FluxSegmented")
        f.write(f"Noise models: {', '.join(enabled) or 'None'}\n\n")

        # ======= EASY EPC 表 =======
        f.write("[Easy EPC Table]\n")
        if not easy_gate_errors:
            f.write("(empty)\n\n")
        else:
            keys_sorted = sorted(easy_gate_errors.keys())
            for k in keys_sorted:
                f.write(f"  {k:<6}: {_fmt_flo(easy_gate_errors[k], 6)}\n")
            f.write("\n")

        # ======= 逐电路结果（按 n、rounds 排序） =======
        f.write("[Per-circuit Results]\n")
        f.write(f"{'Circuit':<22} {'n':>3} {'r':>3}  {'Pred(F_easy)':>14}  {'SimFid(mean±std)':>22}  {'ΔF (sim - pred)':>16}  {'SimTrace(mean±std)':>24}\n")
        f.write("-" * 110 + "\n")

        # 排序输出
        sorted_results = sorted(results, key=lambda x: (x.get("n", 0), x.get("rounds", 0)))
        gaps = []  # 统计预测与真实的差距
        for item in sorted_results:
            cid   = item["circuit_id"]
            n     = item["n"]
            r     = item["rounds"]
            pf    = float(item["predicted_fidelity"])
            sfm   = float(item["sim_fidelity_mean"])
            sfs   = float(item["sim_fidelity_std"])
            stm   = float(item["sim_trace_mean"])
            sts   = float(item["sim_trace_std"])
            gap   = sfm - pf
            gaps.append(gap)

            f.write(f"{cid:<22} {n:>3} {r:>3}  {_fmt_flo(pf,6):>14}  {_fmt_flo(sfm,6):>10}±{_fmt_flo(sfs,6):<10}  {_fmt_flo(gap,6):>16}  {_fmt_flo(stm,6):>10}±{_fmt_flo(sts,6):<10}\n")
        f.write("\n")

        # ======= 简要统计（预测 vs 真实） =======
        if gaps:
            import numpy as _np
            gaps = _np.asarray(gaps, dtype=float)
            f.write("[Summary]\n")
            f.write(f"Avg ΔF (sim - pred): {_fmt_flo(_np.mean(gaps),6)}\n")
            f.write(f"Std ΔF (sim - pred): {_fmt_flo(_np.std(gaps),6)}\n")
            f.write("\n")

        # ======= 噪声与计时快照（简版） =======
        f.write("[Timing]\n")
        f.write(f"tau_c: {timing.get('tau_c','N/A')}\n\n")

        f.write("[Flux Noise]\n")
        f.write(f"sigma: {_get(noise,'flux_quasistatic.sigma','N/A')}\n")
        f.write(f"overflow: {_get(noise,'flux_quasistatic.overflow','N/A')}\n\n")

        f.write("[Notes]\n")
        f.write("- Pred(F_easy) 由 easy gate-level EPC 乘积模型估算；\n")
        f.write("- SimFid 为密度矩阵仿真在真实噪声下的平均保真度（±1σ）；\n")
        f.write("- SimTrace 为与理想态的 trace distance 平均值（±1σ）；\n")
        f.write("- ΔF 用于直观比较 easy 估计与真实仿真的偏差。\n")

    # ---- 写 CSV（便于画图/pivot）----
    with open(csv_file, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "circuit_id", "n", "rounds",
            "pred_fid",
            "sim_fid_mean", "sim_fid_std",
            "sim_trace_mean", "sim_trace_std",
            "gap_fid",
            "gate_counts"  # 存成紧凑字符串，方便后续解析
        ])
        for item in sorted(results, key=lambda x: (x.get("n",0), x.get("rounds",0))):
            pf  = float(item["predicted_fidelity"])
            sfm = float(item["sim_fidelity_mean"])
            sfs = float(item["sim_fidelity_std"])
            stm = float(item["sim_trace_mean"])
            sts = float(item["sim_trace_std"])
            gap = sfm - pf
            # gate_counts 打平成 "Y90+:3;Y:1;Z:5;CZ:2"
            gc  = item.get("gate_counts", {})
            gc_str = ";".join(f"{k}:{v}" for k, v in sorted(gc.items()))
            writer.writerow([
                item["circuit_id"], item["n"], item["rounds"],
                f"{pf:.8f}",
                f"{sfm:.8f}", f"{sfs:.8f}",
                f"{stm:.8f}", f"{sts:.8f}",
                f"{gap:.8f}",
                gc_str
            ])

    print(f"[OK] results  -> {results_file}")
    print(f"[OK] summary  -> {csv_file}")