# cirq-core/cirq/noise/experiments/experiment1.py
from tqdm.auto import tqdm
import cirq, numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime


from cirq.noise.utils.metrics import fidelity, trace_distance
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
)

from cirq.noise.models.segmented_flux_noise_model import SegmentedFluxNoiseModel
from cirq.noise.models.average_flux_noise_model import AverageFluxNoiseModel

from cirq.noise.utils.compilation_scheme import all_in_one_compile

def _require(cfg, path, typ=None):
    v = _get(cfg, path, None)
    if v is None:
        top = list(cfg.keys()) if isinstance(cfg, dict) else type(cfg).__name__
        raise KeyError(f"Config missing '{path}'. Top-level keys: {top}")
    if typ and not isinstance(v, typ):
        raise TypeError(f"Config '{path}' should be {typ}, got {type(v)}: {v!r}")
    return v

# 1) 运行实验：读取 dict（YAML 已在外部加载），返回用于保存/绘图的数据
def run_exp1_experiment(cfg: Dict) -> Tuple[List[Dict], Dict]:
    # ===== 1) 读取配置 & 基本校验 =====  [M1]
    exp_cfg    = _require(cfg, "experiment", dict)
    circ_cfg   = _require(cfg, "circuit", dict)
    timing_cfg = _require(cfg, "timing", dict)
    noise_cfg  = _require(cfg, "noise", dict)

    # ===== 2) 构造电路，并在这里完成“编译流水线” =====
    # 返回：符号电路 c_sym、数值电路 c_raw
    # c_sym, c_raw, _ = build_circuit_from_config(circ_cfg)

    # [M2] 编译：raw → Rz/Ry/CZ → merge ZPow → ASAP
    # c_num = all_in_one_compile(c_raw)

    from supermarq.benchmarks import HamiltonianSimulation
    bc = HamiltonianSimulation(num_qubits=6, time_step=1, total_time = 138)
    c_raw: cirq.Circuit = bc.circuit()[:-1]  # 新版直接 .circuit() 返回 Cirq Circuit

    # from supermarq.benchmarks import BitCode

    # bc = BitCode(num_data_qubits=2, num_rounds=1, bit_state=[0, 1])
    # c: cirq.Circuit = bc.circuit()  # 新版直接 .circuit() 返回 Cirq Circuit

    # c_raw = cirq.Circuit([moment for moment in c if all(not (isinstance(op.gate, cirq.MeasurementGate) or isinstance(op.gate, cirq.ResetChannel)) for op in moment.operations)])

    c_num = all_in_one_compile(c_raw)

    # [M3] qubit 列表统一从编译后电路提取
    qubits = sorted(c_num.all_qubits(), key=lambda q: getattr(q, "x", str(q)))

    # 可选：保存符号电路结构（保持原逻辑）
    # if circ_cfg.get("write_txt"):
    #     with open(circ_cfg["write_txt"], "w") as f:
    #         f.write(str(c_sym))

    # ===== 3) 统计门数/总时长 =====  [M4]
    timing_summary = compute_timing_summary(
        c_num,
        timing_cfg["t1"],
        timing_cfg["t2"],
    )

    # ===== 4) 理想演化（不带噪声） =====
    seed = int(exp_cfg.get("seed", 2025))
    sim_ideal = cirq.DensityMatrixSimulator(seed=seed)
    rho_ideal = sim_ideal.simulate(c_num).final_density_matrix

    # ===== 5) tau_c 列表 & 采样次数 =====  [M5]
    tau_raw = _require(exp_cfg, "tau_c_list")
    if isinstance(tau_raw, (list, tuple)):
        tau_c_list = [float(x) for x in tau_raw]
    elif isinstance(tau_raw, str):
        tau_c_list = [float(s) for s in tau_raw.replace(",", " ").split()]
    else:
        raise TypeError(
            f"'experiment.tau_c_list' should be list/tuple/str, got {type(tau_raw)}"
        )

    n_samples = int(exp_cfg["n_samples"])

    # ===== 6) flux 参数 =====
    flux_sigma, flux_seed = get_flux_sigma_seed(noise_cfg)

    if flux_seed is None:
        base_flux_seed = int(seed)
    else:
        base_flux_seed = int(seed) ^ int(flux_seed)

    # 记录启用的噪声模型名字（只看 YAML 中 enabled 与否）——保持原逻辑
    enabled_names: List[str] = []

    # Idle
    idle_node = noise_cfg.get("idle", {})
    if bool(idle_node.get("enabled", True)):
        enabled_names.append("IdleNoiseModel")

    # Ry gate
    ry_node = noise_cfg.get("ry_gate", {})
    if bool(ry_node.get("enabled", True)):
        enabled_names.append("RyGateNoiseModel")

    # Photon decay  [F1: 这里不再调用 build_photon_decay_from_yaml]
    pd_node = noise_cfg.get("photon_decay", {})
    if bool(pd_node.get("enabled", True)):
        enabled_names.append("PhotonDecayNoiseModel")

    # Flux
    if is_flux_enabled(noise_cfg):
        enabled_names.append("SegmentedFluxNoiseModel")

    results: List[Dict] = []
    samples: Dict[float, Dict[str, List[float]]] = {}

    # ===== 7) 主循环：sweep tau_c =====
    tau_iter = tqdm(tau_c_list, desc="Sweeping τ_c", unit="τ_c")
    for j, tau_c in enumerate(tau_iter):
        fids: List[float] = []
        tds: List[float] = []

        # [M6] 对于给定 tau_c，时序上下文 ctx 是确定的，只需计算一次
        ctx = assign_timed_circuit_context(
            c_num,
            t1=timing_cfg["t1"],
            t2=timing_cfg["t2"],
            tau_c=tau_c,
        )
        max_seg = max(
            info.segment_id
            for infos in ctx.timing_map.values()
            for info in infos
        )

        sample_iter = tqdm(
            range(n_samples),
            desc=f"samples @ τ_c={tau_c}",
            leave=False,
            unit="sample",
        )
        for k in sample_iter:
            # 为本次样本生成种子（保持原逻辑）
            sub_seed = seed + j * 10000 + k

            rng = derive_rng(base_flux_seed, j, k)

            # 1) 采样 δϕ（沿用你原来的写法）
            delta_phis = {
                seg: {q: rng.normal(0, flux_sigma) for q in qubits}
                for seg in range(max_seg + 1)
            }

            # 2) 依据 YAML 构造各噪声模型（和原逻辑一致）
            models = []

            idle = build_idle_from_yaml(timing_cfg, noise_cfg)
            if idle is not None:
                models.append(idle)

            ry = build_ry_from_yaml(timing_cfg, noise_cfg)
            if ry is not None:
                models.append(ry)

            ph = build_photon_decay_from_yaml(noise_cfg, timed_ctx=ctx)
            if ph is not None:
                models.append(ph)

            if is_flux_enabled(noise_cfg):
                models.append(
                    SegmentedFluxNoiseModel(ctx.timing_map, delta_phis)
                )

            composite_model = CompositeNoiseModel(models)

            sim_noisy = cirq.DensityMatrixSimulator(seed=sub_seed)
            rho_noisy = sim_noisy.simulate(
                c_num.with_noise(composite_model)
            ).final_density_matrix

            fids.append(fidelity(rho_ideal, rho_noisy))
            tds.append(trace_distance(rho_ideal, rho_noisy))

        samples[tau_c] = {
            "fidelity": fids,
            "trace_distance": tds,
        }
        results.append(
            {
                "tau_c": tau_c,
                "segs": max_seg + 1,
                "fidelity_mean": float(np.mean(fids)),
                "fidelity_std": float(np.std(fids)),
                "trace_distance_mean": float(np.mean(tds)),
                "trace_distance_std": float(np.std(tds)),
            }
        )

    # =====7.5) 算下 average flux noise 下的 fid 和 td =====

    fids = []
    tds = []
    avg_iter = tqdm(
        range(n_samples),
        desc="Average flux samples",
        unit="sample",
    )
    for k in avg_iter:

        sub_seed = seed + 9999 + k

        models = []

        idle = build_idle_from_yaml(timing_cfg, noise_cfg)
        if idle is not None:
            models.append(idle)

        ry = build_ry_from_yaml(timing_cfg, noise_cfg)
        if ry is not None:
            models.append(ry)

        ph = build_photon_decay_from_yaml(noise_cfg, timed_ctx=ctx)
        if ph is not None:
            models.append(ph)

        if is_flux_enabled(noise_cfg):
            models.append(
                AverageFluxNoiseModel(flux_sigma)
            )

        composite_model = CompositeNoiseModel(models)

        sim_noisy = cirq.DensityMatrixSimulator(seed=sub_seed)
        rho_noisy = sim_noisy.simulate(
            c_num.with_noise(composite_model)
        ).final_density_matrix

        fids.append(fidelity(rho_ideal, rho_noisy))
        tds.append(trace_distance(rho_ideal, rho_noisy))

    samples["average_flux"] = {
        "fidelity": fids,
        "trace_distance": tds,
    }

    results.append(
        {
            "tau_c": "average_flux",
            "segs": None,
            "fidelity_mean": float(np.mean(fids)),
            "fidelity_std": float(np.std(fids)),
            "trace_distance_mean": float(np.mean(tds)),
            "trace_distance_std": float(np.std(tds)),
        }
    )

    # ===== 8) 返回结果与 meta 信息 =====
    return results, {
        "circuit_name": circ_cfg["type"],
        "timing_summary": timing_summary,
        "tau_c_list": tau_c_list,
        "n_samples": n_samples,
        "sigma": flux_sigma,
        "exp_name": exp_cfg["name"],
        "title": exp_cfg["title"],
        "noise_models": enabled_names,
        "seed": seed,
        "n_layers": circ_cfg.get("n_layers"),
        "metrics": cfg.get("metrics", ["fidelity", "trace_distance"]),  # [M7]
        "samples": samples,
    }


# 2) 保存结果：把 header + 表格排版保存
def save_results_exp1(
    results: List[Dict],
    meta: Dict,
    run_dir: str | Path,
    circuit_text: str | None = None,
    results_filename: str = "results.txt",
    samples_filename: str = "samples.tsv",
    circuit_filename: str = "circuit.txt",
    ) -> Dict[str, Path]:
    """
    - 在 run_dir 下写入：
        results.txt（等宽表头 + 对齐列）
        samples.tsv（原始样本，三列：tau_c/metric/value）
        circuit.txt（可选，若提供 circuit_text）
    - 返回写入文件的路径字典
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ts = meta["timing_summary"]
    header = []
    header.append(f"Experiment: {meta['exp_name']}")
    header.append(f"Title: {meta['title']}\n")
    header.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    header.append("[Meta Info]")
    header.append(f"Circuit: {meta['circuit_name']}")
    header.append(f"Qubits: {ts['n_qubits']}")
    header.append(f"Number of Trotter steps: {meta['n_layers']}")
    header.append(f"Circuit depth (moments): {ts['depth']}")
    header.append(f"1q ops: {ts['n_ops_1q']}, 2q ops: {ts['n_ops_2q']}, others: {ts['n_ops_other']}")
    header.append(f"1q gate time (ns): {ts['t1']}, 2q gate time (ns): {ts['t2']}")
    header.append(f"Total duration (ns): {ts['total_duration_ns']:.3f}")
    header.append(f"Noise models: {', '.join(meta['noise_models'])}")
    header.append(f"Samples per τ_c: {meta['n_samples']}")
    header.append(f"τ_c (ns) sweep: {meta['tau_c_list']}")
    header.append(f"Metrics: {meta.get('metrics')}")  # [M7] 新增一行说明
    header.append(f"Seed: {meta.get('seed')}")
    header.append(f"Simulator: DensityMatrixSimulator\n")
    header.append("[Results]")

    # 列宽设置：至少覆盖表头长度
    headers = ["tau_c (ns)", "segs", "mean fidelity", "std", "mean trace_distance", "std"]
    col_widths = [max(len(h), 15) for h in headers]

    # 打印表头
    header_row = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)

    lines = ["\n".join(header), header_row, sep]

    # 打印数据
    for r in results:
        tau_val = r["tau_c"]

        # --- 修改1：允许 tau_c 是字符串标签（如 "average_flux_noise"） ---
        if isinstance(tau_val, (int, float)) and not isinstance(tau_val, bool):
            # 数值型：区分整数/小数
            if isinstance(tau_val, int) or (isinstance(tau_val, float) and tau_val.is_integer()):
                tau_str = f"{int(tau_val):<{col_widths[0]}d}"
            else:
                tau_str = f"{tau_val:<{col_widths[0]}.3f}"
        else:
            # 非数值：直接按字符串左对齐打印
            tau_str = f"{str(tau_val):<{col_widths[0]}}"

        segs_val = r["segs"]
        # --- 修改2：segs 允许为 None 或字符串 ---
        if isinstance(segs_val, int):
            segs_str = f"{segs_val:<{col_widths[1]}d}"
        else:
            # None 打成 "-"，也可以改成 "N/A"
            segs_str = f"{('-' if segs_val is None else str(segs_val)):<{col_widths[1]}}"


        row = [
            tau_str,
            segs_str,
            f"{r['fidelity_mean']:<{col_widths[2]}.10f}",
            f"{r['fidelity_std']:<{col_widths[3]}.10f}",
            f"{r['trace_distance_mean']:<{col_widths[4]}.10f}",
            f"{r['trace_distance_std']:<{col_widths[5]}.10f}",
        ]
        lines.append(" | ".join(row))

    # 写 results.txt
    results_path = run_dir / results_filename
    results_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] results -> {results_path}")

    # 写 samples.tsv
    samples_path = run_dir / samples_filename
    with samples_path.open("w", encoding="utf-8") as f:
        f.write("tau_c\tmetric\tvalue\n")
        for tau, d in meta["samples"].items():
            for v in d["fidelity"]:
                f.write(f"{tau}\tfidelity\t{v:.10f}\n")
            for v in d["trace_distance"]:
                f.write(f"{tau}\ttrace_distance\t{v:.10f}\n")
    print(f"[OK] samples -> {samples_path}")

    # 写 circuit.txt（可选）
    circuit_path = None
    if circuit_text is not None:
        circuit_path = run_dir / circuit_filename
        circuit_path.write_text(circuit_text, encoding="utf-8")
        print(f"[OK] circuit -> {circuit_path}")

    return {
        "results": results_path,
        "samples": samples_path,
        "circuit": circuit_path,
    }


def show():
    print("This is experiment1 module.")

if __name__ == "__main__":
    # 只在你直接 `python experiment1_fid.py` 时执行，
    # 被 notebook import 时不会执行
    from experiments.utils import make_run_dir, load_config

    # 1) 读取 YAML（路径按你自己习惯来）
    cfg = load_config("exp1_fid.yaml")

    # 2) 跑实验
    results, meta = run_exp1_experiment(cfg)

    # 3) 生成运行目录
    run_dir = make_run_dir(exp_name="experiment1")

    # 4) 保存结果
    paths = save_results_exp1(
        results,
        meta,
        run_dir,
        results_filename="results.txt",
        samples_filename="samples.tsv",
        circuit_filename="circuit.txt",
    )

    print("[DONE] Experiment1 finished.")
    for k, p in paths.items():
        if p is not None:
            print(f"  {k}: {p}")