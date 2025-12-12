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
from cirq.noise.utils.randomized_benchmarking import _survival_prob_from_dm, FitResult, fit_rb_decay, build_rb1_circuit, DEFAULT_RB_GATE_SET, clifford_1q_gate_set, fill_rb_gate_samplers, _run_rb_sweep_custom
from cirq.noise.utils.metrics import fidelity

# builder（延迟导入模型）
from cirq.noise.utils.noise_builder import (
    _get,
    eval_number,
    make_noise_model
)

from cirq.noise.models.segmented_flux_noise_model import SegmentedFluxNoiseModel

from cirq.noise.utils.compilation_scheme import all_in_one_compile

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
            "tau_d":    eval_number(pd.get("tau_d", 10e-9)),
            "scale":    eval_number(pd.get("scale", 1.0)),
        }

    fx = _get(noise_cfg, "flux_quasistatic", {})
    if fx and fx.get("enabled", True):
        out["FluxSegmented"] = {
            "sigma":    eval_number(fx.get("sigma", 0.05)),
            "seed":     fx.get("seed", None),
        }

    return out

def _run_one_rb1_sweep(cfg: Dict, seed: int, noise_overrides: Dict[str,bool] | None = None) -> Dict:
    """
        运行一轮 RB 序列深度扫描（m_list），返回拟合结果。
        Args:
            cfg: 实验配置字典
            seed: 随机种子
            noise_overrides: 噪声覆写字典，如 {"use_idle": False, "use_ry": True} 等
        Returns: 拟合结果字典
    """

    exp, circ_cfg, timing_cfg, noise_cfg = cfg["experiment"], cfg["circuit"], cfg["timing"], cfg["noise"]

    metric = exp.get("metric", "fidelity").lower()
    
    m_list = list(exp["m_list"]); n_seq = int(exp["n_sequences"])

    rng_seq = np.random.RandomState(int(seed))

    # 测点fid看看
    log_cfg     = cfg.get("logging", {}).get("fid_probe", {}) if isinstance(cfg.get("logging"), dict) else {}
    probe_on    = bool(log_cfg.get("enabled", False))
    probe_k     = int(log_cfg.get("per_m", 0)) if probe_on else 0
    probe_rows  = []   # 收集 (m, seq_id, P_survival, Fid_to_ideal, rng_tag_a, rng_tag_b)

    # 生成序列 & 生存概率
    mean_Pm = []
    seq_vals_all = []  # ← 保存每个 m 的样本数组，便于算 y_std

    m_iter = tqdm(m_list, desc = f"Closed noise models: None" if noise_overrides == None else f"Closed noise models: {list(noise_overrides.keys())}" , unit = "m")
    for m in m_iter:
        vals = []
        seq_id_iter = tqdm(range(n_seq), desc = f"m = {m}", leave = False, unit = "seq_id")
        for seq_id in seq_id_iter:
            # print(f"m: [{m}], seq_id: [{seq_id}]")
            # 1) 随机 RB 电路（深度 = m）
            qs = cirq.LineQubit.range(circ_cfg["n_qubits"])
            filled_gate_set = fill_rb_gate_samplers(cfg.get("rb_gate_set", DEFAULT_RB_GATE_SET))
            # filled_gate_set = clifford_1q_gate_set()
            c0 = build_rb1_circuit(
                qs, depth = m,
                # gate_set=cfg.get("rb_gate_set", DEFAULT_RB_GATE_SET),
                gate_set = clifford_1q_gate_set(),
                # gate_set = filled_gate_set,
                seed = rng_seq.randint(2**31),
                measure = False
            )
            c = all_in_one_compile(c0)

            # print(c)

            # 2) 噪声模型（可覆写开关）
            ncfg_eff = _apply_noise_overrides(noise_cfg, noise_overrides)
            noise_model, _meta = make_noise_model(
                c, timing_cfg, ncfg_eff,
                seed_base = seed,
                tag_a=int(m), tag_b=int(seq_id)
            )
            # 3) 仿真
            sim_noisy = cirq.DensityMatrixSimulator()

            c_noisy = c.with_noise(noise_model)
            # if seq_id == 1:
            #     print(f"m:[{m}], seq_id:[1],circuit with noise:\n{c_noisy}")
            rho_noisy = sim_noisy.simulate(c_noisy).final_density_matrix

            # rho_noisy = fix_matrix_invalid_values(rho_noisy, label=f"rho_noisy (m={m}, seq_id={seq_id})")

            # 4) Compute the chosen metric for this sequence:
            if metric == "fidelity":
                # Simulate ideal circuit (no noise) to get ideal final state
                sim_ideal = cirq.DensityMatrixSimulator()  # no noise by default
                rho_ideal = sim_ideal.simulate(c).final_density_matrix

                # rho_ideal = fix_matrix_invalid_values(rho_ideal, label=f"rho_ideal (m={m}, seq_id={seq_id})")

                # Compute fidelity between ideal and noisy final states
                # fid = float(fidelity(rho_ideal, rho_noisy))
                # 在 fid 计算部分添加一个 while 循环来重试直到成功
                max_retries = 5  # 设置最大重试次数
                retry_count = 0
                fid = float("nan")

                while retry_count < max_retries:
                    try:
                        # 尝试计算 fid
                        fid = float(fidelity(rho_ideal, rho_noisy))
                        vals.append(fid)
                        break  # 如果成功，跳出循环
                    except ValueError as e:  # 捕获 NaN 或 inf 错误
                        print(f"[WARN] fidelity() failed at m={m}, seq_id={seq_id} — {e}. Retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"[ERROR] Maximum retries reached for m={m}, seq_id={seq_id}. Returning NaN.")
                            fid = float("nan")
                            break

                        # 重新生成电路并重新应用噪声模型等
                        qs = cirq.LineQubit.range(circ_cfg["n_qubits"])
                        c0 = build_rb1_circuit(
                            qs, depth=m, gate_set=clifford_1q_gate_set(), seed=rng_seq.randint(2**31), measure=False
                        )
                        c = all_in_one_compile(c0)
                        ncfg_eff = _apply_noise_overrides(noise_cfg, noise_overrides)
                        noise_model, _meta = make_noise_model(c, timing_cfg, ncfg_eff, seed_base=seed, tag_a=int(m), tag_b=int(seq_id))
                        
                        sim_noisy = cirq.DensityMatrixSimulator()
                        c_noisy = c.with_noise(noise_model)
                        rho_noisy = sim_noisy.simulate(c_noisy).final_density_matrix
                        
                        # 如果在最大重试次数内依然无法计算，将返回 NaN
                        if retry_count >= max_retries:
                            # fid = float("nan")
                            break


                # print(fid)
                # vals.append(fid)
            else:  # "survival"
                # Compute survival probability (overlap with initial |0...0⟩ state)
                P_surv = _survival_prob_from_dm(rho_noisy)
                vals.append(P_surv)

            if probe_on and seq_id < probe_k:
                # 理想终态（无噪声）
                rho_ideal = cirq.DensityMatrixSimulator(
                    seed=rng_seq.randint(2**31)
                ).simulate(c).final_density_matrix

                # 复用“本次 noisy”终态：我们已经有 dm，但那是“存活概率”的 dm
                # 为求 fidelity，建议用同一次 noisy 结果；若你想独立采样，可像原来再跑一次。
                rho_noisy = rho_noisy

                # 若 dm 不是密度矩阵对象，确保是 ndarray
                if not np.all(np.isfinite(rho_ideal)):
                    print(f"[DEBUG] rho_ideal contains NaN/inf at m={m}, seq_id={seq_id}")
                if not np.all(np.isfinite(rho_noisy)):
                    print(f"[DEBUG] rho_noisy contains NaN/inf at m={m}, seq_id={seq_id}")


                # fid = float(fidelity(rho_ideal, rho_noisy))
                try:
                    fid = float(fidelity(rho_ideal, rho_noisy))
                except Exception as e:
                    print(f"[WARN] fidelity() failed at m={m}, seq_id={seq_id} — {e}")
                    print("rho_ideal =\n", rho_ideal)
                    print("rho_noisy =\n", rho_noisy)
                    fid = float("nan")


                # 记录纯数据行（tuple），注意 P_survival 是标量
                P_surv = float(np.real(rho_noisy[0, 0]))
                probe_rows.append((
                    int(m),
                    int(seq_id),
                    P_surv,
                    fid,
                    int(m),       # rng_tag_a
                    int(seq_id),  # rng_tag_b
                ))

        mean_Pm.append(np.mean(vals))
        seq_vals_all.append(vals)

    # —— 拟合（锁定 B=1/d，加入 m=0=1.0，并用每个 m 的标准误加权） —— #
    d = 2 ** circ_cfg["n_qubits"]
    y_std = [np.std(v, ddof=1) / max(1, int(len(v)))**0.5 for v in seq_vals_all]

    fit = fit_rb_decay(
        m_list, mean_Pm,
        d=d,
        fit_curve=str(cfg.get("experiment", {}).get("fit_curve", "exponential")),
        y_std=y_std,
        lock_B=True,
        add_m0=True
    )

    probe_info = None
    if probe_on:
        probe_info = {
            "rows": probe_rows,
            "outfile": log_cfg.get("outfile", "fid_probe.csv"),
            "header": ["m","seq_id","P_survival","fidelity_to_ideal","rng_tag_a","rng_tag_b"],
        }

    return {
        "m_list": m_list,
        "Pbar": mean_Pm,
        "fit": {
            "model": fit.model,
            "params": fit.params,
            "param_stderr": fit.param_stderr,
        },
        "EPC": fit.EPC,
        "EPC_stderr": fit.EPC_stderr,
        "probe": probe_info,
    }

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


if __name__ == "__main__":
    # 只在你直接 `python experiment1_fid.py` 时执行，
    # 被 notebook import 时不会执行
    from experiments.utils import make_run_dir, load_config

    # 1) 读取 YAML
    cfg = load_config("exp2_rb.yaml")

    # 2) 跑实验（返回结果 + 元数据）
    results = run_exp2_experiment(cfg)

    run_dir = make_run_dir(exp_name="experiment2")

    # 3) 保存结果（带 header）

    save_results_exp2(cfg, results, run_dir)

    print("[DONE] Experiment1 finished.")