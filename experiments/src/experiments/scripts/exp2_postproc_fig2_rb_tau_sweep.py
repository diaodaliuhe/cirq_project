from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_obs_sem_from_result(res: dict) -> float:
    """
    尽量从已有结果中提取/计算均值的 SEM。
    优先级：
      1) 直接已有 obs_sem
      2) obs_std / sqrt(n_sequences)
      3) 从 sequence-level 原始数组现算
      4) 实在没有就返回 NaN
    """
    if not isinstance(res, dict):
        return np.nan

    if res.get("obs_sem", None) is not None:
        return _safe_float(res.get("obs_sem"))

    obs_std = res.get("obs_std", None)
    n_seq = res.get("n_sequences", None)
    if obs_std is not None and n_seq not in (None, 0):
        n_seq = int(n_seq)
        if n_seq > 0:
            return float(obs_std) / np.sqrt(n_seq)

    candidate_keys = [
        "survival_values",
        "survivals",
        "Y_values",
        "Y_samples",
        "seq_survivals",
        "sequence_survivals",
        "observed_survivals",
    ]
    for key in candidate_keys:
        arr = res.get(key, None)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size >= 2:
            return float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        if arr.size == 1:
            return 0.0

    return np.nan


def _next_postproc_dir(
    base_dir: Path,
    *,
    subdir: str = "postprocs",
    prefix: str = "fig2_postproc",
    tag: Optional[str] = None,
) -> Path:
    root = base_dir / subdir
    root.mkdir(parents=True, exist_ok=True)

    if tag is not None:
        out = root / f"{prefix}_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    max_k = 0
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            s = p.name[len(prefix):]
            if s.isdigit():
                max_k = max(max_k, int(s))
    out = root / f"{prefix}{max_k + 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_rb_tau_json(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "rb_tau_sweep.json"
    if not path.exists():
        raise FileNotFoundError(f"rb_tau_sweep.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_rows(
    data: Dict[str, Any],
    *,
    metric_key: str = "Ybar",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    conds = data.get("conditions", {})
    if not isinstance(conds, dict) or not conds:
        raise ValueError("No conditions found in rb_tau_sweep.json")

    for label, block in conds.items():
        cond = block["condition"]
        res = block["result"]

        tau_c = float(cond["tau_c"])
        m_list = [int(x) for x in res["m_list"]]
        y_list = [float(x) for x in res[metric_key]]

        # 优先尝试每个 m 自己的 sem 列表；没有就退回单个 sem；再没有就 NaN
        obs_stats = res.get("obs_stats", {}) or {}
        sem_list = obs_stats.get("sem", None)

        if sem_list is None:
            one_sem = _compute_obs_sem_from_result(res)
            sem_list = [one_sem] * len(m_list)
        else:
            sem_list = [float(x) for x in sem_list]

        for m, y, sem in zip(m_list, y_list, sem_list):
            row = {
                "tau_c": _safe_float(tau_c),
                "label": str(label),
                "m": int(m),
                "mean_survival": float(y),
                "obs_sem": _safe_float(sem),
                "A": _safe_float(res.get("A", res.get("fit", {}).get("params", {}).get("A"))),
                "p": _safe_float(res.get("p", res.get("fit", {}).get("params", {}).get("p"))),
                "B": _safe_float(res.get("B", res.get("fit", {}).get("params", {}).get("B"))),
                "EPC": _safe_float(res.get("EPC")),
                "RMSE_single": _safe_float(res.get("RMSE_single")),
                "n_sequences": int(res.get("n_sequences", 0) or 0),
            }
            rows.append(row)

    rows.sort(key=lambda r: (r["m"], r["tau_c"]))
    return rows


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("label\ttau_c\tm\tmean_survival\tobs_sem\n")
        for r in rows:
            f.write(
                f"{r['label']}\t"
                f"{float(r['tau_c']):.6f}\t"
                f"{int(r['m'])}\t"
                f"{float(r['mean_survival']):.10f}\t"
                f"{float(r['obs_sem']):.10f}\n"
            )


def _write_report(path: Path, rows: List[Dict[str, Any]], *, run_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    m_values = sorted({int(r["m"]) for r in rows})
    lines: List[str] = []
    lines.append("Fig.2 RB tau sweep report")
    lines.append(f"run_dir: {run_dir}")
    lines.append("")

    for m in m_values:
        sub = [r for r in rows if int(r["m"]) == m]
        tau = np.array([float(r["tau_c"]) for r in sub], dtype=float)
        y = np.array([float(r["mean_survival"]) for r in sub], dtype=float)
        i_min = int(np.argmin(y))
        i_max = int(np.argmax(y))
        lines.append(
            f"m={m}: min=(tau_c={tau[i_min]:.0f}, mean={y[i_min]:.10f}), "
            f"max=(tau_c={tau[i_max]:.0f}, mean={y[i_max]:.10f}), "
            f"dynamic_range={float(y.max() - y.min()):.10f}"
        )

    lines.append("")
    lines.append("Per-point table")
    lines.append("label | tau_c | m | mean_survival | obs_sem")
    lines.append("-" * 90)
    for r in rows:
        lines.append(
            f"{r['label']:>10s} | {float(r['tau_c']):>7.0f} | {int(r['m']):>4d} | "
            f"{float(r['mean_survival']):.10f} | {float(r['obs_sem']):.10f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_fig2(
    fig_base: Path,
    rows: List[Dict[str, Any]],
    *,
    title: str = "RB mean survival vs tau_c at different depths",
    ylabel: str = "Mean survival probability",
    xlabel: str = r"$\tau_c$ (ns)",
    selected_m: Optional[Sequence[int]] = None,
) -> None:
    fig_base.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)

    m_values = sorted({int(r["m"]) for r in rows})
    if selected_m is not None:
        selected = [int(m) for m in selected_m]
        m_values = [m for m in m_values if m in selected]

    for m in m_values:
        sub = [r for r in rows if int(r["m"]) == m]

        tau = np.array([float(r["tau_c"]) for r in sub], dtype=float)
        y = np.array([float(r["mean_survival"]) for r in sub], dtype=float)
        yerr = np.array([float(r["obs_sem"]) for r in sub], dtype=float)

        idx = np.argsort(tau)
        tau = tau[idx]
        y = y[idx]
        yerr = yerr[idx]

        label = rf"$m={m}$"

        if np.isfinite(yerr).any():
            ax.errorbar(
                tau,
                y,
                yerr=yerr,
                fmt="o-",
                linewidth=2,
                markersize=6,
                capsize=3,
                elinewidth=1.2,
                label=label,
            )
        else:
            ax.plot(
                tau,
                y,
                "o-",
                linewidth=2,
                markersize=6,
                label=label,
            )

    ax.set_xscale("log")
    all_tau = sorted({float(r["tau_c"]) for r in rows})
    ax.set_xticks(all_tau)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    ax.minorticks_off()

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    pdf_path = fig_base.with_suffix(".pdf")
    png_path = fig_base.with_suffix(".png")

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()


def postproc_fig2_rb_tau_sweep(
    run_dir: str | Path,
    *,
    title: str = "RB mean survival vs tau_c at different depths",
    ylabel: str = "Mean survival probability",
    selected_m: Optional[Sequence[int]] = None,
    postproc_tag: Optional[str] = None,
) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    data = _load_rb_tau_json(run_dir)
    rows = _extract_rows(data, metric_key="Ybar")

    out_dir = _next_postproc_dir(
        run_dir,
        prefix="fig2_postproc",
        tag=postproc_tag,
    )

    tsv_path = out_dir / "fig2_rb_tau_merged.tsv"
    report_path = out_dir / "fig2_rb_tau_report.txt"
    fig_path = out_dir / "fig2_rb_tau_sweep.png"

    _write_tsv(tsv_path, rows)
    _write_report(report_path, rows, run_dir=run_dir)
    _plot_fig2(
        fig_path,
        rows,
        title=title,
        ylabel=ylabel,
        selected_m=selected_m,
    )

    print(f"[OK] postproc_dir -> {out_dir}")
    print(f"[OK] tsv         -> {tsv_path}")
    print(f"[OK] report      -> {report_path}")
    print(f"[OK] figure      -> {fig_path}")

    return {
        "postproc_dir": out_dir,
        "tsv": tsv_path,
        "report": report_path,
        "figure": fig_path,
    }


if __name__ == "__main__":
    example_run_dir = "/path/to/results/experiment2/YYYYMMDD-HHMMSS"
    print("Edit example_run_dir before running this file directly.")
