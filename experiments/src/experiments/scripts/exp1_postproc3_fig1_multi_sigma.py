# experiments/scripts/exp1_postproc3_fig1_multi_sigma.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, List
import re
import numpy as np
import matplotlib.pyplot as plt


def _find_result_txt(run_dir: Path) -> Path:
    """
    在 run_dir 下寻找结果文本文件。
    优先尝试常用文件名；找不到时再扫描所有 txt。
    """
    candidates = [
        run_dir / "experiment_results.txt",
        run_dir / "result.txt",
        run_dir / "results.txt",
    ]

    for p in candidates:
        if p.exists():
            return p

    txts = sorted(run_dir.glob("*.txt"))
    for p in txts:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "[Summary Results]" in text:
                return p
        except Exception:
            continue

    raise FileNotFoundError(f"No result txt found under: {run_dir}")

def _extract_n_samples(result_txt: Path) -> int | None:
    """
    从 summary/result txt 中提取:
        Samples per τ_c: 2000
    或
        Samples per tau_c: 2000
    """
    text = result_txt.read_text(encoding="utf-8", errors="ignore")

    patterns = [
        r"Samples per\s*[τt]au_c\s*:\s*(\d+)",
        r"Samples per\s*τ_c\s*:\s*(\d+)",
        r"Samples per\s*tau_c\s*:\s*(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))

    return None

def _parse_summary_results(
    result_txt: Path,
    metric: str = "fidelity1",
    errorbar_mode: str = "std",   # "std" 或 "sem"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 result/summary txt 的 [Summary Results] 表格中解析:
      tau_c, segs, mean(metric), err(metric)

    errorbar_mode:
      - "std": 返回表格中的 std
      - "sem": 返回 std / sqrt(n_samples)
    """
    text = result_txt.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    in_summary = False
    rows: List[Tuple[float, float, float, float]] = []

    n_samples = _extract_n_samples(result_txt)

    for line in lines:
        s = line.strip()

        if s.startswith("[Summary Results]"):
            in_summary = True
            continue

        if not in_summary:
            continue

        if not s:
            continue

        # 下一个 section 开始就停止
        if s.startswith("[") and not s.startswith("[Summary Results]"):
            break

        # 跳过表头 / 分隔线 / average_flux 行
        if s.startswith("tau_c"):
            continue
        if set(s) <= set("-+| "):
            continue
        if s.startswith("average_flux"):
            continue

        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 6:
            continue

        try:
            tau_c = float(parts[0])
            segs = float(parts[1])

            mean_fid = float(parts[2])
            std_fid = float(parts[3])

            mean_td = float(parts[4])
            std_td = float(parts[5])

            if metric == "fidelity1":
                mean_val = mean_fid
                std_val = std_fid
            elif metric == "trace_distance":
                mean_val = mean_td
                std_val = std_td
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if errorbar_mode == "sem":
                if n_samples is None or n_samples <= 0:
                    raise ValueError(
                        f"Cannot compute SEM because n_samples is missing in {result_txt}"
                    )
                err_val = std_val / np.sqrt(n_samples)
            elif errorbar_mode == "std":
                err_val = std_val
            else:
                raise ValueError(f"Unsupported errorbar_mode: {errorbar_mode}")

            rows.append((tau_c, segs, mean_val, err_val))

        except ValueError:
            continue

    if not rows:
        raise ValueError(f"Failed to parse [Summary Results] from: {result_txt}")

    arr = np.array(rows, dtype=float)
    tau = arr[:, 0]
    segs = arr[:, 1]
    y = arr[:, 2]
    yerr = arr[:, 3]

    order = np.argsort(tau)
    return tau[order], segs[order], y[order], yerr[order]


def exp1_postproc3_fig1_multi_sigma(
    run_dirs: Sequence[str | Path],
    sigma_list: Sequence[float],
    metric: str = "fidelity1",
    title: str | None = None,
    ylabel: str | None = None,
    show_errorbar: bool = False,
    errorbar_mode: str = "sem",   # 新增：默认画 SEM
    logx: bool = True,
    marker: str = "o",
    linewidth: float = 1.8,
    markersize: float = 4.5,
    markerfacecolor: str = "none",
    markeredgewidth: float = 1.0,
    capsize: float = 2.5,
    elinewidth: float = 1.0,
    figsize: tuple[float, float] = (7.2, 4.8),
    dpi: int = 160,
    postproc_tag: str = "fig1_multi_sigma",
):
    """
    将不同 sigma 的多次 exp1 结果画在同一张图上。
    """

    if len(run_dirs) != len(sigma_list):
        raise ValueError(
            f"len(run_dirs)={len(run_dirs)} != len(sigma_list)={len(sigma_list)}"
        )

    run_dirs = [Path(p) for p in run_dirs]

    # 解析各条曲线
    series = []
    for run_dir, sigma in zip(run_dirs, sigma_list):
        result_txt = _find_result_txt(run_dir)
        tau, segs, y, yerr = _parse_summary_results(
            result_txt,
            metric=metric,
            errorbar_mode=errorbar_mode if show_errorbar else "std",
        )
        series.append(
            {
                "run_dir": run_dir,
                "sigma": float(sigma),
                "tau": tau,
                "segs": segs,
                "y": y,
                "yerr": yerr,
                "result_txt": result_txt,
            }
        )

    # 按 sigma 排序，保证图例顺序稳定
    series.sort(key=lambda d: d["sigma"])

    # if title is None:
    #     if metric == "fidelity1":
    #         title = "CPMG fidelity vs tau_c under different flux strengths"
    #     else:
    #         title = "CPMG trace distance vs tau_c under different flux strengths"

    if ylabel is None:
        ylabel = "Fidelity" if metric == "fidelity1" else "Trace distance"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for item in series:
        sigma = item["sigma"]
        tau = item["tau"]
        y = item["y"]
        yerr = item["yerr"]

        label = rf"$\sigma={sigma:.2f}$"

        if show_errorbar:
            ax.errorbar(
                tau,
                y,
                yerr=yerr,
                fmt="o-",
                linewidth=1.8,
                markersize=4.4,     # 原来 5.5，降一点就够
                capsize=2.4,        # 原来 3，略收
                elinewidth=0.9,     # 原来 1.2，细一点
                capthick=0.9,
                zorder=2,           # 误差棒在下层
                label=label,
            )
        else:
            ax.plot(
                tau,
                y,
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                label=label,
            )

    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(r"$\tau_c$ (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(frameon=True)

    fig.tight_layout()

    # 输出目录：放在这些 run 的共同父目录下
    common_parent = Path(Path(run_dirs[0]).parent)
    out_dir = common_parent / f"postproc_{postproc_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_base = out_dir / f"{metric}_multi_sigma"
    pdf_path = fig_base.with_suffix(".pdf")
    png_path = fig_base.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved figure -> {fig_base}")

    # 也顺便打印一下每条曲线的基本信息，便于 sanity check
    print("\n[Loaded series]")
    for item in series:
        print(
            f"  sigma={item['sigma']:.2f} | "
            f"n_points={len(item['tau'])} | "
            f"tau_min={item['tau'][0]} | tau_max={item['tau'][-1]} | "
            f"src={item['result_txt']}"
        )

    return {
        "figure_pdf": pdf_path,
        "figure_png": png_path,
        "series": series,
    }