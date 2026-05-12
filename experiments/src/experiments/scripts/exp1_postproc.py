from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import secrets


# ----------------------------
# helpers (internal)
# ----------------------------

def _load_manifest(raw_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = raw_dir / "raw_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"raw_manifest.json not found: {manifest_path}")

    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = obj.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError(f"Invalid or empty manifest: {manifest_path}")

    # sort by tau_c
    items = sorted(items, key=lambda d: float(d["tau_c"]))
    return items


def _resolve_fid_filename(item: Dict[str, Any]) -> str:
    # tolerate different key names
    for k in ("fidelity1_file", "fid_file", "fid", "fidelity_file"):
        if k in item and item[k]:
            return str(item[k])
    raise KeyError(f"Cannot find fidelity file field in manifest item: {item}")

def resolve_seed(seed: Optional[int]) -> int:
    """If seed is None, draw a fresh random seed; otherwise cast to int."""
    if seed is None:
        return secrets.randbits(32)
    return int(seed)

def _get_n_total(item: Dict[str, Any]) -> int:
    shape = item.get("shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) >= 1:
        return int(shape[0])
    # fallback: sometimes manifest might store n_samples separately; best-effort
    if "n_samples" in item:
        return int(item["n_samples"])
    raise KeyError(f"Cannot infer shape/n_total from manifest item: {item}")


def _choose_indices(N: int, n: int, m: int, rng: np.random.Generator, method: str) -> np.ndarray:
    """
    Return indices shaped (n, m).
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    method = method.lower().strip()
    if method == "partition":
        need = n * m
        if N < need:
            raise ValueError(f"partition requires N >= n*m, but N={N}, n*m={need}")
        perm = rng.permutation(N)
        idx = perm[:need].reshape(n, m)
        return idx

    if method == "bootstrap":
        idx = rng.integers(0, N, size=(n, m), endpoint=False)
        return idx

    raise ValueError(f"Unknown method: {method!r}. Use 'partition' or 'bootstrap'.")

def _next_postproc_dir(run_dir: Path, *, subdir: str = "postprocs", prefix: str = "postproc",
                       tag: Optional[str] = None) -> Path:
    """
    Create a post-processing output directory:

      run_dir/
        postprocs/
          postproc1/
          postproc2/
          ...

    If tag is provided, create:
      run_dir/postprocs/{prefix}_{tag}/
    """
    base = run_dir / subdir
    base.mkdir(parents=True, exist_ok=True)

    if tag is not None:
        out = base / f"{prefix}_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    # auto-increment
    max_k = 0
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            s = p.name[len(prefix):]  # e.g. "12"
            if s.isdigit():
                max_k = max(max_k, int(s))
    out = base / f"{prefix}{max_k + 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _summarize_nm(arr: np.ndarray, n: int, m: int, rng: np.random.Generator, method: str) -> Dict[str, float]:
    """
    Given raw sample array arr (length N), resample n batches of size m and compute:
      - batch_mean[j] = mean of m samples in batch j
      - batch_std[j]  = std  of m samples in batch j  (ddof=1)
      - mean_of_means = mean(batch_mean)
      - std_of_means  = std(batch_mean) (ddof=1)
      - mean_batch_std, std_batch_std (for report only)
    """
    N = int(arr.shape[0])
    idx = _choose_indices(N, n, m, rng, method)
    x = arr[idx]  # shape (n, m)

    batch_mean = x.mean(axis=1)
    batch_std = x.std(axis=1, ddof=1) if m >= 2 else np.zeros(n, dtype=float)

    mean_of_means = float(batch_mean.mean())
    std_of_means = float(batch_mean.std(ddof=1)) if n >= 2 else 0.0

    mean_batch_std = float(batch_std.mean())
    std_batch_std = float(batch_std.std(ddof=1)) if n >= 2 else 0.0

    # extra: standard error of mean_of_means estimated from batch means
    sem = float(std_of_means / np.sqrt(n)) if n >= 1 else float("nan")

    return {
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "sem_of_means": sem,
        "mean_batch_std": mean_batch_std,
        "std_batch_std": std_batch_std,
    }


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "tau_c",
        "N_total",
        "n",
        "m",
        "method",
        "mean_of_means",
        "std_of_means",
        "sem_of_means",
        "mean_batch_std",
        "std_batch_std",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write(
                "\t".join(
                    [
                        f"{r['tau_c']}",
                        f"{r['N_total']}",
                        f"{r['n']}",
                        f"{r['m']}",
                        f"{r['method']}",
                        f"{r['mean_of_means']:.10f}",
                        f"{r['std_of_means']:.10f}",
                        f"{r['sem_of_means']:.10f}",
                        f"{r['mean_batch_std']:.10f}",
                        f"{r['std_batch_std']:.10f}",
                    ]
                )
                + "\n"
            )


def _write_report(path: Path, rows: List[Dict[str, Any]], *, run_dir: Path, seed: int, n: int, m: int, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tau_list = [float(r["tau_c"]) for r in rows]
    y = np.array([float(r["mean_of_means"]) for r in rows], dtype=float)
    yerr = np.array([float(r["std_of_means"]) for r in rows], dtype=float)

    best_i = int(np.argmax(y))
    worst_i = int(np.argmin(y))

    lines: List[str] = []
    lines.append("Exp1 post-processing (n×m resampling) report")
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"seed: {seed}")
    lines.append(f"n (batches): {n}")
    lines.append(f"m (samples per batch): {m}")
    lines.append(f"method: {method}")
    lines.append("")
    lines.append("Interpretation notes")
    lines.append("- For each tau_c, we form n batches of size m from raw samples.")
    lines.append("- Each batch produces a batch_mean; we then examine variability across n batch_means.")
    lines.append("- std_of_means quantifies *between-batch* variation; sem_of_means is an estimate of uncertainty of mean_of_means.")
    lines.append("- mean_batch_std summarizes *within-batch* variation (not plotted by default).")
    lines.append("")
    lines.append("Key points")
    lines.append(f"- Best mean_of_means:  tau_c={tau_list[best_i]}  mean={y[best_i]:.10f}  std_of_means={yerr[best_i]:.10f}")
    lines.append(f"- Worst mean_of_means: tau_c={tau_list[worst_i]} mean={y[worst_i]:.10f}  std_of_means={yerr[worst_i]:.10f}")
    lines.append("")
    lines.append("Per-tau table (mean_of_means ± std_of_means, plus within-batch std stats)")
    lines.append("tau_c | mean_of_means | std_of_means | sem_of_means | mean_batch_std | std_batch_std")
    lines.append("-" * 86)
    for r in rows:
        lines.append(
            f"{float(r['tau_c']):>6.0f} | "
            f"{r['mean_of_means']:.10f} | "
            f"{r['std_of_means']:.10f} | "
            f"{r['sem_of_means']:.10f} | "
            f"{r['mean_batch_std']:.10f} | "
            f"{r['std_batch_std']:.10f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_logx(path: Path, rows: List[Dict[str, Any]], *, title: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tau = np.array([float(r["tau_c"]) for r in rows], dtype=float)
    y = np.array([float(r["mean_of_means"]) for r in rows], dtype=float)
    yerr = np.array([float(r["std_of_means"]) for r in rows], dtype=float)

    # sort
    idx = np.argsort(tau)
    tau, y, yerr = tau[idx], y[idx], yerr[idx]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(tau, y, yerr=yerr, fmt="o-", capsize=3, lw=1.8, markersize=5)

    ax.set_xscale("log")
    ax.set_xticks(tau)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    ax.minorticks_off()

    ax.set_xlabel(r"$\tau_c$ (ns)")
    ax.set_ylabel("mean_of_means (fidelity1)")

    if title:
        ax.set_title(title)

    ax.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()


# ----------------------------
# ONE public main function
# ----------------------------

def postprocess_exp1_raw(
    run_dir: str | Path,
    *,
    n: int,
    m: int,
    seed: Optional[int] = None,
    method: str = "partition",
    out_prefix: str = None,
    title: Optional[str] = None,
    # NEW:
    postprocs_dirname: str = "postprocs",
    postproc_prefix: str = "postproc",
    postproc_tag: Optional[str] = None,   # e.g. "nm20x500_seed1"
) -> Dict[str, Path]:
    """
    Main entrypoint (single function):
      - Load raw manifest + fid_tau*.dat
      - Resample n×m samples per tau_c
      - Compute mean_of_means/std_of_means (+ within-batch std stats)
      - Write summary TSV + report TXT + plot PNG

    Parameters
    - run_dir: results/experiment1/<timestamp> folder (contains raw/ subdir)
    - n, m:    n batches × m samples per batch
    - seed:    RNG seed for resampling (NOT the simulation seed)
    - method:  'partition' (no replacement, closest to "n independent runs") or 'bootstrap'
    - out_prefix: output file prefix under run_dir/
    """
    seed_used = resolve_seed(seed)
    rng = np.random.default_rng(seed_used)
    print(f"[postproc] seed_used={seed_used}")

    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    out_dir = _next_postproc_dir(
        run_dir,
        subdir=postprocs_dirname,
        prefix=postproc_prefix,
        tag=postproc_tag,
    )

    items = _load_manifest(raw_dir)

    rows: List[Dict[str, Any]] = []
    for item in items:
        tau_c = float(item["tau_c"])
        N_total = _get_n_total(item)
        fid_name = _resolve_fid_filename(item)
        fid_path = raw_dir / fid_name

        if not fid_path.exists():
            raise FileNotFoundError(f"missing raw file for tau_c={tau_c}: {fid_path}")

        fid_mm = np.memmap(fid_path, mode="r", dtype=np.float64, shape=(N_total,))
        stats = _summarize_nm(fid_mm, int(n), int(m), rng, method=str(method))

        row = {
            "tau_c": tau_c,
            "N_total": int(N_total),
            "n": int(n),
            "m": int(m),
            "method": str(method),
            **stats,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: float(r["tau_c"]))

    if out_prefix == None:
        out_prefix = f"nm_{int(n)}x{int(m)}"

    tsv_path    = out_dir / f"{out_prefix}_resample_summary.tsv"
    report_path = out_dir / f"{out_prefix}_resample_report.txt"
    fig_path    = out_dir / f"{out_prefix}_fid_vs_tau_logx.png"

    _write_tsv(tsv_path, rows)
    _write_report(report_path, rows, run_dir=run_dir, seed=seed_used, n=n, m=m, method=method)

    if title != None:
        _plot_logx(fig_path, rows, title=title)
    else:
        title = f"Exp1 fidelity1 vs tau_c ({int(n)}×{int(m)} resampling)"
        _plot_logx(fig_path, rows, title=title)

    return {
        "postproc_dir": out_dir,
        "tsv": tsv_path,
        "report": report_path,
        "figure": fig_path,
    }
#---------------------
#         cn
#---------------------
def _plot_logx_cn(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
    ylabel: str = "平均保真度",
    avg_flux_value: Optional[float] = None,
    avg_flux_label: str = "avg_flux_model",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tau = np.array([float(r["tau_c"]) for r in rows], dtype=float)
    y = np.array([float(r["mean_of_means"]) for r in rows], dtype=float)
    yerr = np.array([float(r["std_of_means"]) for r in rows], dtype=float)

    # sort
    idx = np.argsort(tau)
    tau, y, yerr = tau[idx], y[idx], yerr[idx]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(tau, y, yerr=yerr, fmt="o-", capsize=3, lw=1.8, markersize=5)

    # NEW: avg_flux_model baseline
    if avg_flux_value is not None:
        ax.axhline(
            avg_flux_value,
            color="0.45",
            lw=1.2,
            linestyle=(0, (6, 4)),
            zorder=0,
        )
        ax.text(
            tau[-1],
            avg_flux_value,
            avg_flux_label,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )

    ax.set_xscale("log")
    ax.set_xticks(tau)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    ax.minorticks_off()

    ax.set_xlabel(r"$\tau_c$ (ns)")
    ax.set_ylabel(ylabel)

    # title 可为空；为空时不显示
    if title is not None:
        ax.set_title(title)

    ax.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

def postprocess_exp1_raw_cn(
    run_dir: str | Path,
    *,
    n: int,
    m: int,
    seed: Optional[int] = None,
    method: str = "partition",
    out_prefix: str = None,
    title: Optional[str] = None,
    # NEW:
    postprocs_dirname: str = "postprocs",
    postproc_prefix: str = "postproc",
    postproc_tag: Optional[str] = None,   # e.g. "nm20x500_seed1"

    # NEW: plotting options
    ylabel: str = "平均保真度",
    avg_flux_value: Optional[float] = None,
    avg_flux_label: str = "avg_flux_model",
    show_title: bool = False,
) -> Dict[str, Path]:
    """
    Main entrypoint (single function):
      - Load raw manifest + fid_tau*.dat
      - Resample n×m samples per tau_c
      - Compute mean_of_means/std_of_means (+ within-batch std stats)
      - Write summary TSV + report TXT + plot PNG

    Parameters
    - run_dir: results/experiment1/<timestamp> folder (contains raw/ subdir)
    - n, m:    n batches × m samples per batch
    - seed:    RNG seed for resampling (NOT the simulation seed)
    - method:  'partition' (no replacement, closest to "n independent runs") or 'bootstrap'
    - out_prefix: output file prefix under run_dir/
    """
    seed_used = resolve_seed(seed)
    rng = np.random.default_rng(seed_used)
    print(f"[postproc] seed_used={seed_used}")

    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    out_dir = _next_postproc_dir(
        run_dir,
        subdir=postprocs_dirname,
        prefix=postproc_prefix,
        tag=postproc_tag,
    )

    items = _load_manifest(raw_dir)

    rows: List[Dict[str, Any]] = []
    for item in items:
        tau_c = float(item["tau_c"])
        N_total = _get_n_total(item)
        fid_name = _resolve_fid_filename(item)
        fid_path = raw_dir / fid_name

        if not fid_path.exists():
            raise FileNotFoundError(f"missing raw file for tau_c={tau_c}: {fid_path}")

        fid_mm = np.memmap(fid_path, mode="r", dtype=np.float64, shape=(N_total,))
        stats = _summarize_nm(fid_mm, int(n), int(m), rng, method=str(method))

        row = {
            "tau_c": tau_c,
            "N_total": int(N_total),
            "n": int(n),
            "m": int(m),
            "method": str(method),
            **stats,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: float(r["tau_c"]))

    if out_prefix == None:
        out_prefix = f"nm_{int(n)}x{int(m)}"

    tsv_path    = out_dir / f"{out_prefix}_resample_summary.tsv"
    report_path = out_dir / f"{out_prefix}_resample_report.txt"
    fig_path    = out_dir / f"{out_prefix}_fid_vs_tau_logx.png"

    _write_tsv(tsv_path, rows)
    _write_report(report_path, rows, run_dir=run_dir, seed=seed_used, n=n, m=m, method=method)

    # NEW: 默认不显示标题；show_title=True 时才显示
    plot_title = title if show_title else None

    _plot_logx_cn(
        fig_path,
        rows,
        title=plot_title,
        ylabel=ylabel,
        avg_flux_value=avg_flux_value,
        avg_flux_label=avg_flux_label,
    )

    return {
        "postproc_dir": out_dir,
        "tsv": tsv_path,
        "report": report_path,
        "figure": fig_path,
    }