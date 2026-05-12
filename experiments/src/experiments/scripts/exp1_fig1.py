from __future__ import annotations

import csv
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ============================
# low-level helpers
# ============================

def resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        return secrets.randbits(32)
    return int(seed)


def _stable_child_seed(master_seed: int, tag: str) -> int:
    h = hashlib.sha256(f"{master_seed}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def _load_manifest(raw_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = raw_dir / "raw_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"raw_manifest.json not found: {manifest_path}")

    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = obj.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError(f"Invalid or empty manifest: {manifest_path}")

    return sorted(items, key=lambda d: float(d["tau_c"]))


def _get_n_total(item: Dict[str, Any]) -> int:
    shape = item.get("shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) >= 1:
        return int(shape[0])
    if "n_samples" in item:
        return int(item["n_samples"])
    raise KeyError(f"Cannot infer shape/n_total from manifest item: {item}")


def _resolve_metric_filename(item: Dict[str, Any], metric: str) -> str:
    metric = metric.lower().strip()
    key_candidates = {
        "fidelity": ("fidelity1_file", "fid_file", "fid", "fidelity_file"),
        "fidelity1": ("fidelity1_file", "fid_file", "fid", "fidelity_file"),
        "trace_distance": ("trace_distance_file", "td_file", "td", "trace_distance"),
        "td": ("trace_distance_file", "td_file", "td", "trace_distance"),
    }
    if metric not in key_candidates:
        raise ValueError(f"Unsupported metric={metric!r}; use 'fidelity'/'fidelity1' or 'trace_distance'/'td'.")

    for k in key_candidates[metric]:
        if k in item and item[k]:
            return str(item[k])
    raise KeyError(f"Cannot find metric file field for metric={metric!r} in manifest item: {item}")


def _choose_indices(N: int, n: int, m: int, rng: np.random.Generator, method: str) -> np.ndarray:
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    method = method.lower().strip()
    if method == "partition":
        need = n * m
        if N < need:
            raise ValueError(f"partition requires N >= n*m, but N={N}, n*m={need}")
        perm = rng.permutation(N)
        return perm[:need].reshape(n, m)

    if method == "bootstrap":
        return rng.integers(0, N, size=(n, m), endpoint=False)

    raise ValueError(f"Unknown method: {method!r}. Use 'partition' or 'bootstrap'.")


def _summarize_nm(arr: np.ndarray, n: int, m: int, rng: np.random.Generator, method: str) -> Dict[str, float]:
    N = int(arr.shape[0])
    idx = _choose_indices(N, n, m, rng, method)
    x = arr[idx]

    batch_mean = x.mean(axis=1)
    batch_std = x.std(axis=1, ddof=1) if m >= 2 else np.zeros(n, dtype=float)

    mean_of_means = float(batch_mean.mean())
    std_of_means = float(batch_mean.std(ddof=1)) if n >= 2 else 0.0
    sem_of_means = float(std_of_means / np.sqrt(n)) if n >= 1 else float("nan")

    mean_batch_std = float(batch_std.mean())
    std_batch_std = float(batch_std.std(ddof=1)) if n >= 2 else 0.0

    return {
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "sem_of_means": sem_of_means,
        "mean_batch_std": mean_batch_std,
        "std_batch_std": std_batch_std,
    }


def _extract_avg_flux_metric(obj: Any, metric: str) -> Optional[float]:
    metric = metric.lower().strip()
    candidates = {
        "fidelity": ["fidelity1", "fidelity", "mean_fidelity1", "mean_fidelity"],
        "fidelity1": ["fidelity1", "fidelity", "mean_fidelity1", "mean_fidelity"],
        "trace_distance": ["trace_distance", "td", "mean_trace_distance", "mean_td"],
        "td": ["trace_distance", "td", "mean_trace_distance", "mean_td"],
    }[metric]

    if isinstance(obj, (int, float)):
        return float(obj)

    if isinstance(obj, dict):
        for k in candidates:
            if k in obj and isinstance(obj[k], (int, float)):
                return float(obj[k])
        for v in obj.values():
            if isinstance(v, dict):
                out = _extract_avg_flux_metric(v, metric)
                if out is not None:
                    return out
    return None


def _load_avg_flux_value(raw_dir: Path, metric: str) -> Optional[float]:
    path = raw_dir / "average_flux.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _extract_avg_flux_metric(obj, metric)


def _next_postproc_dir(base_dir: Path, *, prefix: str = "postproc", tag: Optional[str] = None) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    if tag is not None:
        out = base_dir / f"{prefix}_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    max_k = 0
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            s = p.name[len(prefix):]
            if s.isdigit():
                max_k = max(max_k, int(s))
    out = base_dir / f"{prefix}{max_k + 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================
# run spec + per-run processing
# ============================

@dataclass(frozen=True)
class CurveSpec:
    run_dir: Optional[str] = None
    run_dirs: Sequence[str] = field(default_factory=tuple)
    label: str = ""
    short_label: Optional[str] = None
    color: Optional[str] = None
    linestyle: str = "-"
    marker: str = "o"

    def normalized_run_dirs(self) -> List[str]:
        dirs: List[str] = []
        if self.run_dir is not None:
            dirs.append(self.run_dir)
        dirs.extend([str(x) for x in self.run_dirs])
        dirs = [str(Path(d)) for d in dirs]
        if not dirs:
            raise ValueError(f"CurveSpec(label={self.label!r}) must provide run_dir or run_dirs")
        return dirs


def _load_metric_array(raw_dir: Path, item: Dict[str, Any], metric: str) -> np.ndarray:
    N_total = _get_n_total(item)
    metric_name = _resolve_metric_filename(item, metric)
    metric_path = raw_dir / metric_name
    if not metric_path.exists():
        raise FileNotFoundError(f"missing metric file for tau_c={item['tau_c']}: {metric_path}")
    arr_mm = np.memmap(metric_path, mode="r", dtype=np.float64, shape=(N_total,))
    return np.asarray(arr_mm, dtype=np.float64)


def _postprocess_single_run_dir(
    run_dir: Path,
    *,
    n: int,
    m: int,
    seed: int,
    method: str,
    metric: str,
) -> Dict[str, Any]:
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    items = _load_manifest(raw_dir)
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, Any]] = []
    for item in items:
        tau_c = float(item["tau_c"])
        arr = _load_metric_array(raw_dir, item, metric)
        N_total = int(arr.shape[0])
        stats = _summarize_nm(arr, int(n), int(m), rng, method)
        rows.append(
            {
                "tau_c": tau_c,
                "N_total": int(N_total),
                "n": int(n),
                "m": int(m),
                "method": str(method),
                **stats,
            }
        )

    rows = sorted(rows, key=lambda r: float(r["tau_c"]))
    avg_flux_value = _load_avg_flux_value(raw_dir, metric)
    return {
        "run_dir": run_dir,
        "rows": rows,
        "avg_flux_value": avg_flux_value,
    }


def _postprocess_aggregate_runs(
    run_dirs: Sequence[Path],
    *,
    n: int,
    m: int,
    seed: int,
    method: str,
    metric: str,
) -> Dict[str, Any]:
    if len(run_dirs) == 1:
        return _postprocess_single_run_dir(run_dirs[0], n=n, m=m, seed=seed, method=method, metric=metric)

    manifests = []
    avg_flux_values = []
    for rd in run_dirs:
        raw_dir = rd / "raw"
        if not raw_dir.exists():
            raise FileNotFoundError(f"raw dir not found: {raw_dir}")
        manifests.append(_load_manifest(raw_dir))
        avg_flux_values.append(_load_avg_flux_value(raw_dir, metric))

    tau_ref = [float(it["tau_c"]) for it in manifests[0]]
    for rd, items in zip(run_dirs[1:], manifests[1:]):
        tau_now = [float(it["tau_c"]) for it in items]
        if tau_now != tau_ref:
            raise ValueError(f"tau_c_list mismatch among replicate runs; offending run={rd}")

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for i, tau_c in enumerate(tau_ref):
        arrs = []
        for rd, items in zip(run_dirs, manifests):
            arrs.append(_load_metric_array(rd / "raw", items[i], metric))
        arr = np.concatenate(arrs, axis=0)
        N_total = int(arr.shape[0])
        stats = _summarize_nm(arr, int(n), int(m), rng, method)
        rows.append(
            {
                "tau_c": float(tau_c),
                "N_total": int(N_total),
                "n": int(n),
                "m": int(m),
                "method": str(method),
                **stats,
            }
        )

    rows = sorted(rows, key=lambda r: float(r["tau_c"]))
    avg_flux_valid = [x for x in avg_flux_values if x is not None]
    avg_flux_value = float(np.mean(avg_flux_valid)) if avg_flux_valid else None
    return {
        "run_dir": ", ".join(str(x) for x in run_dirs),
        "rows": rows,
        "avg_flux_value": avg_flux_value,
    }


# ============================
# writing helpers
# ============================

def _write_multicurve_tsv(path: Path, curve_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "label",
        "run_dir",
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)


def _write_delta_tsv(path: Path, delta_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["label", "ref_label", "tau_c", "delta_mean", "delta_std", "delta_sem"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in delta_rows:
            writer.writerow(row)


def _write_report(
    path: Path,
    *,
    curve_specs: Sequence[CurveSpec],
    metric: str,
    n: int,
    m: int,
    method: str,
    master_seed: int,
    results: Dict[str, Dict[str, Any]],
    reference_label: Optional[str],
) -> None:
    lines: List[str] = []
    lines.append("Exp1 multi-curve post-processing report")
    lines.append(f"metric: {metric}")
    lines.append(f"n (batches): {n}")
    lines.append(f"m (samples per batch): {m}")
    lines.append(f"method: {method}")
    lines.append(f"master resampling seed: {master_seed}")
    lines.append("")
    lines.append("Curves")
    for spec in curve_specs:
        run = results[spec.label]
        lines.append(f"- {spec.label}: run_dir={run['run_dir']}")
    lines.append("")
    lines.append("Summary by curve")

    for spec in curve_specs:
        rows = results[spec.label]["rows"]
        y = np.array([float(r["mean_of_means"]) for r in rows], dtype=float)
        tau = np.array([float(r["tau_c"]) for r in rows], dtype=float)
        best_i = int(np.argmax(y))
        worst_i = int(np.argmin(y))
        lines.append(
            f"- {spec.label}: best tau={tau[best_i]:g}, mean={y[best_i]:.10f}; "
            f"worst tau={tau[worst_i]:.10f}, mean={y[worst_i]:.10f}"
        )

    if reference_label is not None and reference_label in results:
        lines.append("")
        lines.append(f"Delta reference: {reference_label}")
        ref_map = {float(r['tau_c']): r for r in results[reference_label]['rows']}
        for spec in curve_specs:
            if spec.label == reference_label:
                continue
            rows = results[spec.label]["rows"]
            deltas = []
            for r in rows:
                tau = float(r["tau_c"])
                rr = ref_map[tau]
                deltas.append(float(r["mean_of_means"]) - float(rr["mean_of_means"]))
            if deltas:
                lines.append(
                    f"- {spec.label} - {reference_label}: max delta={max(deltas):.10f}, min delta={min(deltas):.10f}"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================
# plotting helpers
# ============================

def _resolve_plot_tau(results: Dict[str, Dict[str, Any]], curve_specs: Sequence[CurveSpec], tau_ticks: Optional[Sequence[float]]) -> np.ndarray:
    if tau_ticks is not None:
        return np.asarray(list(tau_ticks), dtype=float)
    tau_ref = np.array([float(r["tau_c"]) for r in results[curve_specs[0].label]["rows"]], dtype=float)
    return np.sort(tau_ref)


def _apply_log_tau_axis(ax, tau_ticks: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.set_xticks(tau_ticks)
    ax.xaxis.set_major_locator(mticker.FixedLocator(tau_ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{int(t)}" if float(t).is_integer() else f"{t:g}" for t in tau_ticks]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.minorticks_off()


def _plot_multicurve_logx(
    path: Path,
    *,
    curve_specs: Sequence[CurveSpec],
    results: Dict[str, Dict[str, Any]],
    title: Optional[str],
    xlabel: str,
    ylabel: str,
    show_average_flux: bool,
    average_flux_from: Optional[str],
    average_flux_label: str,
    legend_loc: str,
    tau_ticks: Optional[Sequence[float]] = None,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    if show_average_flux:
        avg_flux_value = None
        if average_flux_from is not None and average_flux_from in results:
            avg_flux_value = results[average_flux_from].get("avg_flux_value", None)
        else:
            for spec in curve_specs:
                avg_flux_value = results[spec.label].get("avg_flux_value", None)
                if avg_flux_value is not None:
                    break

        if avg_flux_value is not None:
            ax.axhline(avg_flux_value, color="0.45", lw=1.2, linestyle=(0, (6, 4)), zorder=0, label=average_flux_label)

    for spec in curve_specs:
        rows = results[spec.label]["rows"]
        tau = np.array([float(r["tau_c"]) for r in rows], dtype=float)
        y = np.array([float(r["mean_of_means"]) for r in rows], dtype=float)
        yerr = np.array([float(r["sem_of_means"]) for r in rows], dtype=float)
        idx = np.argsort(tau)
        tau, y, yerr = tau[idx], y[idx], yerr[idx]

        ax.errorbar(
            tau,
            y,
            yerr=yerr,
            fmt=spec.marker,
            linestyle=spec.linestyle,
            capsize=3,
            lw=1.8,
            markersize=5,
            label=spec.short_label or spec.label,
            color=spec.color,
        )

    tau_ref = _resolve_plot_tau(results, curve_specs, tau_ticks)
    _apply_log_tau_axis(ax, tau_ref)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def _plot_delta_logx(
    path: Path,
    *,
    reference_label: str,
    curve_specs: Sequence[CurveSpec],
    results: Dict[str, Dict[str, Any]],
    title: Optional[str],
    xlabel: str,
    ylabel: str,
    legend_loc: str,
    tau_ticks: Optional[Sequence[float]] = None,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    ref_rows = results[reference_label]["rows"]
    ref_map = {float(r["tau_c"]): r for r in ref_rows}
    delta_rows: List[Dict[str, Any]] = []

    for spec in curve_specs:
        if spec.label == reference_label:
            continue
        rows = results[spec.label]["rows"]
        tau = []
        dy = []
        dyerr = []
        for r in rows:
            t = float(r["tau_c"])
            rr = ref_map[t]
            mean_delta = float(r["mean_of_means"]) - float(rr["mean_of_means"])
            sem_delta = float(np.sqrt(float(r["sem_of_means"]) ** 2 + float(rr["sem_of_means"]) ** 2))
            std_delta = float(np.sqrt(float(r["std_of_means"]) ** 2 + float(rr["std_of_means"]) ** 2))
            tau.append(t)
            dy.append(mean_delta)
            dyerr.append(sem_delta)
            delta_rows.append(
                {
                    "label": spec.label,
                    "ref_label": reference_label,
                    "tau_c": t,
                    "delta_mean": mean_delta,
                    "delta_std": std_delta,
                    "delta_sem": sem_delta,
                }
            )

        tau = np.asarray(tau, dtype=float)
        dy = np.asarray(dy, dtype=float)
        dyerr = np.asarray(dyerr, dtype=float)
        idx = np.argsort(tau)
        tau, dy, dyerr = tau[idx], dy[idx], dyerr[idx]
        ax.errorbar(
            tau,
            dy,
            yerr=dyerr,
            fmt=spec.marker,
            linestyle=spec.linestyle,
            capsize=3,
            lw=1.8,
            markersize=5,
            label=f"{spec.short_label or spec.label} - {reference_label}",
            color=spec.color,
        )

    tau_ref_arr = _resolve_plot_tau(results, curve_specs, tau_ticks)
    ax.axhline(0.0, color="0.45", lw=1.0, linestyle=(0, (5, 4)), zorder=0)
    _apply_log_tau_axis(ax, tau_ref_arr)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return delta_rows


# ============================
# public API
# ============================

def postprocess_exp1_multicurve(
    curve_specs: Sequence[CurveSpec | Dict[str, Any]],
    *,
    n: int,
    m: int,
    metric: str = "fidelity",
    seed: Optional[int] = None,
    method: str = "partition",
    output_base_dir: Optional[str | Path] = None,
    postprocs_dirname: str = "postprocs",
    postproc_prefix: str = "postproc",
    postproc_tag: Optional[str] = None,
    out_prefix: str = "exp1_multicurve",
    figure_title: Optional[str] = None,
    xlabel: str = r"$\tau_c$ (ns)",
    ylabel: Optional[str] = None,
    show_average_flux: bool = True,
    average_flux_from: Optional[str] = None,
    average_flux_label: str = "average_flux baseline",
    legend_loc: str = "best",
    reference_label: Optional[str] = None,
    delta_ylabel: Optional[str] = None,
    delta_title: Optional[str] = None,
    tau_ticks: Optional[Sequence[float]] = None,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    delta_ylim: Optional[Sequence[float]] = None,
) -> Dict[str, Path]:
    if not curve_specs:
        raise ValueError("curve_specs must be non-empty")

    specs: List[CurveSpec] = [cs if isinstance(cs, CurveSpec) else CurveSpec(**cs) for cs in curve_specs]
    labels = [s.label for s in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("curve labels must be unique")

    master_seed = resolve_seed(seed)
    metric_norm = metric.lower().strip()
    if ylabel is None:
        ylabel = "mean_of_means (fidelity1)" if metric_norm in ("fidelity", "fidelity1") else "mean_of_means (trace_distance)"
    if delta_ylabel is None:
        delta_ylabel = "Δ mean_of_means"

    first_run_dir = Path(specs[0].normalized_run_dirs()[0])
    if output_base_dir is None:
        base_dir = first_run_dir / postprocs_dirname
    else:
        base_dir = Path(output_base_dir)
    out_dir = _next_postproc_dir(base_dir, prefix=postproc_prefix, tag=postproc_tag)

    results: Dict[str, Dict[str, Any]] = {}
    multicurve_rows: List[Dict[str, Any]] = []
    tau_reference: Optional[List[float]] = None

    for spec in specs:
        child_seed = _stable_child_seed(master_seed, spec.label)
        run_dirs = [Path(p) for p in spec.normalized_run_dirs()]
        run = _postprocess_aggregate_runs(
            run_dirs,
            n=n,
            m=m,
            seed=child_seed,
            method=method,
            metric=metric_norm,
        )
        results[spec.label] = run

        tau_list = [float(r["tau_c"]) for r in run["rows"]]
        if tau_reference is None:
            tau_reference = tau_list
        elif tau_list != tau_reference:
            raise ValueError(
                f"tau_c_list mismatch for label={spec.label!r}.\n"
                f"expected={tau_reference}\n"
                f"got={tau_list}"
            )

        for row in run["rows"]:
            multicurve_rows.append(
                {
                    "label": spec.label,
                    "run_dir": run["run_dir"],
                    **row,
                }
            )

    summary_tsv = out_dir / f"{out_prefix}_{metric_norm}_summary.tsv"
    report_txt = out_dir / f"{out_prefix}_{metric_norm}_report.txt"
    figure_png = out_dir / f"{out_prefix}_{metric_norm}_multicurve.png"

    _write_multicurve_tsv(summary_tsv, multicurve_rows)
    _write_report(
        report_txt,
        curve_specs=specs,
        metric=metric_norm,
        n=n,
        m=m,
        method=method,
        master_seed=master_seed,
        results=results,
        reference_label=reference_label,
    )
    _plot_multicurve_logx(
        figure_png,
        curve_specs=specs,
        results=results,
        title=figure_title,
        xlabel=xlabel,
        ylabel=ylabel,
        show_average_flux=show_average_flux,
        average_flux_from=average_flux_from,
        average_flux_label=average_flux_label,
        legend_loc=legend_loc,
        tau_ticks=tau_ticks,
        xlim=xlim,
        ylim=ylim,
    )

    outputs: Dict[str, Path] = {
        "postproc_dir": out_dir,
        "summary_tsv": summary_tsv,
        "report": report_txt,
        "figure": figure_png,
    }

    if reference_label is not None:
        if reference_label not in results:
            raise ValueError(f"reference_label={reference_label!r} not found in curve_specs")
        delta_png = out_dir / f"{out_prefix}_{metric_norm}_delta.png"
        delta_tsv = out_dir / f"{out_prefix}_{metric_norm}_delta.tsv"
        delta_rows = _plot_delta_logx(
            delta_png,
            reference_label=reference_label,
            curve_specs=specs,
            results=results,
            title=delta_title,
            xlabel=xlabel,
            ylabel=delta_ylabel,
            legend_loc=legend_loc,
            tau_ticks=tau_ticks,
            xlim=xlim,
            ylim=delta_ylim,
        )
        _write_delta_tsv(delta_tsv, delta_rows)
        outputs["delta_figure"] = delta_png
        outputs["delta_tsv"] = delta_tsv

    return outputs


if __name__ == "__main__":
    example_specs = [
        CurveSpec(
            run_dir="/path/to/results/experiment1/20260421-005530",
            label="local_correlated",
            short_label="Local-correlated",
        ),
        CurveSpec(
            run_dir="/path/to/results/experiment1/20260421-015015",
            label="rho095",
            short_label=r"Global-trace-correlated ($\rho=0.95$)",
        ),
        CurveSpec(
            run_dirs=[
                "/path/to/results/experiment1/20260421-025115",
                "/path/to/results/experiment1/20260421-125115",
            ],
            label="rho05",
            short_label=r"Global-trace-correlated ($\rho=0.5$)",
        ),
    ]

    # out = postprocess_exp1_multicurve(
    #     example_specs,
    #     n=50,
    #     m=400,
    #     metric="fidelity",
    #     seed=20260421,
    #     method="partition",
    #     output_base_dir="/path/to/results/experiment1/combined_postprocs",
    #     postproc_tag="figure1_main",
    #     out_prefix="exp1_fig1",
    #     figure_title=None,
    #     ylabel="Mean fidelity",
    #     average_flux_label="Average-flux baseline",
    #     reference_label="local_correlated",
    #     delta_ylabel=r"$\Delta F = F_\rho - F_{\mathrm{local}}$",
    #     tau_ticks=[20, 40, 80, 200, 400, 800, 1200, 2400, 5000, 20000, 40000],
    #     ylim=(0.82, 0.93),
    #     delta_ylim=(-0.005, 0.10),
    # )
    # print(out)
    pass
