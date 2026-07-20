import argparse
import copy
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cirq-project"

ROOT = Path(__file__).resolve().parents[5]
CIRQ_CORE = ROOT / "dev" / "Cirq" / "cirq-core"
EXPERIMENTS_SRC = ROOT / "experiments" / "src"
for p in (str(CIRQ_CORE), str(EXPERIMENTS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.scripts.experiment1_fid import (
    run_exp1_experiment_mp,
    save_exp1_run_artifacts,
)


CONFIG_DIR = ROOT / "experiments" / "src" / "experiments" / "configs" / "paper_2026"
DEFAULT_OUT_ROOT = ROOT / "results" / "paper_2026"
CONFIGS = [
    (0.03, CONFIG_DIR / "fig2_cpmg_sigma003.yaml"),
    (0.06, CONFIG_DIR / "fig2_cpmg_sigma006.yaml"),
    (0.10, CONFIG_DIR / "fig2_cpmg_sigma010.yaml"),
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _git_snapshot(path: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except Exception:
        status = []
    return {"commit": commit, "status_short": status}


def _sigma_tag(sigma: float) -> str:
    return f"sigma{int(round(float(sigma) * 1000)):03d}"


def _make_run_dir(out_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / "paper_fig2_cpmg" / f"cpmg_fig2_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _normalize_cfg(cfg: dict, sigma: float, n_workers: int, chunk_size: int | None) -> dict:
    cfg = copy.deepcopy(cfg)
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"paper_2026_fig2_cpmg_{_sigma_tag(sigma)}"
    exp["title"] = f"Paper Fig2 CPMG tau sweep, sigma={sigma:g}"
    exp["n_workers"] = int(n_workers)
    if chunk_size is not None:
        exp["chunk_size"] = int(chunk_size)
    noise = cfg.setdefault("noise", {})
    flux = noise.setdefault("flux_quasistatic", {})
    flux["sigma"] = float(sigma)
    if "flux_seed" not in flux and "seed" in flux:
        flux["flux_seed"] = flux["seed"]
    return cfg


def _load_exp1_samples(run_dir: Path, metric: str = "fidelity1") -> dict[float, np.ndarray]:
    manifest_path = run_dir / "raw" / "raw_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing raw manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[float, np.ndarray] = {}
    raw_dir = manifest_path.parent
    for item in manifest.get("items", []):
        tau_c = float(item["tau_c"])
        shape = tuple(int(x) for x in item.get("shape", []))
        if metric == "fidelity1":
            file_name = item["fidelity1_file"]
        elif metric == "trace_distance":
            file_name = item["trace_distance_file"]
        else:
            raise ValueError(metric)
        path = raw_dir / file_name
        if path.suffix == ".npy":
            arr = np.load(path).astype(float)
        else:
            arr = np.memmap(path, mode="r", dtype=np.float64, shape=shape).astype(float)
        out[tau_c] = np.asarray(arr, dtype=float).copy()
    return dict(sorted(out.items()))


def _metric_samples(samples: dict[float, np.ndarray], metric_kind: str) -> dict[float, np.ndarray]:
    if metric_kind == "reported":
        return {tau: np.asarray(vals, dtype=float) for tau, vals in samples.items()}
    if metric_kind == "state":
        return {tau: (3.0 * np.asarray(vals, dtype=float) - 1.0) / 2.0 for tau, vals in samples.items()}
    raise ValueError(metric_kind)


def _summary_rows(run_samples: dict[float, dict[float, np.ndarray]], metric_kind: str) -> list[dict]:
    rows = []
    for sigma, samples in sorted(run_samples.items()):
        converted = _metric_samples(samples, metric_kind)
        for tau_c, vals in sorted(converted.items()):
            sd = float(np.std(vals, ddof=1))
            rows.append(
                {
                    "sigma": float(sigma),
                    "tau_c_ns": float(tau_c),
                    "n_samples": int(len(vals)),
                    f"mean_{metric_kind}_fidelity": float(np.mean(vals)),
                    f"sd_{metric_kind}_fidelity_ddof1": sd,
                    f"sem_{metric_kind}_fidelity": float(sd / np.sqrt(len(vals))),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    return rows


def _endpoint_rows(summary_rows: list[dict], metric_kind: str) -> list[dict]:
    mean_key = f"mean_{metric_kind}_fidelity"
    sem_key = f"sem_{metric_kind}_fidelity"
    by_sigma_tau = {(float(r["sigma"]), float(r["tau_c_ns"])): r for r in summary_rows}
    rows = []
    for sigma in sorted({float(r["sigma"]) for r in summary_rows}):
        r20 = by_sigma_tau[(sigma, 20.0)]
        r40k = by_sigma_tau[(sigma, 40000.0)]
        diff = float(r40k[mean_key] - r20[mean_key])
        se = float(np.sqrt(float(r20[sem_key]) ** 2 + float(r40k[sem_key]) ** 2))
        rows.append(
            {
                "sigma": sigma,
                "mean_tau20": float(r20[mean_key]),
                "mean_tau40000": float(r40k[mean_key]),
                "endpoint_difference": diff,
                "sem_tau20": float(r20[sem_key]),
                "sem_tau40000": float(r40k[sem_key]),
                "se_endpoint_difference": se,
                "difference_over_se": float(diff / se) if se > 0 else np.nan,
                "endpoint_difference_percentage_points": float(100.0 * diff),
            }
        )
    return rows


def _compression_rows(summary_rows: list[dict], metric_kind: str) -> list[dict]:
    mean_key = f"mean_{metric_kind}_fidelity"
    by_sigma_tau = {(float(r["sigma"]), float(r["tau_c_ns"])): r for r in summary_rows}
    d20 = float(by_sigma_tau[(0.03, 20.0)][mean_key] - by_sigma_tau[(0.10, 20.0)][mean_key])
    d40k = float(by_sigma_tau[(0.03, 40000.0)][mean_key] - by_sigma_tau[(0.10, 40000.0)][mean_key])
    reduction = float(1.0 - d40k / d20) if d20 != 0 else np.nan
    return [
        {
            "separation_tau20": d20,
            "separation_tau40000": d40k,
            "reduction_fraction": reduction,
            "reduction_percent": 100.0 * reduction,
        }
    ]


def _write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = list(rows[0].keys())
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(k, "")) for k in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_fig2(summary_rows: list[dict], metric_kind: str, out_stem: Path) -> dict:
    mean_key = f"mean_{metric_kind}_fidelity"
    sem_key = f"sem_{metric_kind}_fidelity"
    ylabel = "Reported fidelity score" if metric_kind == "reported" else "State fidelity"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for sigma in sorted({float(r["sigma"]) for r in summary_rows}):
        sub = sorted([r for r in summary_rows if float(r["sigma"]) == sigma], key=lambda r: float(r["tau_c_ns"]))
        tau = np.asarray([float(r["tau_c_ns"]) for r in sub], dtype=float)
        mean = np.asarray([float(r[mean_key]) for r in sub], dtype=float)
        sem = np.asarray([float(r[sem_key]) for r in sub], dtype=float)
        ax.errorbar(tau, mean, yerr=sem, marker="o", capsize=3, linewidth=1.8, label=fr"$\sigma={sigma:g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau_c$ (ns)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True)
    fig.tight_layout()
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _two_sig(x: float) -> str:
    return f"{float(x):.2g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Paper 2026 Fig. 2 CPMG sweep and postprocessing.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--n-workers", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--preview-max-per-tau", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = _make_run_dir(args.out_root)
    commands = [
        "python run_paper_fig2_cpmg.py",
        f"python run_paper_fig2_cpmg.py --n-workers {args.n_workers}",
    ]

    cfgs: dict[float, dict] = {}
    sigma_run_dirs: dict[float, Path] = {}
    new_samples: dict[float, dict[float, np.ndarray]] = {}

    provenance = {
        "implementation_tag": "paper_2026",
        "figure": "paper_fig2_cpmg",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "main_repo": _git_snapshot(ROOT),
        "cirq_repo": _git_snapshot(ROOT / "dev" / "Cirq"),
        "local_import_paths": [str(EXPERIMENTS_SRC), str(CIRQ_CORE)],
        "n_workers": int(args.n_workers),
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(_to_builtin(provenance), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] run_dir -> {run_dir}")
    print(f"[INFO] n_workers -> {args.n_workers}")
    if args.dry_run:
        for sigma, cfg_path in CONFIGS:
            cfg = _normalize_cfg(_load_yaml(cfg_path), sigma, args.n_workers, args.chunk_size)
            sigma_dir = run_dir / _sigma_tag(sigma)
            _dump_yaml(cfg, sigma_dir / "config.yaml")
            print(f"[DRY] sigma={sigma:g} config -> {sigma_dir / 'config.yaml'}")
        print("[OK] dry run complete")
        return

    for sigma, cfg_path in CONFIGS:
        cfg = _normalize_cfg(_load_yaml(cfg_path), sigma, args.n_workers, args.chunk_size)
        sigma_dir = run_dir / _sigma_tag(sigma)
        sigma_dir.mkdir(parents=True, exist_ok=False)
        cfg.setdefault("experiment", {})["scratch_dir"] = str(sigma_dir / "raw")
        cfgs[float(sigma)] = cfg
        sigma_run_dirs[float(sigma)] = sigma_dir
        _dump_yaml(cfg, sigma_dir / "config.yaml")
        print(f"[RUN] sigma={sigma:g} -> {sigma_dir}", flush=True)
        results, meta = run_exp1_experiment_mp(cfg)
        paths = save_exp1_run_artifacts(
            results,
            meta,
            sigma_dir,
            preview_max_per_tau=int(args.preview_max_per_tau),
            write_raw=True,
            keep_scratch_files=True,
            raw_mode="inplace",
        )
        new_samples[float(sigma)] = _load_exp1_samples(sigma_dir, "fidelity1")
        print(f"[OK] sigma={sigma:g} summary -> {paths['summary']}", flush=True)

    tables_dir = run_dir / "postproc"
    figures_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_reported = _summary_rows(new_samples, "reported")
    endpoint_reported = _endpoint_rows(summary_reported, "reported")
    compression_reported = _compression_rows(summary_reported, "reported")
    summary_state = _summary_rows(new_samples, "state")
    endpoint_state = _endpoint_rows(summary_state, "state")
    compression_state = _compression_rows(summary_state, "state")

    table_paths = {
        "reported_summary": tables_dir / "fig2_cpmg_summary_reported_fidelity.tsv",
        "reported_endpoint": tables_dir / "fig2_cpmg_endpoint_differences_reported_fidelity.tsv",
        "reported_compression": tables_dir / "fig2_cpmg_curve_compression_reported_fidelity.tsv",
        "state_summary": tables_dir / "fig2_cpmg_summary_state_fidelity.tsv",
        "state_endpoint": tables_dir / "fig2_cpmg_endpoint_differences_state_fidelity.tsv",
        "state_compression": tables_dir / "fig2_cpmg_curve_compression_state_fidelity.tsv",
    }
    _write_tsv(summary_reported, table_paths["reported_summary"])
    _write_tsv(endpoint_reported, table_paths["reported_endpoint"])
    _write_tsv(compression_reported, table_paths["reported_compression"])
    _write_tsv(summary_state, table_paths["state_summary"])
    _write_tsv(endpoint_state, table_paths["state_endpoint"])
    _write_tsv(compression_state, table_paths["state_compression"])

    figure_paths = {
        "reported_figure": _plot_fig2(summary_reported, "reported", figures_dir / "paper_fig2_cpmg_reported_fidelity"),
        "state_figure": _plot_fig2(summary_state, "state", figures_dir / "paper_fig2_cpmg_state_fidelity"),
    }

    summary = {
        **provenance,
        "run_dir": str(run_dir),
        "sigma_run_dirs": {str(k): str(v) for k, v in sigma_run_dirs.items()},
        "tables": {k: str(v) for k, v in table_paths.items()},
        "figures": figure_paths,
        "commands": commands,
        "reported_endpoint_two_sig": [
            {
                "sigma": r["sigma"],
                "diff": _two_sig(r["endpoint_difference"]),
                "se": _two_sig(r["se_endpoint_difference"]),
            }
            for r in endpoint_reported
        ],
        "state_endpoint_two_sig": [
            {
                "sigma": r["sigma"],
                "diff": _two_sig(r["endpoint_difference"]),
                "se": _two_sig(r["se_endpoint_difference"]),
            }
            for r in endpoint_state
        ],
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(_to_builtin(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] run_summary -> {run_dir / 'run_summary.json'}")
    print(f"[OK] reported fig -> {figure_paths['reported_figure']}")
    print(f"[OK] state fig    -> {figure_paths['state_figure']}")


if __name__ == "__main__":
    main()
