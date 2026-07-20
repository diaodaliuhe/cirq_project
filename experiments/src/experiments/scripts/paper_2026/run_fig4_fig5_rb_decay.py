import argparse
import json
import os
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cirq-project"

from experiments.scripts.experiment2_rb import (
    _compute_basic_fit_diagnostics,
    _prepare_seq_seeds_by_m,
    run_exp2_rb_decay,
    save_results_exp2_rb_decay,
)
from experiments.scripts.exp2_postproc_fig3_rb_multicurve import exp2_postproc_fig3_rb_multicurve
from experiments.scripts.exp2_postproc_fig4_rb_stats_vs_tau import exp2_postproc_fig4_rb_stats_vs_tau


ROOT = Path(__file__).resolve().parents[5]
CONFIG_DIR = ROOT / "experiments" / "src" / "experiments" / "configs" / "paper_2026"
DEFAULT_CONFIG = CONFIG_DIR / "fig4_fig5_rb_decay.yaml"
DEFAULT_OUT_ROOT = ROOT / "results" / "paper_2026"
M_LIST = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _to_builtin(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    return obj


def _git_snapshot(path: Path):
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


def _normalize_config(cfg: dict, args) -> dict:
    cfg = dict(cfg)
    exp = cfg.setdefault("experiment", {})
    exp["name"] = "paper_2026_fig4_fig5_rb_decay"
    exp["title"] = "Paper Fig4/Fig5 RB decay under segmented flux-induced phase noise"
    exp["parallel_mode"] = "grid_tau_m"
    exp["m_list"] = list(M_LIST)
    exp["n_sequences"] = int(args.n_sequences)
    exp["n_workers"] = int(args.n_workers)
    exp["bound_estimate_nseq"] = int(args.bound_estimate_nseq)

    flux = cfg.setdefault("noise", {}).setdefault("flux_quasistatic", {})
    if "flux_seed" not in flux and "seed" in flux:
        flux["flux_seed"] = flux["seed"]
    return cfg


def _make_run_dir(out_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / "paper_fig4_fig5_rb_decay" / f"rb_decay_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_sequence_seeds(cfg: dict, run_dir: Path) -> Path:
    exp = cfg["experiment"]
    seeds = _prepare_seq_seeds_by_m(
        [int(m) for m in exp["m_list"]],
        n_seq=int(exp["n_sequences"]),
        seed=int(exp["seed"]),
    )
    path = run_dir / "sequence_seeds_by_m.json"
    path.write_text(json.dumps(_to_builtin(seeds), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_fit_summary(out: dict, run_dir: Path) -> Path:
    path = run_dir / "fit_summary.tsv"
    header = [
        "label",
        "tau_c",
        "A",
        "p",
        "B",
        "EPC",
        "F_avg",
        "RMSE",
        "max_abs_residual",
    ]
    lines = ["\t".join(header)]
    for label, block in out["conditions"].items():
        cond = block["condition"]
        res = block["result"]
        params = res.get("fit", {}).get("params", {}) or {}
        diag = res.get("diagnostics") or _compute_basic_fit_diagnostics(res)
        epc = res.get("EPC")
        f_avg = None if epc is None else 1.0 - float(epc)
        row = [
            label,
            str(float(cond.get("tau_c"))),
            str(float(params.get("A", float("nan")))),
            str(float(params.get("p", float("nan")))),
            str(float(params.get("B", float("nan")))),
            str(float(epc)) if epc is not None else "nan",
            str(float(f_avg)) if f_avg is not None else "nan",
            str(float(diag.get("rmse", float("nan")))),
            str(float(diag.get("max_abs_resid", float("nan")))),
        ]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_saturation_summary(out: dict, run_dir: Path) -> Path:
    path = run_dir / "saturation_reuse_summary.tsv"
    lines = ["label\ttau_c\tm\tis_reused\tsource_tau_c\tsource_tau_idx"]
    for label, block in out["conditions"].items():
        cond = block["condition"]
        res = block["result"]
        for item in res.get("saturation_reuse_by_m", []) or []:
            source_tau_c = item.get("copied_from_tau_c")
            source_tau_idx = item.get("copied_from_tau_idx")
            lines.append(
                "\t".join(
                    [
                        label,
                        str(float(cond.get("tau_c"))),
                        str(int(item["m"])),
                        str(source_tau_c is not None),
                        "None" if source_tau_c is None else str(float(source_tau_c)),
                        "None" if source_tau_idx is None else str(int(source_tau_idx)),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Paper 2026 Fig. 4/Fig. 5 RB decay workflow."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--n-workers", type=int, default=40)
    parser.add_argument("--n-sequences", type=int, default=300)
    parser.add_argument("--bound-estimate-nseq", type=int, default=30)
    parser.add_argument("--no-postproc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = _normalize_config(_load_yaml(args.config), args)
    run_dir = _make_run_dir(args.out_root)
    config_path = run_dir / "config.yaml"
    _dump_yaml(cfg, config_path)
    seeds_path = _write_sequence_seeds(cfg, run_dir)

    provenance = {
        "implementation_tag": "paper_2026",
        "figure": "paper_fig4_fig5_rb_decay",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "config_source": str(args.config),
        "config": str(config_path),
        "sequence_seeds": str(seeds_path),
        "main_repo": _git_snapshot(ROOT),
        "cirq_repo": _git_snapshot(ROOT / "dev" / "Cirq"),
    }
    provenance_path = run_dir / "provenance.json"
    provenance_path.write_text(json.dumps(_to_builtin(provenance), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] run_dir              -> {run_dir}")
    print(f"[INFO] config               -> {config_path}")
    print(f"[INFO] sequence_seeds       -> {seeds_path}")
    print(f"[INFO] n_sequences          -> {cfg['experiment']['n_sequences']}")
    print(f"[INFO] n_workers            -> {cfg['experiment']['n_workers']}")
    print(f"[INFO] bound_estimate_nseq  -> {cfg['experiment']['bound_estimate_nseq']}")
    print(f"[INFO] conditions           -> {len(cfg['experiment'].get('conditions', []))}")
    print(f"[INFO] m_list               -> {cfg['experiment']['m_list']}")

    if args.dry_run:
        print("[OK] dry run complete")
        return

    out = run_exp2_rb_decay(cfg)
    result_txt = save_results_exp2_rb_decay(
        cfg,
        out,
        run_dir,
        results_filename="results_paper_fig4_fig5_rb_decay.txt",
    )
    json_path = run_dir / "rb_tau_sweep.json"
    json_path.write_text(json.dumps(_to_builtin(out), ensure_ascii=False, indent=2), encoding="utf-8")
    fit_summary_path = _write_fit_summary(out, run_dir)
    saturation_summary_path = _write_saturation_summary(out, run_dir)

    postproc_paths = {}
    if not args.no_postproc:
        postproc_paths["paper_fig4_rb_multicurve"] = exp2_postproc_fig3_rb_multicurve(
            run_dir=run_dir,
            title="Standard 1Q RB under segmented flux-induced phase noise",
            outfile_stem="paper_fig4_rb_multicurve",
            show_errorbar=True,
        )
        postproc_paths["paper_fig5_rb_stats_vs_tau"] = exp2_postproc_fig4_rb_stats_vs_tau(
            run_dir=run_dir,
            title=r"RB fit statistics vs $\tau_c$",
            outfile_stem="paper_fig5_rb_stats_vs_tau",
        )

    summary = {
        **provenance,
        "result_txt": str(result_txt),
        "json": str(json_path),
        "fit_summary": str(fit_summary_path),
        "saturation_reuse_summary": str(saturation_summary_path),
        "postproc": _to_builtin(postproc_paths),
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(_to_builtin(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] result_txt           -> {result_txt}")
    print(f"[OK] json                 -> {json_path}")
    print(f"[OK] fit_summary          -> {fit_summary_path}")
    print(f"[OK] saturation_summary   -> {saturation_summary_path}")
    print(f"[OK] summary              -> {summary_path}")


if __name__ == "__main__":
    main()
