from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FormatStrFormatter

def _safe_literal_eval(s: str):
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _parse_result_txt_sections(result_txt: Path) -> Dict[str, Dict[str, Any]]:
    text = result_txt.read_text(encoding="utf-8")
    blocks = re.split(r"\n\[Condition: ", text)
    out: Dict[str, Dict[str, Any]] = {}

    for block in blocks[1:]:
        head, *rest = block.split("]\n", 1)
        label = head.strip()
        body = rest[0] if rest else ""
        sec: Dict[str, Any] = {"label": label}
        for line in body.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip().lower()
            val = v.strip()
            sec[key] = _safe_literal_eval(val) if val else val
        out[label] = sec
    return out


def _find_first(existing: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in existing:
            return existing[k]
    return default


def _normalize_fit_params(obj: Any) -> Dict[str, float]:
    if isinstance(obj, dict):
        out: Dict[str, float] = {}
        for k, v in obj.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                pass
        return out
    return {}


def _label_to_tau(label: str) -> float:
    m = re.search(r"tau(\d+(?:\.\d+)?)", label)
    if m:
        return float(m.group(1))
    return math.nan


def _load_condition_npz(npz_path: Path) -> Dict[str, Any]:
    arr = np.load(npz_path, allow_pickle=True)
    data: Dict[str, Any] = {k: arr[k] for k in arr.files}

    out: Dict[str, Any] = {"source": str(npz_path), "label": npz_path.stem}

    m_list = _find_first(data, ["m_list", "ms", "m"], None)
    if m_list is not None:
        out["m_list"] = np.asarray(m_list, dtype=float)

    seq_vals = _find_first(data, ["seq_vals", "samples", "y_samples"], None)
    if seq_vals is not None:
        seq_vals = np.asarray(seq_vals, dtype=float)
        if seq_vals.ndim == 1:
            seq_vals = seq_vals[None, :]
        out["seq_vals"] = seq_vals
        out["obs_mean"] = np.mean(seq_vals, axis=1)
        if seq_vals.shape[1] >= 2:
            std = np.std(seq_vals, axis=1, ddof=1)
        else:
            std = np.zeros(seq_vals.shape[0], dtype=float)
        out["obs_std"] = std
        out["obs_sem"] = std / np.sqrt(max(1, seq_vals.shape[1]))

    mean = _find_first(data, ["obs_mean", "mean", "Ybar", "ybar"], None)
    if mean is not None and "obs_mean" not in out:
        out["obs_mean"] = np.asarray(mean, dtype=float)

    sem = _find_first(data, ["obs_sem", "sem"], None)
    if sem is not None:
        out["obs_sem"] = np.asarray(sem, dtype=float)

    fit_params = _find_first(data, ["fit_params", "params"], None)
    if fit_params is not None:
        if isinstance(fit_params, np.ndarray) and fit_params.dtype == object and fit_params.shape == ():
            fit_params = fit_params.item()
        out["fit_params"] = _normalize_fit_params(fit_params)

    tau_c = _find_first(data, ["tau_c", "tau", "tau_ns"], None)
    if tau_c is not None:
        if isinstance(tau_c, np.ndarray) and tau_c.shape == ():
            tau_c = tau_c.item()
        try:
            out["tau_c"] = float(tau_c)
        except Exception:
            pass

    return out


def _fit_exponential_from_obs(m_list: np.ndarray, y: np.ndarray, sem: np.ndarray | None) -> Dict[str, Any]:
    try:
        from cirq.noise.utils.randomized_benchmarking import fit_rb_decay
    except Exception as exc:
        raise RuntimeError("Cannot import fit_rb_decay from your environment.") from exc

    y_std = None
    if sem is not None:
        y_std = [float(x) for x in np.asarray(sem, dtype=float)]

    fit = fit_rb_decay(
        list(np.asarray(m_list, dtype=int)),
        list(np.asarray(y, dtype=float)),
        d=2,
        fit_curve="exponential",
        y_std=y_std,
        lock_B=True,
        add_m0=True,
    )
    params = {str(k): float(v) for k, v in fit.params.items()}
    return {
        "fit_params": params,
        "epc": float(fit.EPC) if fit.EPC is not None else math.nan,
    }


def _compute_basic_diag(m_list: np.ndarray, y: np.ndarray, params: Dict[str, float]) -> Dict[str, float]:
    if not {"A", "p", "B"}.issubset(params):
        return {"rmse": math.nan, "max_abs_resid": math.nan}
    A = float(params["A"])
    p = float(params["p"])
    B = float(params["B"])
    y_hat = A * (p ** np.asarray(m_list, dtype=float)) + B
    resid = np.asarray(y, dtype=float) - y_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    max_abs = float(np.max(np.abs(resid)))
    return {"rmse": rmse, "max_abs_resid": max_abs}


def _load_run_data(run_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    run_dir = Path(run_dir)
    result_txt_candidates = [run_dir / "result.txt", run_dir / "results_exp2_rb_decay.txt"]
    result_sections: Dict[str, Dict[str, Any]] = {}
    used_result_path: Path | None = None
    for p in result_txt_candidates:
        if p.exists():
            used_result_path = p
            result_sections = _parse_result_txt_sections(p)
            break

    raw_dir = run_dir / "raw"
    out: Dict[str, Dict[str, Any]] = {}

    if raw_dir.exists():
        for npz_path in sorted(raw_dir.glob("*.npz")):
            label = npz_path.stem
            item = _load_condition_npz(npz_path)
            rs = result_sections.get(label, {})

            if "tau_c" not in item:
                tau_from_rs = rs.get("tau_c")
                if tau_from_rs is not None:
                    try:
                        item["tau_c"] = float(tau_from_rs)
                    except Exception:
                        pass
            if "tau_c" not in item:
                item["tau_c"] = _label_to_tau(label)

            if "m_list" not in item and rs.get("m_list") is not None:
                item["m_list"] = np.asarray(rs["m_list"], dtype=float)

            if "fit_params" not in item:
                if rs.get("fit_params") is not None:
                    item["fit_params"] = _normalize_fit_params(rs["fit_params"])

            if "obs_mean" not in item and rs.get("ybar") is not None:
                item["obs_mean"] = np.asarray(rs["ybar"], dtype=float)

            if "epc" not in item and rs.get("epc") is not None:
                try:
                    item["epc"] = float(rs.get("epc"))
                except Exception:
                    pass

            diagnostics = rs.get("diagnostics")
            if isinstance(diagnostics, dict):
                item["diagnostics"] = diagnostics

            # Fallback: recompute fit and diagnostics from raw observed means/sem
            if ("fit_params" not in item or not item["fit_params"]) and "m_list" in item and "obs_mean" in item:
                fit_out = _fit_exponential_from_obs(
                    np.asarray(item["m_list"], dtype=float),
                    np.asarray(item["obs_mean"], dtype=float),
                    np.asarray(item["obs_sem"], dtype=float) if "obs_sem" in item else None,
                )
                item["fit_params"] = fit_out["fit_params"]
                item.setdefault("epc", fit_out["epc"])

            if "diagnostics" not in item and "m_list" in item and "obs_mean" in item and "fit_params" in item:
                item["diagnostics"] = _compute_basic_diag(
                    np.asarray(item["m_list"], dtype=float),
                    np.asarray(item["obs_mean"], dtype=float),
                    item["fit_params"],
                )

            out[label] = item

    if not out and result_sections:
        for label, rs in result_sections.items():
            tau = rs.get("tau_c")
            try:
                tau = float(tau) if tau is not None else _label_to_tau(label)
            except Exception:
                tau = _label_to_tau(label)
            fit_params = _normalize_fit_params(rs.get("fit_params"))
            diagnostics = rs.get("diagnostics") if isinstance(rs.get("diagnostics"), dict) else {}
            out[label] = {
                "label": label,
                "tau_c": tau,
                "fit_params": fit_params,
                "epc": float(rs.get("epc")) if rs.get("epc") is not None else math.nan,
                "diagnostics": diagnostics,
            }

    if not out:
        details = []
        for p in result_txt_candidates:
            details.append(f"exists={p.exists()}:{p}")
        details.append(f"raw_exists={raw_dir.exists()}:{raw_dir}")
        if used_result_path is not None and not result_sections:
            details.append(f"parsed_zero_condition_sections_from={used_result_path}")
        raise FileNotFoundError("No usable exp2 data found. " + " | ".join(details))

    return out


def exp2_postproc_fig4_rb_stats_vs_tau(
    run_dir: str | Path,
    title: str = r"RB statistics vs $\tau_c$",
    outfile_stem: str = "fig4_rb_stats_vs_tau",
) -> Dict[str, str]:
    data = _load_run_data(run_dir)
    rows: List[Dict[str, Any]] = []
    for label, item in data.items():
        params = item.get("fit_params", {}) or {}
        diagnostics = item.get("diagnostics", {}) or {}
        rows.append(
            {
                "label": label,
                "tau_c": float(item.get("tau_c", math.nan)),
                "p_fit": float(params.get("p", math.nan)),
                "epc": float(item.get("epc", math.nan)),
                "rmse_single": float(diagnostics.get("rmse", math.nan)),
                "max_abs_resid": float(diagnostics.get("max_abs_resid", math.nan)),
            }
        )

    rows = [r for r in rows if not math.isnan(r["tau_c"])]
    rows.sort(key=lambda r: r["tau_c"])

    tau = np.asarray([r["tau_c"] for r in rows], dtype=float)
    p_fit = np.asarray([r["p_fit"] for r in rows], dtype=float)
    rmse = np.asarray([r["rmse_single"] for r in rows], dtype=float)
    maxr = np.asarray([r["max_abs_resid"] for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.4), dpi=160, sharex=True)
    
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax1.plot(tau, p_fit, marker="o", lw=2.0)
    ax1.set_xscale("log")
    ax1.set_ylabel("Single-exp fitted $p$")
    ax1.grid(True, linestyle="--", alpha=0.28)

    # RMSE：主曲线
    ax2.plot(
        tau,
        rmse,
        marker="s",
        ms=6.5,
        lw=2.2,
        label="RMSE to single-exp fit",
        zorder=3,
    )

    # max |resid|：辅助曲线（线更细一点）
    ax2.plot(
        tau,
        maxr,
        marker="^",
        ms=6.0,
        lw=1.8,
        alpha=0.95,
        label=r"Max residual",
        zorder=2,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\tau_c$ (ns)")
    ax2.set_ylabel("Deviation from single-exp fit")
    ax2.grid(True, linestyle="--", alpha=0.28)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)

    # fig.suptitle(title)
    fig.tight_layout()

    run_dir = Path(run_dir)
    out_png = run_dir / f"{outfile_stem}.png"
    out_pdf = run_dir / f"{outfile_stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    tsv_path = run_dir / f"{outfile_stem}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("label\ttau_c\tp_fit\tepc\trmse_single\tmax_abs_resid\n")
        for r in rows:
            f.write(
                f"{r['label']}\t{r['tau_c']}\t{r['p_fit']}\t{r['epc']}\t{r['rmse_single']}\t{r['max_abs_resid']}\n"
            )

    return {"png": str(out_png), "pdf": str(out_pdf), "tsv": str(tsv_path)}


if __name__ == "__main__":
    run_dir = "results/experiment2/20260504-xxxxxx"
    paths = exp2_postproc_fig4_rb_stats_vs_tau(run_dir)
    print(paths)
