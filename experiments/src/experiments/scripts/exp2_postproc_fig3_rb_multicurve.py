from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cirq.noise.utils.randomized_benchmarking import fit_rb_decay

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
    """
    兼容三种输入：
      1. 真 dict，例如 {"A": 0.5, "p": 0.989, "B": 0.5}
      2. numpy 标量包装过的 dict
      3. result.txt 里那种字符串：
         "{'A': np.float64(...), 'p': np.float64(...), 'B': 0.5}"
    """
    out: Dict[str, float] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                pass
        return out

    if isinstance(obj, str):
        s = obj.strip()

        # 先直接尝试 ast.literal_eval
        try:
            tmp = ast.literal_eval(s)
            if isinstance(tmp, dict):
                for k, v in tmp.items():
                    try:
                        out[str(k)] = float(v)
                    except Exception:
                        pass
                if out:
                    return out
        except Exception:
            pass

        # 再处理 np.float64(...) 这种形式
        # 例如 "'A': np.float64(0.4999999)"
        pat = re.compile(
            r"""['"]?(A|p|B)['"]?\s*:\s*(?:np\.float64\()?\s*([-+0-9.eE]+)\s*\)?"""
        )
        for k, v in pat.findall(s):
            try:
                out[str(k)] = float(v)
            except Exception:
                pass

    return out


def _load_condition_npz(npz_path: Path) -> Dict[str, Any]:
    arr = np.load(npz_path, allow_pickle=True)
    data: Dict[str, Any] = {k: arr[k] for k in arr.files}

    out: Dict[str, Any] = {"source": str(npz_path), "label": npz_path.stem}

    m_list = _find_first(data, ["m_list", "ms", "m"], None)
    if m_list is not None:
        out["m_list"] = np.asarray(m_list, dtype=float).tolist()

    seq_vals = _find_first(data, ["seq_vals", "samples", "y_samples"], None)
    if seq_vals is not None:
        seq_vals = np.asarray(seq_vals, dtype=float)
        if seq_vals.ndim == 1:
            seq_vals = seq_vals[None, :]
        out["seq_vals"] = seq_vals

        obs_mean = np.mean(seq_vals, axis=1)
        if seq_vals.shape[1] >= 2:
            obs_std = np.std(seq_vals, axis=1, ddof=1)
        else:
            obs_std = np.zeros(seq_vals.shape[0], dtype=float)
        obs_sem = obs_std / np.sqrt(max(1, seq_vals.shape[1]))

        out.setdefault("obs_mean", obs_mean.tolist())
        out.setdefault("obs_std", obs_std.tolist())
        out.setdefault("obs_sem", obs_sem.tolist())

    mean = _find_first(data, ["obs_mean", "mean", "Ybar", "ybar"], None)
    if mean is not None:
        out["obs_mean"] = np.asarray(mean, dtype=float).tolist()

    sem = _find_first(data, ["obs_sem", "sem"], None)
    if sem is not None:
        out["obs_sem"] = np.asarray(sem, dtype=float).tolist()

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


def _load_run_data(run_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    run_dir = Path(run_dir)
    result_txt_candidates = [run_dir / "result.txt", run_dir / "results_exp2_rb_decay.txt"]
    result_sections: Dict[str, Dict[str, Any]] = {}
    for p in result_txt_candidates:
        if p.exists():
            result_sections = _parse_result_txt_sections(p)
            break

    raw_dir = run_dir / "raw"
    out: Dict[str, Dict[str, Any]] = {}

    if raw_dir.exists():
        for npz_path in sorted(raw_dir.glob("*.npz")):
            label = npz_path.stem
            item = _load_condition_npz(npz_path)

            if label in result_sections:
                rs = result_sections[label]

                item.setdefault("tau_c", rs.get("tau_c"))
                item.setdefault("m_list", rs.get("m_list"))

                # 关键：这里不要只接受“已经是 dict 的 fit_params”
                if "fit_params" not in item or not item["fit_params"]:
                    fp = rs.get("fit_params")
                    if fp is not None:
                        item["fit_params"] = _normalize_fit_params(fp)

                if "obs_mean" not in item and rs.get("ybar") is not None:
                    item["obs_mean"] = list(np.asarray(rs["ybar"], dtype=float))

                if "obs_sem" not in item and rs.get("obs_sem") is not None:
                    item["obs_sem"] = list(np.asarray(rs["obs_sem"], dtype=float))

            out[label] = item

    if not out and result_sections:
        for label, rs in result_sections.items():
            item = {
                "label": label,
                "tau_c": rs.get("tau_c"),
                "m_list": rs.get("m_list"),
                "obs_mean": rs.get("ybar"),
                "fit_params": _normalize_fit_params(rs.get("fit_params")),
            }
            out[label] = item

    return out

def _ensure_fit_params_from_data(item: Dict[str, Any]) -> Dict[str, float]:
    """
    若 item 里没有 fit_params，则用 (m_list, obs_mean, obs_sem) 现算一遍单指数拟合。
    模型仍与你正文保持一致：
        P(m) = A * p^m + B
    其中 fit_rb_decay(...) 会按你当前 exp2 的标准流程跑。
    """
    params = item.get("fit_params", {}) or {}
    if {"A", "p", "B"}.issubset(params):
        return params

    m = item.get("m_list")
    y = item.get("obs_mean")
    sem = item.get("obs_sem", None)

    if m is None or y is None:
        return {}

    m = list(np.asarray(m, dtype=float))
    y = list(np.asarray(y, dtype=float))
    if sem is not None:
        sem = list(np.asarray(sem, dtype=float))

    try:
        fit = fit_rb_decay(
            m,
            y,
            d=2,
            fit_curve="exponential",
            y_std=sem,
            lock_B=True,
            add_m0=True,
        )
        return {
            "A": float(fit.params["A"]),
            "p": float(fit.params["p"]),
            "B": float(fit.params["B"]),
        }
    except Exception:
        return {}


def exp2_postproc_fig3_rb_multicurve(
    run_dir: str | Path,
    labels: Optional[List[str]] = None,
    title: str = "Standard 1Q RB under segmented flux noise",
    outfile_stem: str = "fig3_rb_multicurve",
    show_errorbar: bool = True,
) -> Dict[str, str]:
    """
    图三：多条 RB 曲线同图 + 各自单指数拟合参考线
    """
    run_dir = Path(run_dir)
    data = _load_run_data(run_dir)
    if not data:
        raise FileNotFoundError(f"No usable exp2 data found under {run_dir}")

    if labels is None:
        preferred = ["tau20", "tau400", "tau1200", "tau20000"]
        labels = [x for x in preferred if x in data]
        if len(labels) < 2:
            pairs = sorted(
                [(float(v.get("tau_c", math.inf)), k) for k, v in data.items() if v.get("tau_c") is not None],
                key=lambda x: x[0],
            )
            if len(pairs) <= 4:
                labels = [k for _, k in pairs]
            else:
                idx = [0, len(pairs)//3, 2*len(pairs)//3, len(pairs)-1]
                labels = [pairs[i][1] for i in idx]

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)

    curve_handles = []
    curve_labels = []

    for i, label in enumerate(labels):
        item = data[label]
        m = np.asarray(item["m_list"], dtype=float)
        y = np.asarray(item["obs_mean"], dtype=float)
        sem = np.asarray(item.get("obs_sem", np.zeros_like(y)), dtype=float)
        params = _ensure_fit_params_from_data(item)
        print(f"[fig3] {label}: fit_params={params}")
        tau_c = item.get("tau_c")

        if tau_c is None:
            m_tau = re.match(r"tau([0-9]+(?:\.[0-9]+)?)$", str(label))
            if m_tau:
                tau_c = float(m_tau.group(1))

        if tau_c is not None:
            if float(tau_c).is_integer():
                tau_text = rf"$\tau_c={int(tau_c)}$ ns"
            else:
                tau_text = rf"$\tau_c={tau_c:g}$ ns"
        else:
            tau_text = label
        mk = markers[i % len(markers)]
        color = colors[i % len(colors)]

        # 先画各自的单指数拟合参考线
        # P(m) = A * p^m + B
        if {"A", "p", "B"}.issubset(params):
            A = float(params["A"])
            p = float(params["p"])
            B = float(params["B"])
            xfit = np.linspace(0.0, float(np.max(m)), 500)
            yfit = A * (p ** xfit) + B
            ax.plot(
                xfit,
                yfit,
                lw=2.0,
                color=color,
                alpha=0.95,
                zorder=1,
            )

        # 再画 mean ± SEM，尽量画轻一点
        if show_errorbar and sem.size == y.size:
            ax.errorbar(
                m, y, yerr=sem,
                fmt=mk,
                linestyle="none",
                ms=5.0,
                capsize=2.0,
                elinewidth=0.9,
                capthick=0.9,
                color=color,
                ecolor=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                alpha=0.98,
                zorder=3,
            )
        else:
            ax.plot(
                m, y,
                mk,
                ms=5.0,
                color=color,
                markerfacecolor="white",
                markeredgewidth=1.3,
                linestyle="none",
                zorder=3,
            )

        # 主 legend 只保留 tau 曲线身份
        curve_handles.append(
            Line2D(
                [0], [0],
                color=color,
                marker=mk,
                linestyle="none",
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.3,
            )
        )
        curve_labels.append(f"{tau_text}")

    # ax.set_title(title)
    ax.set_xlabel("m (sequence depth)")
    ax.set_ylabel("Mean survival probability")
    ax.grid(True, linestyle="--", alpha=0.28)

    # 主 legend：各条 tau_c
    leg1 = ax.legend(
        curve_handles,
        curve_labels,
        loc="upper right",
        frameon=True,
        fontsize=10,
        ncol=2,
    )
    ax.add_artist(leg1)

    # 辅助 legend：说明点和线各自表示什么
    style_handles = [
        Line2D(
            [0], [0],
            color="black",
            marker="o",
            linestyle="none",
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Observed mean ± SEM",
        ),
        Line2D(
            [0], [0],
            color="black",
            lw=2.0,
            label="Single-exp fit",
        ),
    ]
    ax.legend(
        handles=style_handles,
        loc="lower left",
        frameon=True,
        fontsize=9,
    )

    out_png = run_dir / f"{outfile_stem}.png"
    out_pdf = run_dir / f"{outfile_stem}.pdf"
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf)}


if __name__ == "__main__":
    run_dir = "results/experiment2/20260504-xxxxxx"
    paths = exp2_postproc_fig3_rb_multicurve(run_dir)
    print(paths)
