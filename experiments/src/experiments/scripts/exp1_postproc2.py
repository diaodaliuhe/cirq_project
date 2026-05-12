from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

# ----------------------------
# reuse: seed / resampling
# ----------------------------

def resolve_seed(seed: Optional[int]) -> int:
    import secrets
    return secrets.randbits(32) if seed is None else int(seed)


def _choose_indices(N: int, n: int, m: int, rng: np.random.Generator, method: str) -> np.ndarray:
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
    x = arr[idx]  # (n, m)

    batch_mean = x.mean(axis=1)
    batch_std = x.std(axis=1, ddof=1) if m >= 2 else np.zeros(n, dtype=float)

    mean_of_means = float(batch_mean.mean())
    std_of_means = float(batch_mean.std(ddof=1)) if n >= 2 else 0.0
    sem = float(std_of_means / np.sqrt(n)) if n >= 1 else float("nan")

    mean_batch_std = float(batch_std.mean())
    std_batch_std = float(batch_std.std(ddof=1)) if n >= 2 else 0.0

    return {
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "sem_of_means": sem,
        "mean_batch_std": mean_batch_std,
        "std_batch_std": std_batch_std,
    }


# ----------------------------
# io helpers
# ----------------------------

def _next_postproc_dir(run_root: Path, *, subdir: str = "postprocs", prefix: str = "postproc",
                      tag: Optional[str] = None) -> Path:
    base = run_root / subdir
    base.mkdir(parents=True, exist_ok=True)
    if tag is not None:
        out = base / f"{prefix}_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    max_k = 0
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            s = p.name[len(prefix):]
            if s.isdigit():
                max_k = max(max_k, int(s))
    out = base / f"{prefix}{max_k + 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _find_run_dirs(exp_root: Path) -> List[Path]:
    """
    Find run dirs like:
      exp_root/<timestamp>/
        raw/raw_manifest.json
    """
    out = []
    for p in exp_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "raw" / "raw_manifest.json").exists():
            out.append(p)
    out = sorted(out)
    if not out:
        raise FileNotFoundError(f"No run dirs with raw/raw_manifest.json under: {exp_root}")
    return out


def _load_manifest(raw_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = raw_dir / "raw_manifest.json"
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = obj.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError(f"Invalid or empty manifest: {manifest_path}")
    return items


def _resolve_file(item: Dict[str, Any], metric: str) -> str:
    """
    metric: 'fidelity1' or 'trace_distance'
    """
    if metric == "fidelity1":
        for k in ("fidelity1_file", "fid_file", "fid", "fidelity_file"):
            if k in item and item[k]:
                return str(item[k])
    elif metric == "trace_distance":
        for k in ("trace_distance_file", "td_file", "td", "trace_file"):
            if k in item and item[k]:
                return str(item[k])
    raise KeyError(f"Cannot resolve file for metric={metric!r} in item: {item}")


def _get_n_total(item: Dict[str, Any]) -> int:
    shape = item.get("shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) >= 1:
        return int(shape[0])
    if "n_samples" in item:
        return int(item["n_samples"])
    raise KeyError(f"Cannot infer N_total from item: {item}")


# ----------------------------
# extract wait_ns (run-level)
# ----------------------------

def extract_wait_ns(run_dir: Path) -> float:
    """
    Priority:
      1) run_dir/config.yaml : circuit.wait_duration   (recommended, you already have it)
      2) run_dir/run_meta.json with wait_ns            (optional)
      3) run_dir/environment.json with wait_ns         (optional)
      4) run_dir/raw/average_flux.json with wait_ns    (optional)
    """
    # ---- 1) config.yaml (BEST) ----
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8", errors="ignore")

        # 1a) try PyYAML if available
        try:
            import yaml  # type: ignore
            obj = yaml.safe_load(text)
            v = obj.get("circuit", {}).get("wait_duration", None)
            if v is not None:
                return float(v)
        except Exception:
            pass

        # 1b) fallback: manual parse "circuit:" block
        lines = text.splitlines()
        in_circuit = False
        base_indent: Optional[int] = None

        for ln in lines:
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue

            stripped = ln.lstrip()
            indent = len(ln) - len(stripped)

            # enter circuit block
            if re.match(r"^\s*circuit\s*:\s*$", ln):
                in_circuit = True
                base_indent = indent
                continue

            if in_circuit:
                # leave circuit block when indentation decreases back to base level or less
                if base_indent is not None and indent <= base_indent and not re.match(r"^\s+", ln):
                    in_circuit = False
                    base_indent = None
                    continue

                # match wait_duration: 60
                m = re.match(r"^\s*wait_duration\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", ln)
                if m:
                    return float(m.group(1))

        raise ValueError(f"config.yaml exists but cannot find circuit.wait_duration in: {cfg}")

    # ---- 2) run_meta.json (optional fallback) ----
    meta = run_dir / "run_meta.json"
    if meta.exists():
        obj = json.loads(meta.read_text(encoding="utf-8"))
        if "wait_ns" in obj:
            return float(obj["wait_ns"])

    # ---- 3) environment.json (optional fallback) ----
    env = run_dir / "environment.json"
    if env.exists():
        obj = json.loads(env.read_text(encoding="utf-8"))
        if "wait_ns" in obj:
            return float(obj["wait_ns"])

    # ---- 4) average_flux.json (optional fallback) ----
    af = run_dir / "raw" / "average_flux.json"
    if af.exists():
        obj = json.loads(af.read_text(encoding="utf-8"))
        for k in ("wait_ns", "wait_duration_ns", "wait_duration"):
            if k in obj:
                return float(obj[k])

    raise FileNotFoundError(
        f"Cannot extract wait_duration for run_dir={run_dir}. "
        f"Expected config.yaml with circuit.wait_duration."
    )

# ----------------------------
# select dirs
# ----------------------------
def _resolve_run_dirs(
    *,
    # 方式 A：显式传入 run_dirs（推荐）
    run_dirs: Optional[Sequence[str | Path]] = None,

    # 方式 B：从文件读取 run_dirs（每行一个路径）
    run_list_file: Optional[str | Path] = None,

    # 方式 C：从若干根目录扫描（可配合 include/exclude）
    exp_roots: Optional[Sequence[str | Path]] = None,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,

    # 可选：按 config.yaml 的 circuit.type 过滤（比如只要 CPMG）
    require_circuit_type: Optional[str] = None,
) -> List[Path]:
    """
    Return a list of run_dir Paths that contain:
      - run_dir/config.yaml
      - run_dir/raw/raw_manifest.json

    Selection priority:
      1) run_dirs
      2) run_list_file
      3) exp_roots scan

    Filters:
      - include_regex/exclude_regex match on run_dir path string
      - require_circuit_type matches config.yaml: circuit.type
    """
    picked: List[Path] = []

    def _is_valid_run_dir(p: Path) -> bool:
        return (p / "raw" / "raw_manifest.json").exists() and (p / "config.yaml").exists()

    # ---- choose candidates ----
    if run_dirs is not None:
        picked = [Path(x) for x in run_dirs]
    elif run_list_file is not None:
        lines = Path(run_list_file).read_text(encoding="utf-8").splitlines()
        lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        picked = [Path(ln) for ln in lines]
    else:
        if exp_roots is None:
            raise ValueError("You must provide one of: run_dirs / run_list_file / exp_roots")
        roots = [Path(x) for x in exp_roots]
        for root in roots:
            if not root.exists():
                continue
            for p in root.iterdir():
                if p.is_dir() and _is_valid_run_dir(p):
                    picked.append(p)

    # normalize + filter existence/validity
    picked = [p.resolve() for p in picked if _is_valid_run_dir(p)]

    # ---- include/exclude regex on path string ----
    if include_regex:
        inc = re.compile(include_regex)
        picked = [p for p in picked if inc.search(str(p))]

    if exclude_regex:
        exc = re.compile(exclude_regex)
        picked = [p for p in picked if not exc.search(str(p))]

    # ---- circuit.type filter ----
    if require_circuit_type is not None:
        # lightweight parse: regex in config.yaml to avoid mandatory PyYAML
        typ_pat = re.compile(r"^\s*type\s*:\s*([A-Za-z0-9_\-]+)\s*$", re.MULTILINE)
        filtered: List[Path] = []
        for p in picked:
            txt = (p / "config.yaml").read_text(encoding="utf-8", errors="ignore")
            # try to only match inside circuit block (good enough for your config format)
            # simple approach: if config contains "circuit:" and later a "type: X"
            if "circuit:" not in txt:
                continue
            m = typ_pat.search(txt)
            if m and m.group(1) == require_circuit_type:
                filtered.append(p)
        picked = filtered

    picked = sorted(set(picked))
    if not picked:
        raise FileNotFoundError("No valid run_dir found after applying selection/filters.")
    return picked

# ----------------------------
# plotting (Plan A)
# ----------------------------

def _log_edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if np.any(vals <= 0):
        raise ValueError("log axis requires all values > 0")
    if len(vals) == 1:
        v = vals[0]
        return np.array([v / 2, v * 2], dtype=float)

    mids = np.sqrt(vals[:-1] * vals[1:])
    edges = np.empty(len(vals) + 1, dtype=float)
    edges[1:-1] = mids
    r0 = vals[1] / vals[0]
    r1 = vals[-1] / vals[-2]
    edges[0] = vals[0] / np.sqrt(r0)
    edges[-1] = vals[-1] * np.sqrt(r1)
    return edges


def plot_planA_scatter(fig_path: Path, rows: List[Dict[str, Any]], *, title: str) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    wait = np.array([float(r["wait_ns"]) for r in rows], dtype=float)
    tau  = np.array([float(r["tau_c"]) for r in rows], dtype=float)
    Fhat = np.array([float(r["mean_of_means"]) for r in rows], dtype=float)

    eps = 1e-15
    Z = np.log10(np.clip(1.0 - Fhat, eps, None))

    vmin, vmax = np.percentile(Z, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.min(Z)), float(np.max(Z))

    cmap = plt.cm.viridis.copy()

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.set_xscale("log")

    # 新增横轴数据点
    ax.set_xticks(wait)  # wait 是你采样过的 wait_duration 列表/数组
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    ax.minorticks_off()  # 可选：关掉次刻度，避免太乱
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_horizontalalignment("right")

    ax.set_yscale("log")
    ax.set_xlabel("wait_duration (ns)")
    ax.set_ylabel("tau_c (ns)")
    ax.set_title(title)

    sc = ax.scatter(wait, tau, c=Z, cmap=cmap, vmin=vmin, vmax=vmax,
                    marker="s", s=220, edgecolors="none")

    lo = max(np.min(wait), np.min(tau))
    hi = min(np.max(wait), np.max(tau))
    if lo < hi:
        line = np.logspace(np.log10(lo), np.log10(hi), 300)
        ax.plot(line, line, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}(1 - F_{\hat{}})$")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()


# ----------------------------
# outputs
# ----------------------------

def write_grid_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "wait_ns", "tau_c", "N_total", "n", "m", "method",
        "mean_of_means", "std_of_means", "sem_of_means",
        "mean_batch_std", "std_batch_std",
        "run_dir", "raw_file",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join([
                f"{r['wait_ns']}",
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
                str(r["run_dir"]),
                str(r["raw_file"]),
            ]) + "\n")


def write_report(path: Path, rows: List[Dict[str, Any]], *, exp_root: Path, seed: int, n: int, m: int,
                 method: str, metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.array([float(r["mean_of_means"]) for r in rows], dtype=float)
    best_i = int(np.nanargmax(y))
    worst_i = int(np.nanargmin(y))
    b, w = rows[best_i], rows[worst_i]

    lines = []
    lines.append("CPMG 2D postproc report (wait_ns × tau_c)")
    lines.append(f"exp_root: {exp_root}")
    lines.append(f"metric: {metric}")
    lines.append(f"seed_used: {seed}")
    lines.append(f"n×m: {n}×{m}   method={method}")
    lines.append("")
    lines.append(f"#grid points: {len(rows)}")
    lines.append(f"unique wait: {len(set(float(r['wait_ns']) for r in rows))}")
    lines.append(f"unique tau_c: {len(set(float(r['tau_c']) for r in rows))}")
    lines.append("")
    lines.append("Best point (max F_hat)")
    lines.append(f"  wait={b['wait_ns']} ns, tau_c={b['tau_c']} ns, F_hat={b['mean_of_means']:.10f}, std={b['std_of_means']:.10f}")
    lines.append("Worst point (min F_hat)")
    lines.append(f"  wait={w['wait_ns']} ns, tau_c={w['tau_c']} ns, F_hat={w['mean_of_means']:.10f}, std={w['std_of_means']:.10f}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------
# ONE public entry
# ----------------------------

def postprocess_cpmg_2d(
    exp_root: str | Path,
    *,
    # --- NEW: selection controls (optional) ---
    run_dirs: Optional[Sequence[str | Path]] = None,
    run_list_file: Optional[str | Path] = None,
    exp_roots: Optional[Sequence[str | Path]] = None,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    require_circuit_type: Optional[str] = "CPMG",   # default protect you from mixing circuits

    # --- existing knobs ---
    n: int,
    m: int,
    metric: str = "fidelity1",          # or "trace_distance"
    seed: Optional[int] = None,
    method: str = "partition",
    postprocs_dirname: str = "postprocs",
    postproc_prefix: str = "postproc",
    postproc_tag: Optional[str] = None,
    out_prefix: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Path]:
    """
    2D postproc: wait_duration (from config.yaml) × tau_c (from raw_manifest)

    Directory selection (priority):
      1) run_dirs
      2) run_list_file
      3) exp_roots (if None, fallback to [exp_root])

    exp_root is kept for backward compatibility and as default output root.
    """
    exp_root = Path(exp_root)

    # ---- pick run dirs (NEW) ----
    if exp_roots is None:
        # If user didn't pass exp_roots explicitly, treat exp_root as the default scan root.
        exp_roots = [exp_root]

    run_dirs_resolved = _resolve_run_dirs(
        run_dirs=run_dirs,
        run_list_file=run_list_file,
        exp_roots=exp_roots,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        require_circuit_type=None,  # we'll do a safer type-filter below
    )

    # ---- safer circuit.type filter (prefer YAML, fallback to regex) ----
    if require_circuit_type is not None:
        filtered: List[Path] = []
        for rd in run_dirs_resolved:
            cfg = rd / "config.yaml"
            ok = False
            if cfg.exists():
                txt = cfg.read_text(encoding="utf-8", errors="ignore")
                # try PyYAML first (most reliable)
                try:
                    import yaml  # type: ignore
                    obj = yaml.safe_load(txt) or {}
                    ctype = (obj.get("circuit", {}) or {}).get("type", None)
                    ok = (str(ctype) == str(require_circuit_type))
                except Exception:
                    # fallback: search within circuit block only
                    in_circuit = False
                    base_indent: Optional[int] = None
                    for ln in txt.splitlines():
                        if not ln.strip() or ln.lstrip().startswith("#"):
                            continue
                        stripped = ln.lstrip()
                        indent = len(ln) - len(stripped)
                        if re.match(r"^\s*circuit\s*:\s*$", ln):
                            in_circuit = True
                            base_indent = indent
                            continue
                        if in_circuit:
                            # leave circuit block when indentation drops back
                            if base_indent is not None and indent <= base_indent and not re.match(r"^\s+", ln):
                                in_circuit = False
                                base_indent = None
                                continue
                            mm = re.match(r"^\s*type\s*:\s*([A-Za-z0-9_\-]+)\s*$", ln)
                            if mm:
                                ok = (mm.group(1) == require_circuit_type)
                                break
            if ok:
                filtered.append(rd)
        run_dirs_resolved = sorted(filtered)

    if not run_dirs_resolved:
        raise FileNotFoundError("No run_dir selected after applying filters. "
                                "Check run_dirs/run_list_file/exp_roots and require_circuit_type.")

    # ---- RNG ----
    seed_used = resolve_seed(seed)
    rng = np.random.default_rng(seed_used)
    print(f"[cpmg2d] seed_used={seed_used}")
    print(f"[cpmg2d] selected {len(run_dirs_resolved)} run dirs")

    # ---- output dir: write under exp_root (keep old behavior) ----
    out_dir = _next_postproc_dir(exp_root, subdir=postprocs_dirname, prefix=postproc_prefix, tag=postproc_tag)

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[float, float]] = set()

    for run_dir in run_dirs_resolved:
        wait_ns = float(extract_wait_ns(run_dir))
        raw_dir = run_dir / "raw"
        items = _load_manifest(raw_dir)

        for item in items:
            tau_c = float(item["tau_c"])
            key = (wait_ns, tau_c)
            if key in seen:
                raise ValueError(
                    f"Duplicate grid point (wait_ns,tau_c)={key} encountered.\n"
                    f"Current run_dir={run_dir}\n"
                    f"If you want pooling across runs, implement pooling (concat/pool raw samples) instead."
                )
            seen.add(key)

            N_total = _get_n_total(item)
            filename = _resolve_file(item, metric=metric)
            fpath = raw_dir / filename
            if not fpath.exists():
                raise FileNotFoundError(f"Missing raw file: {fpath}")

            mm = np.memmap(fpath, mode="r", dtype=np.float64, shape=(N_total,))
            stats = _summarize_nm(mm, int(n), int(m), rng, method=str(method))

            rows.append({
                "wait_ns": wait_ns,
                "tau_c": tau_c,
                "N_total": int(N_total),
                "n": int(n),
                "m": int(m),
                "method": str(method),
                "run_dir": run_dir,
                "raw_file": filename,
                **stats,
            })

    rows = sorted(rows, key=lambda r: (float(r["tau_c"]), float(r["wait_ns"])))

    if out_prefix is None:
        out_prefix = f"{metric}_nm_{int(n)}x{int(m)}"

    tsv_path = out_dir / f"{out_prefix}_grid.tsv"
    report_path = out_dir / f"{out_prefix}_report.txt"
    fig_path = out_dir / f"{out_prefix}_planA_heatmap.png"

    write_grid_tsv(tsv_path, rows)
    write_report(report_path, rows, exp_root=exp_root, seed=seed_used, n=n, m=m, method=method, metric=metric)

    if title is None:
        title = f"Plan A: log10(1 - F_hat), {metric}, {int(n)}×{int(m)}"

    if metric != "fidelity1":
        raise ValueError(
            "Heatmap currently implemented for fidelity1 (log10(1-F_hat)). "
            "If you want trace_distance heatmap, choose a transform (e.g., td or log10(td+eps))."
        )

    plot_planA_scatter(fig_path, rows, title=title)

    return {"postproc_dir": out_dir, "tsv": tsv_path, "report": report_path, "figure": fig_path}
