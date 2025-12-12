# experiments/utils.py
from importlib.resources import files
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml

def find_project_root_by_dir(start: str | Path = ".", anchor_dir: str = "mynotebooks0416") -> Path:
    """
    从 start 开始向上找，遇到目录名=anchor_dir 就把它的父目录当作项目根。
    """
    cur = Path(start).resolve()
    while True:
        if (cur / anchor_dir).is_dir():
            return cur   # 找到包含 anchor_dir 的那一层
        if cur.parent == cur:
            raise RuntimeError(f"未找到包含 {anchor_dir} 的目录")
        cur = cur.parent

def make_run_dir(exp_name: str = "exp1", tag: str | None = None) -> Path:
    """
    在 {project_root}/results/{exp_name}/ 下创建唯一的运行目录：
    例如 results/exp1/20250901-143522-q6l4-t4-s10
    若时间戳重复（极少见），自动在末尾加 -2, -3...
    """
    project_root = find_project_root_by_dir(".")
    base = project_root / "results" / exp_name
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = ts if tag is None else f"{ts}-{tag}"

    run_dir = base / stem
    if run_dir.exists():
        i = 2
        while (base / f"{stem}-{i}").exists():
            i += 1
        run_dir = base / f"{stem}-{i}"

    run_dir.mkdir()
    return run_dir

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """default 在下，override 在上；返回新 dict（不改入参）。"""
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(
    name_or_path: str | Path,
    default_path: str | Path | None = None,
    search_package: bool = True,
) -> Dict[str, Any]:
    """
    智能读取配置：
    - 如果传入的是可存在的**文件路径**（绝对/相对），就读这个文件；
    - 否则（无路径分隔符或找不到），且 search_package=True，则从已安装包
      `experiments/configs/<name>` 里读；
    - 可选：提供 default_path（或 default dict），与用户配置做 **用户优先** 的递归合并。

    返回: dict，并在 _debug 里记录实际读取的路径。
    """
    # 1) 解析用户配置
    user_cfg: Dict[str, Any] = {}
    user_path: Optional[Path] = None

    p = Path(str(name_or_path)).expanduser()
    if any(sep in str(p) for sep in ("/", "\\")) or p.exists():
        # 看起来像路径 → 尝试按文件读
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        user_path = p
        user_cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    elif search_package:
        # 当作包内资源名称（例如 "exp1_fid.yaml"）
        res = files("experiments").joinpath("configs", str(name_or_path))
        if not res.is_file():
            raise FileNotFoundError(
                f"Config '{name_or_path}' not found in package resources "
                f"(experiments/configs)."
            )
        user_path = Path(str(res))
        user_cfg = yaml.safe_load(res.read_text(encoding="utf-8")) or {}
    else:
        raise FileNotFoundError(f"Config not found: {name_or_path}")

    # 2) 读取默认（可选）
    default_cfg: Dict[str, Any] = {}
    default_src: Optional[str] = None
    if default_path:
        dp = Path(str(default_path)).expanduser().resolve()
        if not dp.exists():
            raise FileNotFoundError(f"Default config not found: {dp}")
        default_src = str(dp)
        default_cfg = yaml.safe_load(dp.read_text(encoding="utf-8")) or {}

    # 3) 合并（✅ 用户覆盖默认）
    cfg = _deep_merge(default_cfg, user_cfg)
    cfg.setdefault("_debug", {})
    cfg["_debug"].update({
        "loaded_path": str(user_path) if user_path else None,
        "default_path": default_src,
        "from_package": user_path is not None and "site-packages" in str(user_path),
    })
    return cfg
    
