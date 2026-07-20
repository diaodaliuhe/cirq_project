import os
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cirq-project")

ROOT = Path(__file__).resolve().parent
for path in (
    ROOT / "experiments" / "src",
    ROOT / "dev" / "Cirq" / "cirq-core",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.scripts.paper_2026.run_fig2_cpmg import main


if __name__ == "__main__":
    main()
