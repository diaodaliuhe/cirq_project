# Paper 2026 Reproduction Scripts

This directory contains the implementation scripts used to reproduce the paper figures from the current implementation.

The older scripts in `experiments.scripts` and the notebooks under `mynotebooks0416/` are retained as development history. They are not required for reproducing the final paper figures.

## Entry Points

- Root-level launchers:
  - `run_paper_fig2_cpmg.py`
  - `run_paper_fig3_rb_tau_sweep.py`
  - `run_paper_fig4_fig5_rb_decay.py`
- `run_fig2_cpmg.py`: CPMG state-fidelity sweep for Figure 2.
- `run_fig3_rb_tau_sweep.py`: fixed-depth RB survival sweep for Figure 3.
- `run_fig4_fig5_rb_decay.py`: RB decay curves and fit statistics for Figures 4 and 5.

Each script uses the corresponding YAML files in `experiments/src/experiments/configs/paper_2026/` by default and writes to `results/paper_2026/`.

## Example Commands

```bash
python run_paper_fig2_cpmg.py --n-workers 40
python run_paper_fig3_rb_tau_sweep.py --n-workers 40
python run_paper_fig4_fig5_rb_decay.py --n-workers 40
```

For a configuration-only check:

```bash
python run_paper_fig2_cpmg.py --dry-run
python run_paper_fig3_rb_tau_sweep.py --dry-run
python run_paper_fig4_fig5_rb_decay.py --dry-run
```
