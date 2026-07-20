# Reproducibility

This repository contains a Cirq fork under `dev/Cirq/` and a local experiment package under `experiments/`.

The final paper reproduction entry points are the root-level launchers:

```text
run_paper_fig2_cpmg.py
run_paper_fig3_rb_tau_sweep.py
run_paper_fig4_fig5_rb_decay.py
```

The implementation scripts are under:

```text
experiments/src/experiments/scripts/paper_2026/
```

The corresponding configuration files are under:

```text
experiments/src/experiments/configs/paper_2026/
```

Historical notebooks and exploratory scripts are retained for development context, but they are not required to reproduce the final paper figures.

## Environment

From the repository root:

```bash
pip install -e dev/Cirq
pip install -e experiments
```

## Figure 2

```bash
python run_paper_fig2_cpmg.py --n-workers 40
```

Default output:

```text
results/paper_2026/paper_fig2_cpmg/
```

## Figure 3

```bash
python run_paper_fig3_rb_tau_sweep.py --n-workers 40
```

Default output:

```text
results/paper_2026/paper_fig3_rb_tau_sweep/
```

## Figures 4 and 5

```bash
python run_paper_fig4_fig5_rb_decay.py --n-workers 40
```

Default output:

```text
results/paper_2026/paper_fig4_fig5_rb_decay/
```

## Included Outputs

One complete set of paper-figure outputs is tracked under:

```text
results/paper_2026/
```

These outputs include the saved configs, provenance, raw arrays, postprocessed tables, summaries, and generated figures for Figures 2--5. Other historical or exploratory result directories remain ignored.

## Notes

- The default sample counts and tau grids are defined in the YAML files under `experiments/src/experiments/configs/paper_2026/`.
- The root launchers set BLAS/OpenMP thread limits before importing the experiment package.
- The scripts write configs, provenance, summaries, and figures into timestamped output directories.
- Large historical result directories are not required for reproduction.
