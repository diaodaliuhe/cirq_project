# Execution-Aware Segmented Flux-Noise Simulation

This repository contains the code and reproduction artifacts for the manuscript

**Execution-Aware Segmented Modeling of Temporally Correlated Flux-Induced Phase Noise in Quantum Circuits**.

The project extends a local fork of [Cirq](https://github.com/quantumlib/Cirq) with circuit-level noise-modeling utilities for studying temporally correlated, flux-induced phase noise in superconducting quantum-circuit simulations. The main use case is to evaluate how a tunable segment duration `tau_c` affects circuit-level observables under matched execution settings.

The model implemented here is an execution-aware, circuit-level, phenomenological segmented phase-noise model. It is not intended to be a full device-level reconstruction of a calibrated flux-noise power spectral density.

## Repository Layout

```text
.
├── dev/Cirq/                         # Cirq fork used by this project
│   └── cirq-core/cirq/noise/          # Custom noise models and timing utilities
├── experiments/                       # Local experiment package
│   └── src/experiments/
│       ├── configs/paper_2026/        # Final paper-figure configurations
│       └── scripts/paper_2026/        # Final paper reproduction workflows
├── results/paper_2026/                # One tracked set of paper-figure outputs
├── run_paper_fig2_cpmg.py             # Root launcher for Figure 2
├── run_paper_fig3_rb_tau_sweep.py     # Root launcher for Figure 3
├── run_paper_fig4_fig5_rb_decay.py    # Root launcher for Figures 4 and 5
└── REPRODUCIBILITY.md                 # Compact reproduction notes
```

Historical notebooks and exploratory scripts are retained for development context, but the final paper figures are reproduced through the root-level launchers and the `paper_2026` configuration set.

## Main Components

The Cirq fork adds a `cirq.noise` module focused on execution-aware noisy-circuit construction.

Key modules include:

- `SegmentedFluxNoiseModel`: applies sampled flux-induced phase offsets according to the segment assigned to each eligible duration-bearing operation.
- `assign_timed_circuit_context`: constructs a timing map from a compiled circuit, including operation start times, durations, and segment IDs.
- `CompositeNoiseModel`: combines operation-level and moment-level noise models.
- `PhotonDecayNoiseModel`: models photon-induced dephasing over circuit execution time.
- `IdleNoiseModel` and `RyGateNoiseModel`: background circuit-level noise components.
- `metrics.py`: state-fidelity and trace-distance utilities used by the experiments; the paper workflows also export both reported-score and state-fidelity summaries for Figure 2.

The segmented flux-noise construction samples phase offsets per segment and qubit. In the final paper workflows, waits and zero-duration virtual `Z` operations are excluded from flux-noise insertion, while eligible duration-bearing physical rotations retain the phase-noise rule.

## Experiments

The included workflows reproduce the paper figures:

- **Figure 2**: CPMG state-fidelity sweep over `tau_c` for three flux-noise strengths.
- **Figure 3**: fixed-depth randomized-benchmarking survival probability versus `tau_c`.
- **Figures 4 and 5**: randomized-benchmarking decay curves and fit statistics.

The final YAML configurations are under:

```text
experiments/src/experiments/configs/paper_2026/
```

The corresponding workflow implementations are under:

```text
experiments/src/experiments/scripts/paper_2026/
```

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules --branch paper-2026-release https://github.com/diaodaliuhe/cirq_project.git
cd cirq_project
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the local Cirq fork and the experiment package in editable mode:

```bash
pip install -e dev/Cirq
pip install -e experiments
```

The project was developed with Python 3.11. The root launchers set BLAS/OpenMP thread limits before importing NumPy, SciPy, Cirq, or Matplotlib, which helps avoid oversubscription in multiprocessing runs.

## Reproducing the Paper Figures

From the repository root:

```bash
python run_paper_fig2_cpmg.py --n-workers 40
python run_paper_fig3_rb_tau_sweep.py --n-workers 40
python run_paper_fig4_fig5_rb_decay.py --n-workers 40
```

For a quick configuration-only check:

```bash
python run_paper_fig2_cpmg.py --dry-run
python run_paper_fig3_rb_tau_sweep.py --dry-run
python run_paper_fig4_fig5_rb_decay.py --dry-run
```

If the machine has less available memory, reduce the worker count:

```bash
python run_paper_fig2_cpmg.py --n-workers 20
```

Each run writes a timestamped directory under `results/paper_2026/`.

## Included Outputs

This repository tracks one complete set of outputs corresponding to the paper figures:

```text
results/paper_2026/paper_fig2_cpmg/cpmg_fig2_20260721-044956
results/paper_2026/paper_fig3_rb_tau_sweep/rb_tau_20260721-045718
results/paper_2026/paper_fig4_fig5_rb_decay/rb_decay_20260721-053908
```

These directories include saved configs, provenance, raw arrays where applicable, postprocessed tables, summaries, and generated figures. Older exploratory results are intentionally ignored.

## Scope and Limitations

This code is designed for circuit-level studies of temporally correlated flux-induced phase noise. The segmented model controls how sampled phase offsets are shared across the execution timeline through `tau_c`; it does not claim to reproduce a unique calibrated device noise spectrum across different `tau_c` values.

The paper experiments use single-qubit CPMG and randomized-benchmarking circuits as controlled testbeds. Multiqubit schedules, calibrated device-specific spectra, syndrome-extraction circuits, and logical-level error-correction studies are natural extensions but are not part of the final reproduction workflows in this repository.

## Citation

If you use this repository, please cite the associated manuscript once the final bibliographic information is available:

```text
Hongxiang Zhu, Xinxuan Chen, Hui-Hai Zhao, Feng Wu, and Zhaofeng Su,
"Execution-Aware Segmented Modeling of Temporally Correlated Flux-Induced Phase Noise in Quantum Circuits", 2026.
```
