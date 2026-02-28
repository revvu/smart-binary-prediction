# Experiments

Each experiment should be isolated in its own folder:

- `expNN_<short_name>/`

Recommended per-experiment structure:

- `src/` or top-level scripts for experiment logic
- `outputs/` for generated figures/tables
- `README.md` with:
  - purpose
  - dependencies
  - run commands
  - output file map

## Existing experiments

- `exp01_best_of_all_worlds/`: SMART / AdaHedge / binary prediction experiments and legacy graph outputs.
- `exp02_online_linear_classification/`: OCO comparison experiments (FTRL, FTL, SMART variants).
- `exp03_time_varying_mu_oco/`: SMART for OCO quadratic losses with time-varying $\mu_t$ and figure generation scripts.

## Adding a new experiment

1. Create `experiments/expNN_<short_name>/`.
2. Add a minimal `README.md` with exact run commands.
3. Keep generated artifacts local to that experiment first.
4. Move only final paper-ready figures into `paper/figures/`.
