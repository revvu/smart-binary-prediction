# AGENTS.md

Repository-level operating guide for coding agents.

## Scope

This repository combines:

- experiment code for generating results/figures
- manuscript files for paper context

Primary goal: make it easy to add and iterate on new experiments, then promote selected outputs into paper assets.

## Canonical structure

- `experiments/`
  - `exp01_best_of_all_worlds/`
  - `exp02_online_linear_classification/`
  - `exp03_time_varying_mu_oco/`
  - future experiments: `expNN_<short_name>/`
- `paper/`
  - `manuscripts/`
  - `shared/`
  - `figures/`
- `archive/`

## Guardrails

- Treat `paper/` as context-only unless the user explicitly requests paper edits.
- Default to editing only inside `experiments/` for analysis, code, and figure generation.
- Keep experiment changes self-contained; do not couple experiments via ad hoc relative imports.
- Do not delete historical outputs unless asked.

## New experiment conventions

For each new experiment under `experiments/expNN_<short_name>/`:

- include a local `README.md` with:
  - experiment question
  - dependency setup
  - exact run commands
  - output paths
- place generated artifacts in a local `outputs/` folder (or clearly named equivalent)
- use deterministic seeds where possible and document them
- include a single entry-point command/script for reproduction

## Figure promotion workflow

1. Generate candidate figures under the owning experiment folder.
2. Select final figure(s) for manuscript use.
3. Copy/move final figure(s) into `paper/figures/` with stable filenames.
4. Keep old exploratory plots in experiment folders.

## Safety

- Prefer non-destructive reorganizations.
- If a move could break imports or run scripts, update paths in the same change.
- If uncertain whether a file is manuscript-critical, leave it in `paper/` and ask before editing.
