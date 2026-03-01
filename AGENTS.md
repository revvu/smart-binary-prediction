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
  - `exp04_leader_path_synthesis/`
  - future experiments: `expNN_<short_name>/`
- `paper/`
  - `manuscripts/`
  - `shared/`
  - `figures/`
- `archive/`

## Required context files

Before proposing or implementing SMART-related changes, read:

- `smart_algorithm.md` (source of truth for SMART properties, calibration caveats, and experiment priorities)
- `experiments/<target>/README.md` (target experiment goals, design, run path, and known issues)

If a request touches SMART theory/behavior and `smart_algorithm.md` is not referenced in your reasoning, stop and read it first.

## Guardrails

- Treat `paper/` as context-only unless the user explicitly requests paper edits.
- Default to editing only inside `experiments/` for analysis, code, and figure generation.
- Keep experiment changes self-contained; do not couple experiments via ad hoc relative imports.
- Do not delete historical outputs unless asked.

## New experiment conventions

For each new experiment under `experiments/expNN_<short_name>/`:

- include a local `README.md` with:
  - experiment question
  - experiment design and known limitations
  - dependency setup
  - exact run commands
  - output paths
- place generated artifacts in a local `outputs/` folder (or clearly named equivalent)
- use deterministic seeds where possible and document them
- include a single entry-point command/script for reproduction
- add a curated `figures/` folder with:
  - only paper-candidate plots
  - descriptive filenames
  - `INDEX.md` containing label/title/source mapping

## Figure promotion workflow

1. Generate candidate figures under the owning experiment folder.
2. Curate final paper-candidate figures into `<experiment>/figures/` with `INDEX.md`.
3. Ensure `README.md` figure references match `figures/INDEX.md`.
4. Only then copy/move selected figure(s) into `paper/figures/` with stable filenames.
5. Keep exploratory/diagnostic plots in experiment-local output folders.

## SMART workflow defaults

When iterating on SMART experiments, prioritize this sequence:

1. Define regime families (benign, drift, abrupt shift, corruption bursts).
2. Run baselines (`FTL`, robust policy) and SMART variants (theoretical and empirical thresholds).
3. Log switch diagnostics (`Sigma_t`, threshold, switch round, pre/post switch regret).
4. Sweep threshold scaling to quantify calibration sensitivity.
5. Promote only the clearest figures to `<experiment>/figures/`.

## Progress optimization checklist

Use this checklist for each experiment update:

1. Reproducibility:
- fixed seeds
- one-command run path
2. Comparability:
- same horizons/replicates across compared algorithms
- explicit comparator definition in docs
3. Interpretability:
- at least one diagnostic figure explaining *why* SMART behaved as observed
4. Documentation sync:
- update `README.md` and `figures/INDEX.md` together

## Safety

- Prefer non-destructive reorganizations.
- If a move could break imports or run scripts, update paths in the same change.
- If uncertain whether a file is manuscript-critical, leave it in `paper/` and ask before editing.
