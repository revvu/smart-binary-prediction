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

## Required experimental design gate

Before implementing or revising any experiment, explicitly document:

1. the SMART behavior claim the experiment must show (e.g., preserve optimism, protect in hard regimes, interpretable switch),
2. the exact sequence families used to show that claim,
3. why each sequence is realistic/plausible for the domain,
4. acceptance criteria for success (what plot behavior would count as evidence).

Do not proceed to code/plot generation until this sequence-design gate is written in the experiment `README.md`.

## Required worst-case baseline specification

Each experiment README must include a dedicated section that states:

1. which algorithm is used as the worst-case/robust baseline,
2. where that specification comes from (theory, standard reduction, or implementation derivation),
3. why that specific formulation is appropriate for the experiment's loss/domain.

If using FTRL, do not write only "FTRL". You must specify the concrete objective (regularizer, constraints, update form) and explain why it is the correct FTRL instantiation for the setting.

## Primary figure contract (paper default)

Unless the user explicitly asks otherwise, experiment figures should use:

1. x-axis: horizon (`n`),
2. y-axis: regret at that horizon,
3. for each horizon, evaluate on fresh sequence(s) of exactly that length,
4. plot mean and uncertainty bands across replications when stochastic.

Do not treat single-sequence prefix regret traces (`regret vs t` on one fixed-length run) as the primary result figure. Prefix traces can be included only as supplementary diagnostics.

## Publication figure standards

For figures intended for manuscript inclusion:

1. use descriptive, professional chart titles (specific but concise),
2. use clean axis labels (e.g., `Horizon`, `Regret`) without verbose sentences,
3. use descriptive output filenames that encode experiment and figure purpose,
4. avoid vague labels like `plot1`, `result`, `curve`, or ambiguous abbreviations in final figure names.

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

1. Define the paper claim first (what SMART property is being demonstrated).
2. Design sequence families to elicit that property (benign, hard, representative mixed regime).
3. Build horizon grids and run baselines (`FTL`, robust policy) and SMART variants over fresh sequences at each horizon.
4. Produce primary `regret vs horizon` plots first.
5. Log switch diagnostics (`Sigma_t`, threshold, switch round, pre/post switch regret) as supplementary.
6. Sweep threshold scaling to quantify calibration sensitivity.
7. Promote only the clearest figures to `<experiment>/figures/`.

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
4. Sequence quality:
- each sequence has an explicit rationale tied to a SMART claim
- at least one benign, one hard, and one representative mixed regime
5. Documentation sync:
- update `README.md` and `figures/INDEX.md` together
6. Baseline clarity:
- README includes a robust-baseline specification section with source and appropriateness justification

## Safety

- Prefer non-destructive reorganizations.
- If a move could break imports or run scripts, update paths in the same change.
- If uncertain whether a file is manuscript-critical, leave it in `paper/` and ask before editing.
