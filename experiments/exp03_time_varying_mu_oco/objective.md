# Objective: Experiment 03 (Time-Varying $\mu_t$ in Quadratic OCO)

## What this experiment is trying to show

This experiment operationalizes the new extension request:

- Implement SMART when losses are
  $\ell_t(a)=\frac{1}{2}(a-\mu_t)^2$
  with time-varying centers `mu_t`, not only constant `mu_t=1/4`.

Core empirical question:

- Does SMART adapt appropriately as the environment shifts over time (e.g., step changes in `mu_t`) while retaining robustness when the optimistic trajectory is no longer best?

This directly supports the paper's broader claim that SMART is modular and extends beyond binary prediction into OCO-style settings.

### Paper-ready background paragraph

This experiment isolates a tractable convex setting where the loss landscape evolves through a time-varying center parameter `mu_t`. It is designed to test whether SMART can remain competitive when the environment is nonstationary but structured: constant phases, abrupt regime shifts, and stochastic drift in `mu_t`. The central empirical question is whether the switching mechanism can preserve optimistic behavior when appropriate while maintaining robustness when temporal variation makes pure FTL behavior unreliable.

## Experiment design

Implemented components:

- `src/oco_smart.py`:
  - FTL for quadratic sequence
  - OGD baseline (fixed `eta=2/sqrt(n)` and optional anytime `eta_t=1/sqrt(t)`)
  - SMART single-switch policy using Eq.(6)-style cumulative statistic and threshold `2*sqrt(n)`
- `src/sequence_generators.py`:
  - `constant_0.25`
  - `step_0.75_to_0.25` (requested)
  - `sine`
  - `uniform_random`
- `run_experiments.py`:
  - runs scenarios
  - summarizes final regret
  - outputs `mu` traces and regret plots

Notebook sources are archived under `notebooks/` for traceability.

## What counts as success

1. Constant scenario reproduces baseline behavior (sanity check).
2. Step-change scenario reveals whether SMART detects deterioration and benefits from switching.
3. Additional scenarios (`sine`, `uniform_random`) stress test adaptation under smooth/random drift.

### Figures intended for paper inclusion

Curated figure files are provided in `figures/` with labels and titles listed in `figures/INDEX.md`.

- `fig:exp03_constant_mu_regret` establishes baseline behavior under stationary loss centers.
- `fig:exp03_step_mu_schedule` and `fig:exp03_step_mu_regret` jointly present the regime-shift setup and algorithm response.
- `fig:exp03_random_mu_regret` supports claims about adaptation under stochastic nonstationarity.

## Problems and limitations we are currently encountering

1. Switching not triggering in current tested settings:
- In our current runs, SMART often never switches (switch round `n+1`), effectively matching FTL.
- This indicates the chosen threshold/statistic may be too conservative for these specific generated sequences.

2. Threshold calibration uncertainty:
- Directly porting `2*sqrt(n)` from earlier settings may not be best calibrated for all quadratic `mu_t` dynamics.

3. Limited adversarial `mu_t` stress tests so far:
- Current built-ins include simple step/sine/random, but not fully worst-case sequence construction for maximizing pre-switch optimistic regret.

4. Comparator/model-choice sensitivity:
- Conclusions depend on domain bounds, loss form, and OGD schedule choices.

## Immediate next steps for this experiment

1. Add stronger adversarial `mu_t` generators specifically designed to force bad FTL behavior before switching.
2. Evaluate threshold variants (theoretical vs empirical) and inspect switch statistics directly.
3. Add side-by-side diagnostics of `Sigma_t` versus threshold for each scenario to explain switching outcomes.
