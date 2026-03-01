# exp04_leader_path_synthesis

This experiment builds **realistic-looking online linear classification sequences** by steering toward target leader trajectories while keeping sampled points plausible.

It is designed to demonstrate the full SMART story without relying on simplistic handcrafted sequences.

## Key idea

Instead of manually hard-coding obvious label patterns, we:

1. Define a target leader direction path for each regime.
2. Construct a realizable cumulative gradient path `theta_t` with bounded increments `||Delta_t|| <= 0.5`.
3. Convert each increment into a valid `(z_t, y_t)` pair so the FTL update recovers the desired increment.
4. Add realistic mismatch windows (for corruption-burst only) without breaking bounded-feature constraints.

This keeps leader computation cheap and closed-form while ensuring realizability by construction.

## Objective

This experiment is designed to produce more realistic and interpretable demonstrations of SMART in online linear classification while still covering the full range of behaviors needed for the paper:

1. Benign regimes where optimistic behavior (FTL-like) is strong.
2. Non-stationary/adversarial phases where robust behavior matters.
3. SMART adapts across regimes and tracks the better side of the optimism/robustness tradeoff.

### Paper-facing sequence suite

Experiment 4 should focus on exactly three realistic sequences:

- `stable_benign`
- `corruption_burst`
- `drift_plus_shift`

#### 1) `stable_benign` (FTL-dominant)

- Construction: fixed latent separator, mild covariate drift, low symmetric label noise.
- Why realistic: stable environment with ordinary measurement noise.
- Intended takeaway: SMART should stay close to FTL (minimal optimism tax).

#### 2) `corruption_burst` (adversarial-realistic)

- Construction: mostly stable stream plus short windows of targeted label corruption and feature concentration.
- Why realistic: transient data quality/manipulation events.
- Intended takeaway: FTL suffers during bursts; SMART should switch and cap damage.

#### 3) `drift_plus_shift` (representative mixed regime)

- Construction: gradual separator drift followed by one moderate abrupt shift, with moderate noise.
- Why realistic: population drift with occasional context/policy changes.
- Intended takeaway: SMART should interpolate between optimism and robustness with interpretable switch timing.

### Core takeaways this experiment must support

1. SMART preserves upside in easy regimes (`stable_benign`).
2. SMART protects against hard but realistic events (`corruption_burst`).
3. SMART adapts coherently in mixed nonstationarity (`drift_plus_shift`).

### Primary plotting contract

- Primary paper figure type: `final regret vs horizon`.
- For each horizon `n`, evaluate on fresh sequences of length `n`.
- Do not use single-sequence switch/statistic traces in curated paper figures.

## Online learning setup and FTRL definition

At round `t`, the learner picks `x_t` in the unit Euclidean ball:

`X = {x in R^d : ||x||_2 <= 1}`.

Given `(z_t, y_t)` with `||z_t||_2 <= 1` and `y_t in {-1, +1}`, we use

`ell_t(x) = 0.5 * |<z_t, x> - y_t|`.

In code, the linearized per-round gradient used by all methods is

`g_t = 0.5 * sign(<z_t, x_t> - y_t) * z_t` (with `0` at ties),

and `theta_t = sum_{s=1}^t g_s`.

FTRL is defined as

`x_t^FTRL = argmin_{x in X} { <theta_{t-1}, x> + (sqrt(t)/(2*eta0)) * ||x||_2^2 }`.

This has the closed form implemented in `src/eval.py`:

`x_t^FTRL = Proj_X( -(eta0/sqrt(t)) * theta_{t-1} )`.

So in this experiment, "FTRL" is the standard quadratic-regularized, linearized-loss variant (equivalently projected gradient with decaying step size).

FTL is

`x_t^FTL = argmin_{x in X} <theta_{t-1}, x> = -theta_{t-1} / ||theta_{t-1}||` (or `0` if `theta_{t-1}=0`),

and SMART starts from FTL and switches once its internal FTL-regret statistic crosses a threshold.

## Files

- `src/synthesis.py`: realizable theta/leader path generators and sequence synthesis.
- `src/eval.py`: FTL/FTRL/SMART regret curves and switch-stat diagnostics.
- `run_experiment.py`: end-to-end runner producing figures.

## Run

```bash
cd experiments/exp04_leader_path_synthesis
python run_experiment.py --t-max 1000 --t-step 100 --runs 8 --d 5 --max-delta-norm 0.45
```

Notes:

- The runner uses an empirically calibrated threshold scale for SMART (`--threshold-scale`, default `0.01`) to expose meaningful switch behavior in this OLC setting.
- A single threshold scale is used across all three sequences for comparability.
- For paper figures, keep outputs restricted to horizon-vs-final-regret comparisons across the three sequence types.

## Outputs

- `outputs/figures/exp04_olc_final_regret_by_horizon.png`

## Figures

Curated paper-candidate figures live in `figures/` with labels/titles mapped in `figures/INDEX.md`.

## Known issues

1. Sequence realism is model-based; it depends on chosen drift, burst windows, and mismatch rates.
2. Threshold calibration remains sensitive; too high suppresses switching, too low switches too early.
3. The construction targets realizability and interpretability, not distributional fidelity to any specific real dataset.
4. Regret comparisons depend on the chosen surrogate loss and comparator definition.
