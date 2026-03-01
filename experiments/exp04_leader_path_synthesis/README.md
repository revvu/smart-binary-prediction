# exp04_leader_path_synthesis

Experiment 04 studies SMART in online linear classification with a sequence-construction method that keeps the experiment realistic and computationally tractable.

## Background (Undergraduate-Level)

At each round `t`, the learner chooses a vector `x_t` in the unit ball (`||x_t||<=1`) and then sees an example `(z_t, y_t)`:

- `z_t` is a feature vector (`||z_t||<=1`),
- `y_t` is a label in `{ -1, +1 }`.

The loss used here is:

- $\ell_t(x)=\frac{1}{2}\left|\langle z_t,x\rangle-y_t\right|$.

Intuition:

1. `x_t` makes a score `q_t = <z_t, x_t>`.
2. If `q_t` is close to label `y_t`, loss is small.
3. If `q_t` is far from `y_t`, loss is large.

This experiment compares:

- `FTL` (optimistic),
- `FTRL` (robust baseline),
- `SMART` (start optimistic, switch when evidence says optimism is failing).

## Worst-Case Baseline Specification (Why This Is FTRL)

Robust baseline used in Exp04:

- **FTRL with quadratic regularization on the unit Euclidean ball**.

Concrete specification:

`x_t^FTRL = argmin_{x in X} { <theta_{t-1}, x> + (sqrt(t)/(2*eta0)) * ||x||_2^2 }`,  
with `X = {x : ||x||_2 <= 1}` and `theta_{t-1}` the cumulative linearized gradient.

Where this specification comes from:

1. This is the standard FTRL template: cumulative linearized loss + strongly convex regularizer.
2. Quadratic regularization over an `l2` ball yields a closed-form projected update:
   - `Proj_X( -(eta0/sqrt(t)) * theta_{t-1} )`.

Why this is appropriate here:

1. The loss is linearized each round in vector space, so FTRL is directly applicable.
2. The domain is exactly bounded in `l2`, matching the projection form.
3. It provides a robust, stable baseline that is meaningful against optimistic FTL in nonstationary regimes.

## Why True FTL Becomes Difficult in This Setting

This is the core motivation for the experiment redesign.

In online linear classification, a literal “true FTL” benchmark over prefixes can be expensive because:

1. At each round `t`, FTL depends on cumulative gradient/state up to `t-1`.
2. If you build arbitrary synthetic sequences and then try to recover exact leader behavior by repeated optimization, you may need many prefix solves.
3. Doing this over many horizons and many trials multiplies the cost heavily.

In short: if sequence generation and leader computation are decoupled, exact repeated prefix optimization can become a bottleneck.

## How We Redesigned the Experiment

We redesigned Exp04 so leader behavior is computable in closed form by construction.

High-level idea:

1. Instead of first generating arbitrary `(z_t, y_t)` and then solving for leaders,
2. we construct a realizable path of cumulative gradients `theta_t`,
3. and then generate `(z_t, y_t)` to match that path.

This makes FTL updates simple and fast.

## Step-by-Step: Why Results Are Computable in the New Formulation

Let `u_t` be a desired leader direction at round `t`.

### Step 1: Choose regime-specific target directions

For each scenario, define a target direction path:

- `u_1, ..., u_T`, where each `u_t` is unit-norm.

### Step 2: Choose magnitude schedule

Define radii `r_t` controlling how strongly the cumulative state points in that direction.

Construct desired cumulative state:

- `theta_t^des = - r_t * u_t`.

### Step 3: Enforce per-round realizability

Per-round gradient updates must be feasible, so we cap increment size:

- `Delta_t = theta_t - theta_{t-1}`,
- enforce `||Delta_t|| <= 0.5`.

This matches the maximum update size implied by bounded features in this loss setup.

### Step 4: Convert `Delta_t` into an example `(z_t, y_t)`

We choose feature-label pairs so the induced gradient equals the desired update.

Using the gradient form

- `g_t = 0.5 * sign(<z_t, x_t> - y_t) * z_t`,

we set `z_t` proportional to `Delta_t` (with sign and label choice selected to match plausibility and update direction), so `g_t` reproduces `Delta_t`.

### Step 5: Compute FTL cheaply

Now cumulative state is simply:

- `theta_t = sum_{s<=t} g_s`.

FTL action is closed form:

- `x_t^FTL = -theta_{t-1} / ||theta_{t-1}||` (or `0` if denominator is zero).

No repeated numerical optimizer is needed.

### Step 6: Run horizon-level evaluation

For each horizon `n`:

1. generate fresh sequences of length `n` from the same regime recipe,
2. run `FTL`, `FTRL`, and `SMART`,
3. record regret at horizon `n`.

This yields the paper-facing `Regret` vs `Horizon` figure.

## Metric-Driven Sequence Selection (New)

To improve figure quality, Exp04 now uses a metric-driven selection stage before the main horizon sweep.

What is searched:

1. `max_delta_norm` candidates: controls how quickly cumulative state can move each round.
2. `label_mismatch_prob` candidates: controls base mismatch/corruption intensity.

How candidates are scored:

1. `stable_benign` score favors `SMART ~= FTL` and penalizes cases where robust beats FTL.
2. `persistent_shift` score favors large improvement of SMART over FTL under a sustained post-change regime.
3. `delayed_hardening` score favors meaningful improvement over FTL while preserving a mixed-regime narrative.
4. All regimes include penalties for unstable/non-smooth curves and excessively wide confidence bands.

How selected parameters are used:

1. For each regime, evaluate a small parameter grid on probe horizons.
2. Choose the best-scoring parameter pair for that regime.
3. Run the full horizon figure using those selected parameters.

This keeps realism constraints while making the paper figure more illustrative.

## Input Sequence Design (Exact Regime Recipes)

This section is the most important part of Exp04.

### 1) `stable_benign` (FTL-dominant)

Purpose:
- show optimism should be rewarded when environment is stable.

Generation details:
1. target directions stay near a fixed anchor direction with small noise,
2. radius schedule grows smoothly,
3. low mismatch/noise in label orientation choice.

Expected behavior:
- `FTL ≈ SMART`, robust baseline more conservative.

### 2) `persistent_shift` (hard regime)

Purpose:
- model a single sustained regime break after a long benign prefix.

Generation details:
1. pre-shift rounds: target directions stay near a stable anchor,
2. post-shift rounds: target direction jumps to a different anchor and stays there,
3. cumulative-state magnitude is reduced after the break to make optimism less reliable,
4. mismatch probability remains controlled to keep variance lower than bursty constructions.

Expected behavior:
- FTL regret worsens clearly at larger horizons; SMART should switch and reduce damage.

### 3) `delayed_hardening` (representative mixed regime)

Purpose:
- model a realistic progression from easy to moderately hard to hard.

Generation details:
1. early phase: stable benign dynamics,
2. middle phase: gradual hardening via smooth drift,
3. late phase: moderate persistent shift to a new regime,
4. bounded increments and controlled noise preserved.

Expected behavior:
- SMART should provide a middle ground between optimism and robustness with cleaner horizon-level separation.

## Primary Figure Contract

Paper-facing figure is:

- x-axis: `Horizon`,
- y-axis: `Regret`,
- each horizon uses fresh sequences of that exact length.

No single-sequence switch-statistic traces are included in curated paper figures.

## Acceptance Criteria (Paper-Facing)

1. `stable_benign`: SMART remains close to FTL over horizons.
2. `persistent_shift`: SMART is clearly better than FTL over horizons.
3. `delayed_hardening`: SMART tracks a favorable tradeoff in mixed nonstationarity.

## Online Learning Definition Used in Code

- Action set: `X = {x in R^d : ||x||_2 <= 1}`.
- Loss: `ell_t(x) = 0.5 * |<z_t, x> - y_t|`.
- FTRL baseline:
  - `x_t^FTRL = argmin_{x in X} { <theta_{t-1}, x> + (sqrt(t)/(2*eta0)) * ||x||_2^2 }`
  - implemented as projection of `-(eta0/sqrt(t))*theta_{t-1}` onto `X`.

## Run

```bash
cd experiments/exp04_leader_path_synthesis
python run_experiment.py --t-max 1000 --t-step 100 --runs 12 --d 5 --max-delta-norm 0.45
```

Disable metric-driven auto-selection and force manual parameters:

```bash
python run_experiment.py --no-auto-select --max-delta-norm 0.45 --label-mismatch-prob 0.02
```

## Outputs

- `outputs/figures/exp04_olc_regret_by_horizon.png`

Curated paper figure:

- `figures/fig_exp04_olc_regret_by_horizon.png`

## Files

- `src/synthesis.py`: regime definitions and realizable sequence construction.
- `src/eval.py`: FTL/FTRL/SMART evaluation.
- `run_experiment.py`: horizon sweep and figure generation.

## Known limitations

1. Sequences are synthetic but structured; realism is controlled by model assumptions.
2. Threshold scale calibration remains important.
3. Conclusions depend on the chosen surrogate and comparator definitions.
