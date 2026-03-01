# exp03_time_varying_mu_oco

Experiment 03 studies SMART in a simple 1D online convex optimization setting with quadratic losses:

- $\ell_t(a)=\frac{1}{2}(a-\mu_t)^2$

The implementation in this folder is a cleaned Python version derived from the uploaded notebooks.

## Background (Undergraduate-Level)

At each round `t`, the algorithm chooses a number `a_t` in a fixed interval (the code uses `[-1, 1]`), then observes a loss function

- $\ell_t(a)=\frac{1}{2}(a-\mu_t)^2$.

This is just a parabola centered at `mu_t`. Intuitively:

- if `a_t` is close to `mu_t`, loss is small;
- if `a_t` is far from `mu_t`, loss is large.

So the full problem is about tracking a moving target `mu_t` over time.

Why this experiment matters for SMART:

1. It is mathematically clean and easy to interpret.
2. It isolates sequence effects: how the environment (`mu_t`) changes controls which algorithm wins.
3. It lets us test SMART’s intended role: optimistic when safe, robust when needed.

## Objective

Demonstrate three core SMART behaviors:

1. In benign environments, SMART should behave like the optimistic baseline (FTL).
2. In harder nonstationary environments, SMART should avoid optimistic failures.
3. Across horizons, SMART should track a favorable tradeoff between optimistic and robust baselines.

## Algorithms in this experiment

- `FTL`: Follow-The-Leader baseline.
- `OGD`: robust baseline (fixed or anytime learning rate).
- `SMART`: starts optimistically and switches to robust behavior using the paper’s switching statistic.

## Input Sequence Design: What `mu_t` Means and How We Generate It

The input sequence is the most important design choice in this experiment.  
Each scenario defines a full horizon-length sequence `mu_1, ..., mu_n`.

### 1) `stable_benign` (FTL-dominant)

Goal:
- represent a mostly stable environment where optimism should work.

Generation (step-by-step):
1. Start near a positive center (`~0.60`).
2. Update with an AR(1)-style rule:
   - strong pull to the center,
   - small Gaussian noise each round.
3. Clip values to stay in a bounded range.

Effect:
- `mu_t` moves slowly with low noise, so FTL should perform well.

### 2) `corruption_burst` (adversarial-realistic)

Goal:
- represent a stable process with short, severe disruptions.

Generation (step-by-step):
1. Start from a stable positive baseline (`~0.55`) plus small noise.
2. Insert two time windows (bursts).
3. Inside each burst, force `mu_t` to a strong negative regime (plus noise).
4. Outside bursts, return to baseline behavior.

Effect:
- an optimistic policy that trusts recent history can be damaged during bursts.
- SMART should gain by switching.

### 3) `drift_plus_shift` (representative mixed regime)

Goal:
- represent a practical nonstationary stream with both gradual and abrupt change.

Generation (step-by-step):
1. Early phase: long benign regime near `~0.55`.
2. Middle phase: gradual drift downward.
3. Late phase: one sustained shift to a negative regime (`~ -0.70` with noise).
4. Clip to bounds.

Effect:
- this combines easy and hard periods in one realistic-style trajectory.

## Experimental Protocol (Primary Figure Contract)

Primary figure type is `Regret` vs `Horizon`:

1. Choose a grid of horizons `n` (for example `100, 200, ..., 1000`).
2. For each horizon `n`, generate fresh sequences of length `n`.
3. Run all algorithms on each sequence.
4. Record regret at the end of that horizon.
5. Plot mean regret (and uncertainty bands when stochastic) against horizon.

This is intentionally not a single-sequence prefix trace.

## Acceptance Criteria (Paper-Facing)

1. `stable_benign`: SMART is close to FTL across horizons.
2. `corruption_burst`: SMART is substantially better than FTL as horizon grows.
3. `drift_plus_shift`: SMART lies between optimistic and robust extremes in a meaningful way.

## Threshold Calibration

In this quadratic setting, the raw switching statistic can be numerically small relative to default theoretical scaling, so:

- we expose `--threshold-scale` (default `0.0035`),
- and report results under that explicit calibration.

## Run

```bash
cd experiments/exp03_time_varying_mu_oco
python run_experiments.py --n-max 1000 --n-step 100 --trials 30 --threshold-scale 0.0035
```

Anytime OGD variant:

```bash
python run_experiments.py --n-max 1000 --n-step 100 --trials 30 --threshold-scale 0.0035 --anytime-lr
```

Run specific scenarios:

```bash
python run_experiments.py --scenario stable_benign corruption_burst drift_plus_shift
```

## Outputs

- `outputs/figures/exp03_quadratic_oco_regret_by_horizon.png`

Curated paper figure:

- `figures/fig_exp03_quadratic_oco_regret_by_horizon.png`

## Known limitations

1. Sequences are synthetic but structured; they are not fitted from a real dataset.
2. Conclusions depend on the chosen domain bounds and robust baseline (OGD schedule).
3. Threshold calibration remains an explicit modeling choice.
