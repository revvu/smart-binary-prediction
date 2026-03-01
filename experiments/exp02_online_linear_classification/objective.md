# Objective: Experiment 02 (Online Linear Classification)

## What this experiment is trying to show

This experiment extends the SMART narrative beyond binary prediction to online linear classification.

Primary claim to demonstrate:

- SMART can preserve FTL-like/FTRL-like strong performance on benign data while preventing blow-ups on adversarial label patterns, consistent with the paper's modular instance-optimality message for broader online learning settings.

Secondary claim:

- Using empirical estimates of the robust threshold `g(T)` can improve practical switching behavior versus a fixed theoretical bound.

## Experiment design

Problem setup (as implemented):

- Sequential linear prediction with bounded feature vectors and comparator in bounded norm-ball.
- Loss is an absolute-margin style surrogate (`0.5 * |z_t · x_t - y_t|` in code family).

Compared policies:

- FTL/FTRL baselines
- SMART with theoretical threshold (`sqrt(2T)`-style reference)
- SMART with empirical `g(T)`

Sequence families (`sequence_generation.py`):

- Random i.i.d.-style linearly separable stream
- Massart-noise stream
- Label flips (highly adversarial)
- Switching leaders

Drivers:

- `driver.py`: fast baseline pipeline for main comparison plots.
- `fast_driver.py`: optimized variant.
- `exact_ftl_driver.py`: exact FTL benchmark path via convex optimization.

## What counts as success

- On benign streams: SMART tracks optimistic baseline quality.
- On adversarial streams: SMART avoids linear-regret collapse and remains competitive with robust baseline.
- Empirical-threshold SMART shows meaningful practical gains when switching earlier/later than conservative theoretical thresholds.

## Problems and limitations we are currently encountering

1. Exact FTL is computationally intensive (key issue):
- True FTL in this setting requires solving repeated constrained optimization problems over prefixes.
- `exact_ftl.py`/`exact_ftl_driver.py` rely on CVXPY and repeated solves; runtime scales poorly with horizon and number of replications.
- This makes full-grid experiments expensive and limits exact benchmarking throughput.

2. Cost of empirical `g(T)` estimation:
- Estimating worst-case-style threshold empirically requires many runs (`g_runs`) and can be expensive, especially combined with exact solvers.

3. Benchmark-definition sensitivity:
- Small differences in loss surrogate, normalization, or comparator definition materially change regret values, so configuration drift is a risk.

4. Practical trade-off:
- Fast approximations are tractable but can diverge from exact FTL behavior; exact runs are principled but slow.

## Immediate next steps for this experiment

1. Cache/warm-start prefix solves more aggressively and reduce redundant exact solves.
2. Restrict exact-FTL runs to smaller diagnostic grids; use fast pipeline for broad sweeps.
3. Add explicit "paper config" presets (horizon, runs, replicates, solver settings) to lock comparability.
