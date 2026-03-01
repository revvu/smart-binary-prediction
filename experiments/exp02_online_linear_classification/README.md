# exp02_online_linear_classification

OCO experiment set comparing FTRL, FTL, SMART, and empirical SMART variants.

## Objective

This experiment extends the SMART narrative to online linear classification:

- SMART should preserve optimistic performance on benign streams.
- SMART should avoid blow-ups on adversarial label patterns.
- Empirical threshold estimates can improve practical switch timing versus conservative theory-only thresholds.

## Experimental design

Problem setup:

- Sequential linear prediction with bounded features and comparator in a bounded norm ball.
- Loss surrogate: `0.5 * |z_t · x_t - y_t|` (code family).

Compared policies:

- FTL/FTRL baselines
- SMART with theoretical threshold
- SMART with empirical `g(T)`

Sequence families (`sequence_generation.py`):

- random i.i.d.-style linearly separable stream
- Massart-noise stream
- adversarial label-flip stream
- switching-leader stream

## Entry points

- `python driver.py`
- `python fast_driver.py`
- `python exact_ftl_driver.py`

## Main code locations

- `algorithms.py`, `fast_algorithms.py`, `exact_ftl.py`
- `sequence_generation.py`
- pre-generated figures at repository root of this experiment

## Figures

Curated paper-candidate figures live in `figures/` with labels/titles mapped in `figures/INDEX.md`.

## Known issues

1. Exact FTL is computationally expensive due to repeated constrained prefix solves.
2. Empirical `g(T)` estimation is expensive for large `g_runs`.
3. Regret values are sensitive to surrogate/normalization/comparator definitions.
4. Fast approximations are tractable but can drift from exact FTL behavior.

## Notes

- This experiment retains original layout for compatibility.
- Use this folder as the source of truth for OLC figures.
