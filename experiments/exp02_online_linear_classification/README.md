# Experiment 02: Online Linear Classification SMART Study

## Why this experiment exists

This experiment tests SMART in a vector-valued online linear classification setting where sequence hardness can vary from easy separable data to adversarial label patterns.

Core question:

Can SMART keep optimistic behavior when FTL is good, while switching in time to avoid optimistic failure modes on harder streams?

A second question in this experiment is calibration:

Does using an empirical robust threshold $g(T)$ improve practical switching versus purely theoretical scaling?

## Problem setup

At each round, the learner predicts with $x_t$ constrained to the unit Euclidean ball, observes $(z_t, y_t)$, and is scored with the surrogate used in code:

$$
\ell_t(x) = \tfrac{1}{2}\,|\langle z_t, x\rangle - y_t|.
$$

The reported metric is cumulative regret relative to the best fixed comparator action on the full horizon.

## Algorithms compared

- `FTRL` (robust baseline)
- `FTL` (optimistic baseline)
- `SMART` with theoretical threshold ($\theta = \sqrt{2T}$)
- `SMART` with empirical threshold ($\theta = g_{\text{emp}}(T)$)

### Robust baseline specification

The robust baseline is FTRL with Euclidean projection ($\eta_0=\sqrt{2}$), matching the bounded linear-prediction geometry used throughout the experiment.

Why this baseline is appropriate here:

1. It is the canonical robust online method for this constrained linear setting.
2. It gives the fallback behavior SMART switches to.
3. The same baseline defines empirical $g(T)$ (worst observed robust regret across adversarial families), so threshold calibration and robust comparator are internally consistent.

## Sequence families (exact conceptual definitions)

### 1) Random i.i.d. (separable)

- sample a fixed unit separator $u$
- sample $z_t \sim \mathcal{N}(0, I_d)$, then normalize each row to norm $\le 1$
- set labels $y_t = \operatorname{sign}(\langle z_t, u\rangle)$

Purpose:

- benign baseline where optimism should be competitive.

### 2) Massart noise 10%

- same construction as separable stream
- independently flip each label with probability $p = 0.10$

Purpose:

- mostly benign stream with moderate stochastic corruption.

### 3) Label flips

- deterministic feature direction ($z_t = e_1$)
- labels alternate every round: $+1, -1, +1, -1, \ldots$

Purpose:

- hard oscillatory stream that stresses optimistic stability.

### 4) Switching leaders

- deterministic feature direction ($z_t = e_1$)
- labels in fixed blocks (default block length 20):
  - $+1$ block, then $-1$ block, alternating

Purpose:

- clean leader-switching regime without adding geometric drift.

## SMART threshold calibration used here

Two SMART variants are intentionally evaluated:

1. Theory-scaled SMART: $\theta = \sqrt{2T}$.
2. Empirical-threshold SMART: $\theta = g_{\text{emp}}(T)$.

$g_{\text{emp}}(T)$ is estimated by running robust FTRL over designated hard families and taking the maximum observed robust regret for each horizon in $T$-grid.

This isolates a practical question: whether calibrated thresholds improve switch timing and regret trade-offs.

## Evaluation protocol

Default horizon grid in driver:

- $T = 100, 200, \ldots, 1000$

For each sequence family, the evaluation uses case-specific averaging budgets (`runs` x `replicates`) and reports mean regret with 95% confidence intervals.

The primary deliverables are:

- `algorithm_comparison.png` (algorithm regret by sequence family)
- `empirical_g_T.png` ($g_{\text{emp}}(T)$ versus reference curves)

Curated paper figures are tracked in `figures/INDEX.md`.

## Current empirical interpretation

Based on curated figures in `figures/`:

1. On separable / low-noise families, optimistic behavior is strong and SMART is competitive with FTL.
2. On label-flip and switching-leader families, pure optimism is less reliable; SMART variants improve robustness by switching.
3. Empirical-threshold SMART can differ materially from theory-threshold SMART, showing that threshold calibration is not cosmetic.

## What this experiment supports

This experiment supports two paper-relevant claims:

- SMART's adaptation story extends from scalar toy settings to vector OLC streams.
- Threshold calibration ($g(T)$ modeling) is an important empirical component of SMART, not a minor implementation detail.

## Limits and non-claims

- Sequence families are synthetic stress tests, not dataset-level claims.
- Results are surrogate-loss and comparator-definition dependent.
- The current fast path should not be treated as a trusted true-FTL implementation on boundary-heavy deterministic streams. With $\|z_t\|\le 1$ and $y_t\in\{\pm1\}$, the surrogate loss is globally linear on the unit ball:

$$
\ell_t(x)=\tfrac12(1-y_t\langle z_t,x\rangle).
$$

True FTL is therefore computable in closed form from $M_t=\sum_{i\le t}y_i z_i$:

$$
x_t^{\mathrm{FTL}}=
\begin{cases}
M_{t-1}/\|M_{t-1}\|_2, & M_{t-1}\ne 0,\\
0, & M_{t-1}=0.
\end{cases}
$$

The older fast implementation updates using a chosen subgradient at the played prediction. On deterministic norm-one streams such as blockwise $z_t=e_1$, this tie convention can diverge from the exact linear-loss FTL path and can even produce invalid negative regret because the comparator is then computed from the wrong state. A redesign should replace subgradient-state FTL with the closed-form linear-loss state above.

## Outputs and code map

- Curated figures index: `figures/INDEX.md`
- Stream definitions: `sequence_generation.py`
- Main algorithm simulation: `algorithms.py`
- Fast implementation path: `fast_algorithms.py`, `fast_driver.py`
- Exact FTL path: `exact_ftl.py`, `exact_ftl_driver.py`
- Main experiment driver: `driver.py`
