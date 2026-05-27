# Experiment 01: Best-of-All-Worlds Binary Prediction

## Objective

This experiment is the binary-prediction companion to the OLC experiment. It studies the same SMART question in the simplest two-expert setting:

Can SMART stay close to an optimistic FTL-style policy on easy binary sequences while protecting against FTL's failures on adversarially structured sequences?

The experiment is useful for mechanism intuition. The OLC experiment is the cleaner paper-facing vector setting with the standardized regret-by-horizon protocol.

## Problem Setup

- Outcomes are binary: $x_t \in \{0,1\}$.
- Algorithms output probability-like actions $w_t \in [0,1]$ for predicting 1.
- Instant loss is Bernoulli linear loss:

$$
(1-x_t)w_t + x_t(1-w_t).
$$

- Regret is measured against the best fixed binary expert unless a specific script states a different comparator.

## Algorithms

The main SMART comparison uses:

- `FTL_binary` as the optimistic policy.
- `Cover_binary` as the robust fallback for deterministic SMART.
- `SMART_det_binary` and `SMART_random_binary` as the switching policies.

Other baselines in the folder, including AdaHedge, FlipFlop, ABProd, and Blackwell variants, are useful for exploratory comparison but are not the primary dashboard story.

## SMART Definitions

### Deterministic SMART

The implementation uses the binary FTL-regret proxy

$$
\mathrm{reg}_{\mathrm{ftl}}(t)=0.5\cdot\text{cumulative\_count}\{S_t=0\},
$$

where $S_t$ is the signed cumulative imbalance. The threshold is

$$
\theta=\sqrt{\frac{n}{2\pi}}.
$$

The policy plays FTL until the threshold is crossed, then switches to `Cover_binary` for the suffix.

### Randomized SMART

The randomized variant mixes FTL and Cover with a time-varying weight based on the same regret proxy and global scale

$$
g_n=\sqrt{\frac{n}{2\pi}}.
$$

It is intended to mirror the randomized-threshold SMART idea in a binary-expert setting.

## Sequence Families

### I.I.D. Bernoulli

`generate_iid_sequence(n,p)` samples each bit independently from $\mathrm{Bernoulli}(p)$.

Purpose: benign stochastic baseline where optimism should not be heavily penalized.

### FMG Sequence

`generate_FMG_sequence(n,c)` creates $c$ randomly oriented alternating pairs, then fills the remainder with ones. It requires $2c\le n$.

Purpose: control crossings and ties, exposing when following the leader is fragile.

### High-Loss FMG

`generate_highlossFMG_ftl(n,c,p)` starts with alternating pairs, appends `[1,1]`, then fills the rest with a Bernoulli tail.

Purpose: produce FMG-style structure with heavier optimistic stress.

### Loss-Shaped Sequences

`loss_based_sequence_generation` greedily chooses the next bit to keep cumulative expert-loss difference near a target function such as $n^{0.4}$.

Purpose: construct trajectories with specific regret-growth patterns and reveal differences among adaptive methods.

## Interpretation

This experiment supports the same qualitative SMART story as the OLC tab:

1. On benign binary streams, SMART should stay close to FTL.
2. On FMG, high-loss, and loss-shaped streams, SMART should reduce downside relative to pure FTL.
3. Across easy and hard families, SMART should act as a transparent compromise between optimism and robustness.

## Evaluation and Outputs

Main entry points:

- `graph_driver.py`
- `blackwells_worst_experimentation.py`
- `graphing_gui.py`

Core analysis code:

- algorithms: `Scripts/algorithms.py`
- sequence generation: `Scripts/sequence_generation.py`
- plotting: `Scripts/graphing.py`

Historical plots are stored under `Graphs/` and experiment root PNGs.

## Limits and Non-Claims

- This folder predates the current standardized experiment template.
- Comparator definitions vary across some scripts, so claims should be tied to the exact script used.
- Use this tab for binary mechanism intuition; use the OLC tab for the current paper-facing empirical story.
