# Experiment 01: Best-of-All-Worlds Binary Prediction (Legacy)

## Why this experiment exists

This is the original binary-prediction sandbox where the SMART idea was first stress-tested against both optimistic and robust alternatives.

Core question:

In two-expert binary prediction, can SMART track optimistic performance on easy sequences while avoiding the worst optimistic failures on adversarially structured sequences?

This folder is intentionally kept as a legacy experiment environment. It is useful for conceptual evidence and historical baselines, but it is less standardized than later experiments.

## Problem setup

- Outcomes are binary: $x_t \in \{0,1\}$.
- Algorithms output probability-like actions $w_t \in [0,1]$ for predicting 1.
- Instant loss is Bernoulli linear loss:
  $$
  (1-x_t)w_t + x_t(1-w_t)
  $$
- Regret is measured against the best fixed binary expert (always-0 vs always-1) unless a deeper context comparator is explicitly used.

## Algorithms emphasized for SMART comparison

- `FTL_binary` (optimistic)
- `Cover_binary` (robust fallback used by deterministic SMART in this experiment code)
- `SMART_det_binary` and `SMART_random_binary`

Additional baselines in this legacy stack (AdaHedge, FlipFlop, ABProd, Blackwell variants) remain available for broader comparisons.

## SMART definitions used in this legacy code

### Deterministic SMART (`SMART_det_binary`)

- FTL-regret proxy in this implementation:
  $$
  \mathrm{reg}_{\mathrm{ftl}}(t) = 0.5\cdot \text{cumulative\_count}\{S_t=0\},
  $$
  where $S_t$ is signed cumulative imbalance.
- threshold:
  $$
  \theta = \sqrt{\frac{n}{2\pi}}
  $$
- policy:
  - play FTL until threshold crossing, then switch to Cover for the suffix

### Randomized SMART (`SMART_random_binary`)

- mixes FTL and Cover with time-varying weight based on the same regret proxy and global scale
  $$
  g_n = \sqrt{\frac{n}{2\pi}}
  $$
- intended to emulate the randomized-threshold spirit in a binary-expert setting

## Sequence families (exact constructions)

### 1) i.i.d. Bernoulli (`generate_iid_sequence(n,p)`)

- sample each bit independently from $\mathrm{Bernoulli}(p)$.

Purpose:

- benign/stochastic baseline where optimism should not be heavily penalized.

### 2) FMG sequence (`generate_FMG_sequence(n,c)`)

- create $c$ randomly oriented alternating pairs ($01$ or $10$)
- fill the remainder with ones
- constraint: $2c \le n$

Purpose:

- control number of crossings/ties and stress optimistic tie behavior.

### 3) High-loss FMG (`generate_highlossFMG_ftl(n,c,p)`)

- first $2c$ bits are alternating pairs
- then append $[1,1]$
- fill remainder with Bernoulli tail (default $p=0.3$)

Purpose:

- produce FMG-style structure with heavier optimistic stress.

### 4) Loss-shaped sequences (`loss_based_sequence_generation`)

- greedily choose next bit to keep cumulative expert-loss difference close to a target function (for example $n^{0.4}$)

Purpose:

- construct trajectories that reproduce specific regret-growth patterns and expose weaknesses of some adaptive-hedging baselines.

## What this experiment is intended to demonstrate

1. On i.i.d.-style sequences, SMART should stay close to optimistic behavior.
2. On FMG / high-loss / loss-shaped sequences, SMART should reduce downside relative to pure FTL.
3. Across easy and hard families, SMART should behave like a practical compromise between optimism and robustness.

## Evaluation and outputs in this legacy stack

Main entry points:

- `graph_driver.py`
- `blackwells_worst_experimentation.py`
- `graphing_gui.py`

Core analysis code:

- algorithms: `Scripts/algorithms.py`
- sequence generation: `Scripts/sequence_generation.py`
- plotting: `Scripts/graphing.py`

Historical plots are stored under `Graphs/` and experiment root PNGs.

## Current empirical interpretation

Historically, this experiment established the qualitative SMART narrative:

- SMART behaves close to FTL on benign binary streams.
- SMART avoids some of FTL's worst binary adversarial constructions by switching to a safer policy.
- Carefully designed loss-shaped sequences are important for revealing differences among adaptive methods.

## Limits and non-claims

- This is a legacy, non-unified pipeline (multiple drivers, historical outputs mixed together).
- Comparator definitions vary across some scripts (binary expert vs context-tree expert), so claims must be tied to the exact script used.
- This folder is strongest as historical mechanism evidence, not as the final standardized paper pipeline.
