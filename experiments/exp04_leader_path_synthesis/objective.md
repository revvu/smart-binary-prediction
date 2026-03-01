# Objective: Experiment 04 (Realistic Leader-Path Synthesis for Online Linear Classification)

## What this experiment is trying to show

This experiment is designed to produce **more realistic and interpretable demonstrations** of SMART in online linear classification, while still spanning the full range of behaviors needed for the paper:

1. Benign regimes where optimistic behavior (FTL-like) is strong.
2. Non-stationary/adversarial phases where robust behavior matters.
3. SMART adapts across regimes and tracks the better side of the optimism/robustness tradeoff.

The purpose is to replace overly obvious synthetic patterns (e.g., trivial alternating labels) with sequences that look more like plausible data streams under covariate and concept shift.

## Experimental setup

### Data generation model

- Dimension fixed to `d = 5`.
- Features are sampled from a drifting Gaussian model with bounded norm:
  - time-varying per-dimension scales,
  - fixed random rotation,
  - projection to unit ball.
- Base labels come from a latent separator path `w*_t` with additive noise.

### Leader-driven synthesis

At each round `t`, for a chosen target leader `u_t`:

1. Sample many candidate points `z_t` from the realism prior.
2. Construct candidate labels from latent base label (with optional flip).
3. Score each candidate by a weighted objective:
   - leader matching: next realized FTL leader should be close to target `u_t`,
   - realism: avoid implausible margins/extremes,
   - flip penalty: discourage excessive artificial label inversions.
4. Pick the best candidate and append it to the sequence.

This produces a **realizable sequence of leaders by construction** (for generated data), without solving a full inverse optimization at each prefix.

### Regimes used

- `stable_benign`
- `gradual_drift`
- `regime_shift`
- `bursty_corruption`

These are meant to mimic practical transitions rather than toy worst-case constructions.

### Evaluation

We evaluate FTL, FTRL, and SMART on synthesized sequences:

- final regret vs horizon across regimes,
- diagnostic plots with:
  - regret trajectories,
  - SMART switch statistic vs threshold,
  - target-vs-realized leader alignment.

For demonstration clarity, SMART uses a calibrated threshold scale (base plus regime-specific adjustment for corruption bursts), analogous to empirical-threshold tuning in prior OLC experiments.

## What counts as success

1. Regime plots show distinct difficulty profiles and nontrivial behavior.
2. SMART performs near FTL in benign/stable settings.
3. SMART avoids large degradation in harder/non-stationary regimes.
4. Diagnostics show that switching dynamics are interpretable relative to regime changes.

## Known limitations and issues

1. Approximate, not exact inverse realization:
- We optimize one step ahead with candidate search; we do not solve a global inverse problem for a prescribed full leader path.

2. Objective-weight sensitivity:
- Results depend on leader/realism/flip weights and candidate count.

3. Computational scaling:
- Larger horizon and candidate sets increase runtime.

4. Comparator choice dependence:
- Regret conclusions depend on the same surrogate and comparator definitions used in this OLC codebase.

## Why this helps the paper

This setup preserves the conceptual point of the existing experiments (showing SMART across easy-to-hard regimes) while improving realism and narrative quality: behavior changes are induced by plausible covariate/label dynamics, not by obviously hand-engineered sequences.
