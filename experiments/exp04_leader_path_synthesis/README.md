# Experiment 04: Leader-Path Synthesis for SMART in Online Linear Classification

## Why this experiment exists

This experiment is designed to answer one practical question for the paper:

Can SMART preserve optimistic performance in benign regimes while still protecting against nonstationary regime shifts in online linear classification?

The experiment is intentionally *illustrative* rather than fully realistic. The goal is to produce clean, interpretable evidence for the mechanism behind SMART (switching from optimistic to robust behavior at the right time), not to claim a domain benchmark.

## What this experiment is testing

We evaluate three policies:

- `FTL` (optimistic baseline)
- `FTRL` (robust baseline)
- `SMART` (starts as FTL, switches once to FTRL based on an online regret signal)

The target empirical story is:

1. In benign conditions, SMART should look like FTL (no optimism tax).
2. Under sustained shift, SMART should improve over pure FTL.
3. In mixed regimes, SMART should land between optimistic and robust extremes in a sensible way.

## Redesign gate: true-FTL OLC revision

The next revision should keep the useful part of this experiment--closed-form true FTL--but remove the hand-tuned leader-path narrative as the primary evidence.

### SMART behavior claim

The experiment should demonstrate three paper-facing claims:

1. **Preserve optimism:** on benign predictable streams, SMART should match true FTL and avoid the robust FTRL tax.
2. **Protect in hard regimes:** on sustained adversarial or high-variation streams, SMART should switch before FTL regret grows linearly and should remain close to FTRL.
3. **Interpretable switch:** in mixed streams, the switch should be explained by the exact monotone FTL trace $\Sigma_t$ crossing a horizon-calibrated threshold near the onset of sustained deterioration.

### Exact sequence families

Primary families should be generated directly in feature-label space with $\|z_t\|_2\le \rho<1$ to avoid boundary subgradient ambiguity while preserving exact FTL:

1. **I.i.d. separable margin stream:** sample a unit separator $u$, draw normalized covariates, optionally bias samples toward a positive margin, and set $y_t=\operatorname{sign}(\langle z_t,u\rangle)$.
2. **Massart-noise stream:** use the same separator model, but flip labels with probability bounded by $\eta<1/2$, with both a constant-noise version and a late high-noise suffix.
3. **Alternating anti-leader stream:** use one or two fixed directions and choose signs that force repeated leader cancellation or reversal. This is illustrative, not representative; it verifies robust protection in a known FTL failure mode.
4. **Benign-to-hard suffix stream:** start with the separable margin stream, then transition to either high-rate Massart noise or alternating anti-leader blocks. This is the main SMART sequence because it has an easy prefix where FTL is valuable and a sustained hard suffix where switching is valuable.
5. **Abrupt/gradual separator drift diagnostic:** rotate the separator over a window while keeping bounded features. This is representative of concept drift, but should be interpreted carefully because static regret to the best fixed comparator can reward adaptation in ways that are not purely worst-case-regret behavior.

### Why these sequences are appropriate

The i.i.d. and Massart families test whether SMART avoids unnecessary switching under standard online classification noise models. The anti-leader family is a controlled worst-case construction for FTL instability. The benign-to-hard suffix family is the closest match to SMART's single-switch design: optimism should be useful early, then the robust suffix policy should cap damage. The drift diagnostic links the experiment to practical nonstationarity, but should not be the sole evidence because static comparator regret is not the same as dynamic tracking quality.

### Acceptance criteria

A successful primary figure should show regret versus horizon on fresh sequences of exactly that horizon:

1. Benign: `SMART` and true `FTL` curves are visually indistinguishable or within a small constant gap; `FTRL` is higher.
2. Hard: true `FTL` grows substantially faster than `FTRL`; `SMART` switches with high probability and remains close to `FTRL`.
3. Benign-to-hard: `SMART` is below both fixed baselines or near the lower envelope: it keeps the low-loss FTL prefix and avoids most of the FTL hard-suffix blow-up.
4. Diagnostics: $\Sigma_t$ is monotone, matches final FTL regret at $t=T$, and crosses the threshold near the designed hardening point.
5. Calibration: a threshold-scale sweep includes the theory scale and shows the cost of switching too early or too late.

### Required implementation corrections

For this loss/domain, true FTL and the comparator should be computed from the linear state $M_t=\sum_{i\le t}y_i z_i$, not from a played-point subgradient convention. SMART should switch causally after observing the round that crosses threshold, and FTRL should reset both its cumulative state and its time index on the suffix.

## Formal setup

At round $t$, the learner chooses $x_t$ in the unit Euclidean ball:

$$
X = \{x \in \mathbb{R}^d : \|x\|_2 \le 1\}
$$

Then it observes $(z_t, y_t)$ with label $y_t \in \{-1, +1\}$ and incurs loss:

$$
\ell_t(x) = \tfrac{1}{2}\,|\langle z_t, x\rangle - y_t|
$$

Reported metric is regret to the best fixed action on the prefix:

$$
\mathrm{Reg}_A(n) = \sum_{t=1}^n \ell_t(x_t^A) - \min_{x \in X}\sum_{t=1}^n \ell_t(x)
$$

The robust baseline is concrete FTRL with quadratic regularization and $\sqrt{t}$ scaling:

$$
x_t^{\mathrm{FTRL}} = \Pi_X\!\left(-\frac{\eta_0}{\sqrt{t}}\,\theta_{t-1}\right),\quad \eta_0 = \sqrt{2}.
$$

This is the correct robust comparator family in this implementation because updates are linearized in vector space and actions are constrained to an $\ell_2$ ball.

## Why leader-path synthesis is used

### The computational issue

In this setting, two quantities repeatedly require solving prefix-level optimization structure:

1. The comparator term $\min_x \sum_{i\le t} \ell_i(x)$ (for every $t$, horizon, run, regime).
2. SMART's online switch statistic $\Sigma_t$, which also depends on prefix best-action losses.

Even with the current optimized evaluation code, these prefix recomputations are quadratic in horizon ($O(T^2 d)$ work per run for comparator-like quantities). If we additionally generated arbitrary sequences and tried to recover exact leader behavior by generic prefix optimization each round, cost would increase further and quickly dominate the horizon sweep.

With 3 regimes, 48 runs, and 40 horizons ($25..1000$ step $25$), brute-force prefix optimization on arbitrary synthetic sequences becomes a practical bottleneck.

### The design response

Instead of sampling $(z_t, y_t)$ first and solving for leaders afterward, this experiment does the reverse:

1. Construct a target leader-direction path over time.
2. Construct a realizable cumulative gradient path $\theta_t$ with bounded increments.
3. Convert each increment $\Delta_t = \theta_t - \theta_{t-1}$ into a feature-label pair $(z_t, y_t)$ that realizes that update.

This makes the generated sequence explicitly compatible with the intended leader dynamics and avoids an additional expensive outer optimization loop during sequence construction.

## Exact data-generating process (DGP)

Default dimension is $d=5$.

For each regime, synthesis builds:

1. A target direction path on the unit sphere (`target_leaders`).
2. A latent separator path (`w_star_path`) used to orient labels plausibly.
3. A radius schedule $r_t$ controlling cumulative-gradient magnitude.
4. A realizable $\theta_t$ path with per-step cap $\|\Delta_t\| \le \texttt{max\_delta\_norm}$.
5. A mismatch schedule that controls deterministic label flips over time.

### Common construction details

- $\theta_t^{\text{des}} = -r_t u_t$ where $u_t$ is target direction.
- Realizability cap: if $\|\Delta_t\| > \texttt{max\_delta\_norm}$, scale $\Delta_t$ down to the cap.
- Example realization uses $z_t = \pm 2\Delta_t$ and $y_t \in \{-1,+1\}$ so induced gradient matches $\Delta_t$ under the chosen loss.
- Label mismatch is deterministic-rate controlled (accumulator-based flip mask), not i.i.d. Bernoulli, to reduce unnecessary Monte Carlo jaggedness.

### Regime A: `stable_benign`

Purpose: optimism safety.

- Direction path stays near a fixed anchor.
- Radius grows smoothly: $r_t \approx 0.10 + 0.020\sqrt{t+1}$ with small noise.
- Very low mismatch (0.00 in fixed illustrative mode).

Fixed illustrative parameters used for the main plot:

- `max_delta_norm=0.35`
- `label_mismatch_prob=0.00`
- `threshold_scale=0.015`
- `direction_noise_scale=0.65`

### Regime B: `persistent_shift`

Purpose: sustained shift where pure FTL should fail.

- Early segment near anchor $v_1$, then smooth transition toward $v_2$.
- Direction shift window in synthesis path: roughly rounds $140..900$.
- Radius shrinks post-shift (down to ~60% of pre-shift trend) to reduce leader stability.
- Mismatch rate increases during the hard window (base + boosted schedule).

Fixed illustrative parameters:

- `max_delta_norm=0.45`
- `label_mismatch_prob=0.05`
- `threshold_scale=0.006`
- `direction_noise_scale=1.20`

### Regime C: `delayed_hardening`

Purpose: representative mixed nonstationarity.

- Early benign phase.
- Mid-phase gradual drift.
- Later transition toward a new anchor ($v_3$) with higher mismatch.
- Hardening boundaries are delayed relative to Regime B.

Fixed illustrative parameters:

- `max_delta_norm=0.45`
- `label_mismatch_prob=0.12`
- `threshold_scale=0.005`
- `direction_noise_scale=0.50`

## SMART switch definition used here

SMART tracks an online lower-bound-style FTL regret statistic $\Sigma_t$ and switches once when:

$$
\Sigma_t \ge \text{threshold}, \qquad \text{threshold} = \texttt{threshold\_scale}\cdot\sqrt{2T}.
$$

After switching, SMART uses the robust FTRL policy for the remainder of the horizon.

## Evaluation protocol

Primary figure contract (paper-facing):

- x-axis: Horizon
- y-axis: regret at that horizon
- each horizon is a fresh sequence of exactly that length
- horizons: $25, 50, \ldots, 1000$
- runs per horizon: 48
- seed base: 7

Uncertainty bands are $\text{mean} \pm 1.96\,\text{SE}$.

## Results from current default run

Latest default run (`python run_experiment.py --engine auto`) produced:

- `stable_benign`: `FTL=0.208`, `FTRL=0.710`, `SMART=0.208`, `switch_mean=1001.0`
- `persistent_shift`: `FTL=0.556`, `FTRL=0.534`, `SMART=0.342`, `switch_mean=376.2`
- `delayed_hardening`: `FTL=0.231`, `FTRL=0.432`, `SMART=0.255`, `switch_mean=907.3`

Interpretation:

1. Stable benign: SMART is effectively identical to FTL, which is the intended no-tax behavior.
2. Persistent shift: SMART clearly improves over both pure baselines in this constructed mixed regime by combining favorable prefix/suffix behavior.
3. Delayed hardening: SMART is slightly worse than FTL but substantially better than robust FTRL, consistent with a pragmatic middle policy in gradual nonstationarity.

## What this experiment supports for the paper

This experiment supports the claim that SMART can be a useful single-switch compromise between optimism and robustness:

- It does not pay unnecessary cost when optimism is reliable.
- It mitigates sustained regime-shift damage.
- It remains interpretable through explicit switch timing.

## Limits and non-claims

- Sequences are synthetic and mechanism-oriented; they are not a claim of real-world prevalence.
- Results depend on the specific illustrative regime construction and threshold calibration.
- This is evidence for the adaptation mechanism, not a universal dominance claim over all nonstationary classification streams.

## Outputs and code map

- Main figure: `figures/fig_exp04_olc_regret_by_horizon.png`
- Generated output: `outputs/figures/exp04_olc_regret_by_horizon.png`
- Sequence synthesis: `src/synthesis.py`
- Evaluation logic (FTL/FTRL/SMART, regrets, switch signal): `src/eval.py`
- Experiment runner and plotting: `run_experiment.py`
