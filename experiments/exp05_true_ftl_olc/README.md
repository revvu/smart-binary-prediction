# Experiment 05: True-FTL Online Linear Classification

## Why this experiment exists

This experiment is the replacement candidate for the older online linear classification studies in experiments 2 and 4.

The purpose is to demonstrate SMART in a vector-valued online linear classification setting where:

1. true FTL is computed exactly,
2. the robust FTRL baseline is specified concretely,
3. the sequence families expose benign, hard, and mixed regimes,
4. SMART's switch is explained by the exact adapted regret trace $\Sigma_t$.

The paper-facing claim is:

SMART preserves FTL's advantage on benign streams, switches to protect against sustained hard regimes, and gives an interpretable single-switch compromise on streams with an easy prefix followed by a hard suffix.

## What was wrong in experiments 2 and 4

### Experiment 2: subgradient-state FTL produced invalid negative regret

Experiment 2 used a "fast path" FTL implementation (`fast_algorithms.py`) that tracked FTL state via subgradient updates at the played prediction. On stochastic streams this worked adequately, but on deterministic norm-one streams it failed.

The surrogate loss $\ell_t(x)=\frac12|{\langle z_t,x\rangle-y_t}|$ is globally linear on the unit ball whenever $\|z_t\|\le 1$, so its subgradient is unique almost everywhere. But at the boundary, when $\langle z_t,x\rangle = y_t$ exactly, the subgradient is not unique. The fast implementation chose a subgradient at the *played* prediction, and on deterministic sequences like "Switching leaders" (fixed $z_t=e_1$, labels in alternating blocks of 20) this tie convention caused the internal state to diverge from the true linear-loss FTL path. The comparator was then computed from the wrong accumulated state, producing **negative regret**—a mathematical impossibility for FTL that signaled a bug, not a genuine result.

The README for experiment 2 acknowledged this (lines 127–143) and noted that a redesign should replace the subgradient-state tracker with the closed-form $M_t$-based computation. This experiment is that redesign.

### Experiment 4: worked around a bottleneck that did not actually exist

Experiment 4 was built on the premise that computing true FTL and the prefix comparator on arbitrary sequences required solving $T$ optimization problems per run—$O(T^2 d)$ work per trial—making brute-force evaluation impractical at scale (3 regimes $\times$ 48 runs $\times$ 40 horizons).

To avoid this cost, experiment 4 reversed the sequence construction: instead of generating $(z_t,y_t)$ and solving for FTL afterward, it designed a target leader-direction path first, manufactured a realizable cumulative-gradient path with bounded increments, and then converted each increment into a feature-label pair that realized the desired FTL trajectory. This "leader-path synthesis" approach used several engineering tricks: deterministic mismatch schedules (accumulator-based flip masks instead of i.i.d. Bernoulli), latent separator paths for label plausibility, smoothstep phase boundaries, and per-regime hand-tuned parameters.

**The premise was wrong.** Because the loss is globally linear on the unit ball, both true FTL and the comparator are available in closed form from the moment $M_t=\sum_{i\le t}y_i z_i$:

- FTL action: $x_t^{\mathrm{FTL}}=M_{t-1}/\|M_{t-1}\|_2$ (one normalization, $O(d)$)
- Comparator loss: $\min_{x\in X}\sum_{i\le t}\ell_i(x)=\frac{t}{2}-\frac12\|M_t\|_2$ (one norm, $O(d)$)

A full run is $O(Td)$, not $O(T^2 d)$. There is no prefix optimization bottleneck. The synthesis machinery, the hand-tuned regime parameters, and the reverse-construction approach were all unnecessary. Experiment 4's README acknowledged this partially in its "redesign gate" section, noting that "the next revision should keep the useful part of this experiment—closed-form true FTL—but remove the hand-tuned leader-path narrative as the primary evidence."

### What experiment 5 does differently

This experiment computes true FTL and the prefix comparator directly from $M_t$ using the closed forms above. No convex solver, no subgradient proxy, no synthesis tricks. Each round costs $O(d)$ and a full trial costs $O(Td)$. The $\Sigma_t$ trace is guaranteed monotone by construction, and the `assert_trace_invariants` check verifies non-negative regret on every run.

The `switching_leaders` sequence from experiment 2—the specific sequence that produced invalid negative regret—is included as a scenario in this experiment. Under exact FTL it is benign: FTL handles the block structure correctly and regret stays small and positive throughout.

## Math derivation: why true FTL is computable here

The online linear classification problem in this experiment looks nonlinear because the loss is written with an absolute value:

$$
\ell_t(x)=\frac12|\langle z_t,x\rangle-y_t|.
$$

However, the experiment always uses $x\in X=\{x:\|x\|_2\le 1\}$, $\|z_t\|_2\le \rho\le 1$, and $y_t\in\{-1,+1\}$. Therefore $|\langle z_t,x\rangle|\le 1$. If $y_t=+1$, then $|\langle z_t,x\rangle-1|=1-\langle z_t,x\rangle$; if $y_t=-1$, then $|\langle z_t,x\rangle+1|=1+\langle z_t,x\rangle$. In both cases,

$$
\ell_t(x)=\frac12(1-y_t\langle z_t,x\rangle).
$$

Thus the loss is exactly a constant plus a linear loss over the whole feasible set, not an approximation. Define the signed feature vector $s_t=y_tz_t$ and cumulative signed feature sum

$$
M_t=\sum_{i=1}^t s_i=\sum_{i=1}^t y_i z_i.
$$

The cumulative loss of any fixed action $x$ on a prefix is

$$
L_t(x)=\sum_{i=1}^t\ell_i(x)
=\frac{t}{2}-\frac12\langle M_t,x\rangle.
$$

The best fixed comparator solves

$$
\min_{\|x\|_2\le 1} L_t(x)
=
\frac{t}{2}
-\frac12
\max_{\|x\|_2\le 1}\langle M_t,x\rangle.
$$

By Cauchy-Schwarz, $\max_{\|x\|_2\le 1}\langle M_t,x\rangle=\|M_t\|_2$, so

$$
\min_{\|x\|_2\le 1} L_t(x)
=
\frac{t}{2}-\frac12\|M_t\|_2.
$$

The prefix leader after observing $t$ rounds is therefore

$$
x_t^*
=
\begin{cases}
M_t/\|M_t\|_2, & M_t\ne 0,\\
0, & M_t=0.
\end{cases}
$$

FTL plays the previous prefix leader, so at round $t$:

$$
x_t^{\mathrm{FTL}}
=
\begin{cases}
M_{t-1}/\|M_{t-1}\|_2, & M_{t-1}\ne 0,\\
0, & M_{t-1}=0.
\end{cases}
$$

This is the main distinction from experiments 2 and 4. True FTL and the comparator require only the running vector $M_t$, one norm, and one normalization per round. The evaluation cost is $O(Td)$ per trial. No convex solver, prefix optimization loop, subgradient convention, or synthesized leader path is needed.

SMART's switch statistic is the exact FTL regret trace:

$$
\Sigma_t
=
\sum_{i=1}^t \ell_i(x_i^{\mathrm{FTL}})
-
\left(
\frac{t}{2}-\frac12\|M_t\|_2
\right).
$$

This quantity is adapted because it is known after observing round $t$, monotone by the FTL be-the-leader decomposition, and satisfies $\Sigma_T=\mathrm{Reg}_T(\mathrm{FTL})$. That is why experiment 5 is a faithful SMART implementation: the stopping rule uses the true online-computable FTL regret trace, not a proxy.

## Design gate

### SMART behavior claim

The experiment must show:

1. **Preserve optimism:** on benign predictable streams, SMART should remain visually close to true FTL and avoid FTRL's robustness tax.
2. **Protect in hard regimes:** on high-variation or adversarial streams, SMART should switch and avoid FTL's large regret growth.
3. **Interpretable switch:** in mixed streams, the switch should occur when $\Sigma_t$ crosses the calibrated threshold near the onset of sustained deterioration.

### Sequence families

The main regret-by-horizon graph uses these primary sequence families:

1. `iid_separable_margin`: i.i.d. bounded-margin separable stream.
2. `massart_10`: the same stream with independent 10% label flips.
3. `alternating_antileader`: fixed feature direction with alternating labels, used as an illustrative FTL failure mode.
4. `benign_to_hard_suffix`: separable prefix followed by an adaptive anti-leader suffix, the main SMART single-switch regime.
5. `separator_drift`: gradual rotation of the latent separator, included as a nonstationary diagnostic rather than the central proof example.

The `switching_leaders` sequence is reserved for the switch diagnostics plot. It is the fixed-direction, block-label sequence that produced invalid negative regret in experiment 2. It is not included in the horizon-sweep regret grid because its deterministic block structure creates a sawtooth artifact at coarse horizon spacing; the diagnostic plot is the right place to show that exact FTL gives non-negative regret and no spurious SMART switch.

### Why these sequences are appropriate

`iid_separable_margin` samples a unit separator $u$, random labels $y_t\in\{-1,+1\}$, and bounded features whose signed versions $y_tz_t$ align with $u$ up to orthogonal noise:

$$
z_t=\rho y_t\operatorname{unit}(m u+\xi_t),
$$

where $\xi_t$ is orthogonal noise and the default margin parameter is positive. This is a representative benign stream: FTL should identify the stable direction quickly, and SMART should not pay a robustness tax.

`massart_10` uses the same margin model but flips each label independently with probability $0.10$. This is a representative mild-noise stream: optimism should still be useful, but the result checks that SMART is not hypersensitive to small stochastic corruption.

`alternating_antileader` fixes $z_t=\rho e_1$ and alternates labels $+1,-1,+1,-1,\ldots$. This is illustrative rather than representative. It forces $M_t$ to repeatedly cancel or reverse, so FTL follows stale leaders and accumulates large regret. Its role is to verify robust protection in a controlled FTL failure mode.

`benign_to_hard_suffix` uses a separable margin prefix for the first $45\%$ of the horizon, then switches to an adaptive anti-leader suffix. In the suffix, the generator points the next feature along the current FTL direction and assigns the opposite label:

$$
z_t=\rho\,\operatorname{unit}(M_{t-1}),\qquad y_t=-1.
$$

Equivalently, the signed update $y_tz_t$ pushes against the current leader. This is the central SMART sequence: the prefix rewards optimism, the suffix creates sustained trace growth, and a one-switch policy is structurally appropriate.

`separator_drift` gradually rotates the latent separator from one direction to another over the middle of the horizon. This is a practical nonstationarity diagnostic. It should not be read as a dynamic-regret benchmark because the reported metric is still static regret against the best fixed action.

`switching_leaders` fixes $z_t=\rho e_1$ and uses label blocks of length 20, alternating sign by block. Its basis is diagnostic rather than paper-primary: it is the exact style of deterministic sequence that broke the old subgradient-state implementation. Under exact FTL it confirms that the previous negative-regret behavior was an implementation artifact, not a property of the OLC problem.

### Acceptance criteria

A successful run should show:

1. In `iid_separable_margin`, SMART and FTL have nearly identical regret and FTRL is higher.
2. In `alternating_antileader`, FTL grows faster than FTRL and SMART switches.
3. In `benign_to_hard_suffix`, SMART is near or below the lower envelope of the fixed baselines by combining FTL's prefix and FTRL's suffix.
4. The diagnostic plot shows monotone $\Sigma_t$ and switch timing that is explainable from the threshold crossing.
5. In `switching_leaders`, the diagnostic trace confirms FTL regret is strictly positive at all rounds (resolving experiment 2's negative-regret bug) and $\Sigma_t$ stays below threshold (no switch).
6. The threshold sweep shows the cost of switching too early or too late.

## Formal setup

At each round, the learner chooses $x_t$ in the unit Euclidean ball:

$$
X = \{x\in\mathbb{R}^d:\|x\|_2\le 1\}.
$$

It observes $(z_t,y_t)$ with $\|z_t\|_2\le \rho\le 1$ and $y_t\in\{-1,+1\}$, and incurs:

$$
\ell_t(x)=\frac12|\langle z_t,x\rangle-y_t|.
$$

Because $|\langle z_t,x\rangle|\le 1$, this equals the linear loss:

$$
\ell_t(x)=\frac12(1-y_t\langle z_t,x\rangle).
$$

Define the signed feature sum:

$$
M_t=\sum_{i=1}^t y_i z_i.
$$

The best fixed comparator loss on a prefix is:

$$
\min_{x\in X}\sum_{i=1}^t \ell_i(x)=\frac{t}{2}-\frac12\|M_t\|_2.
$$

Regret is measured against this best fixed comparator at each horizon.

Because the comparator is fixed while the algorithms are adaptive, FTRL or SMART can have negative static regret on some nonstationary synthetic streams. This is not a bug; it means the adaptive policy outperformed the best fixed action on that sequence.

## Algorithms compared

### True FTL

FTL is exact and closed-form:

$$
x_t^{\mathrm{FTL}}=
\begin{cases}
M_{t-1}/\|M_{t-1}\|_2, & M_{t-1}\ne 0,\\
0, & M_{t-1}=0.
\end{cases}
$$

No convex solver or subgradient proxy is used.

### Robust FTRL baseline

The robust baseline is quadratic FTRL on the linear loss vectors $c_t=-\frac12 y_t z_t$:

$$
x_t^{\mathrm{FTRL}}
=\arg\min_{\|x\|_2\le 1}
\left\langle \sum_{i<t} c_i,x\right\rangle
+\frac{\sqrt{t}}{2\eta_0}\|x\|_2^2,
\qquad \eta_0=\sqrt{2}.
$$

Equivalently:

$$
x_t^{\mathrm{FTRL}}
=
\Pi_X\left(\frac{\eta_0}{2\sqrt{t}}M_{t-1}\right).
$$

This is the appropriate robust baseline because the loss is linear over a bounded Euclidean domain, and quadratic FTRL is the standard $O(\sqrt{T})$ no-assumption fallback for this geometry.

### SMART

SMART starts with true FTL and computes:

$$
\Sigma_t
=
\sum_{i=1}^t \ell_i(x_i^{\mathrm{FTL}})
-\min_{x\in X}\sum_{i=1}^t\ell_i(x).
$$

For this exact setting, $\Sigma_t$ is monotone and $\Sigma_T=\mathrm{Reg}_T(\mathrm{FTL})$.

SMART switches after observing the first round where:

$$
\Sigma_t\ge \theta.
$$

The default theory threshold is $\theta=\sqrt{2T}$. The calibrated SMART variant uses $\theta=g_{\mathrm{emp}}(T)$, where $g_{\mathrm{emp}}(T)$ is the largest observed FTRL regret across designated hard calibration streams.

After switching, FTRL is reset and run on the suffix only.

## Evaluation protocol

Default run:

```bash
python experiments/exp05_true_ftl_olc/run_experiment.py
```

Quick smoke run:

```bash
python experiments/exp05_true_ftl_olc/run_experiment.py --quick
```

Invariant checks:

```bash
python experiments/exp05_true_ftl_olc/run_experiment.py --self-test
```

Default dashboard configuration:

- primary dimension: $d=20$
- feature radius: $\rho=0.8$
- horizons: $T=100,200,\ldots,1000$
- trials per horizon: 24
- empirical-threshold calibration trials: 8
- deterministic seed: 7
- dimension diagnostic: $d\in\{5,20,100\}$

Heavier paper-profile run:

```bash
python experiments/exp05_true_ftl_olc/run_experiment.py --paper-profile
```

The paper profile uses horizons through $T=2000$ with 64 trials per horizon. It is intentionally not the default because dashboard regeneration should stay practical during iteration.

Each horizon uses fresh sequences of exactly that length. Plots report mean regret with 95% confidence bands where applicable.

## Outputs and figure provenance

Generated outputs:

- `outputs/figures/exp05_olc_regret_by_horizon.png`
- `outputs/figures/exp05_olc_empirical_threshold.png`
- `outputs/figures/exp05_olc_switch_diagnostics.png`
- `outputs/figures/exp05_olc_threshold_calibration.png`
- `outputs/figures/exp05_olc_dimension_sweep.png`
- `outputs/summary_regret.csv`
- `outputs/empirical_g.csv`

Curated dashboard figures:

- `figures/fig_exp05_olc_regret_by_horizon.png`
- `figures/fig_exp05_olc_empirical_threshold.png`
- `figures/fig_exp05_olc_switch_diagnostics.png`
- `figures/fig_exp05_olc_threshold_calibration.png`
- `figures/fig_exp05_olc_dimension_sweep.png`
- `figures/INDEX.md`

The dashboard can be regenerated from the repository root with:

```bash
python experiments/dashboard/generate_dashboard.py
```

## Interpretation

This experiment is designed to support a mechanism claim, not a dataset benchmark claim. The strongest paper-facing evidence is the joint behavior of:

1. regret-by-horizon curves,
2. switch diagnostics from $\Sigma_t$,
3. threshold calibration sensitivity,
4. dimension robustness.

The expected result is not universal dominance by SMART. The intended story is that SMART tracks FTL when FTL is safe, switches when the exact trace indicates FTL has become risky, and provides a clean compromise between optimism and robustness in mixed regimes.

Latest default run produced the following mean final regrets at $T=1000$:

- `iid_separable_margin`: `FTL=5.861`, `FTRL=6.603`, `SMART-theory=5.861`, `SMART-calibrated=5.861`.
- `massart_10`: `FTL=6.618`, `FTRL=7.527`, `SMART-theory=6.618`, `SMART-calibrated=6.618`.
- `alternating_antileader`: `FTL=200.000`, `FTRL=6.925`, `SMART-theory=50.474`, `SMART-calibrated=20.270`.
- `benign_to_hard_suffix`: `FTL=178.119`, `FTRL=-1.878`, `SMART-theory=48.900`, `SMART-calibrated=18.256`.
- `separator_drift`: `FTL=5.459`, `FTRL=6.123`, `SMART-theory=5.459`, `SMART-calibrated=5.459`.

The main interpretation is:

1. In benign and mild-noise regimes, SMART is indistinguishable from true FTL and improves over robust FTRL.
2. In hard anti-leader regimes, SMART substantially reduces FTL's blow-up; FTRL remains the strongest pure robust baseline.
3. In the mixed benign-to-hard regime, SMART switches after the hard suffix begins and sharply reduces FTL regret, while FTRL can achieve negative static regret because adaptive policies may beat the best fixed comparator on this synthetic nonstationary stream.
4. The `switching_leaders` sequence—which produced invalid negative regret in experiment 2—is shown in the switch diagnostics plot rather than the horizon sweep (its deterministic block structure creates sawtooth artifacts at coarse horizon spacing). The per-round diagnostic trace confirms that FTL regret is strictly positive at every round, $\Sigma_t$ is monotone and stays below threshold, and SMART does not switch. The negative regret observed in experiment 2 was entirely an artifact of the subgradient-state divergence, not a property of the sequence.

## Limits and non-claims

- The streams are synthetic and mechanism-oriented.
- The anti-leader stream is illustrative rather than representative.
- The separator drift diagnostic is not a dynamic-regret benchmark.
- Empirical threshold calibration is itself a modeling choice and should be reported alongside the regret results.

## Code map

- Exact algorithms and invariants: `src/olc_exact.py`
- Sequence families: `src/sequences.py`
- Figure generation and dashboard curation: `run_experiment.py`
