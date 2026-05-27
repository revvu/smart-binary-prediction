# Experiment 05: True-FTL Online Linear Classification

## Objective

This experiment is the paper-facing replacement for the older online linear classification studies in experiments 2 and 4. It demonstrates SMART in a vector-valued online linear classification setting where:

1. FTL is computed exactly, not approximated by played-point subgradients.
2. The robust baseline is a concrete quadratic-FTRL algorithm with a standard online-convex-optimization interpretation.
3. The SMART switch statistic is the exact adapted FTL regret trace $\Sigma_t$.
4. The sequence families are bounded OLC feature-label streams designed to show clean benign, weak-signal, delayed-signal, corrupted, and classical leader-instability regimes.

The paper-facing claim is:

SMART preserves FTL's advantage on benign streams, switches when the exact FTL regret trace indicates sustained danger, and gives an interpretable one-switch compromise between optimism and worst-case robustness.

## Formal Setting

At each round $t$, the learner chooses a vector in the Euclidean unit ball:

$$
X=\{x\in\mathbb{R}^d:\|x\|_2\le 1\}.
$$

The learner then observes a bounded feature-label pair $(z_t,y_t)$ with

$$
\|z_t\|_2\le \rho\le 1,\qquad y_t\in\{-1,+1\},
$$

and incurs the online linear classification surrogate

$$
\ell_t(x)=\frac12|\langle z_t,x\rangle-y_t|.
$$

Because $\|x\|_2\le1$ and $\|z_t\|_2\le\rho\le1$, we have $|\langle z_t,x\rangle|\le1$. Therefore the absolute value collapses to a linear loss on the entire feasible set:

$$
\ell_t(x)=\frac12(1-y_t\langle z_t,x\rangle).
$$

Define the signed feature vector and cumulative signed feature sum:

$$
s_t=y_tz_t,\qquad M_t=\sum_{i=1}^t s_i=\sum_{i=1}^t y_i z_i.
$$

For any fixed comparator $x\in X$, cumulative loss on a prefix is

$$
L_t(x)=\sum_{i=1}^t \ell_i(x)
=\frac{t}{2}-\frac12\langle M_t,x\rangle.
$$

The best fixed comparator loss is therefore

$$
\min_{\|x\|_2\le1}L_t(x)
=
\frac{t}{2}
-\frac12
\max_{\|x\|_2\le1}\langle M_t,x\rangle.
$$

By Cauchy-Schwarz,

$$
\max_{\|x\|_2\le1}\langle M_t,x\rangle=\|M_t\|_2,
$$

so the exact prefix comparator loss is

$$
L_t^*=\min_{x\in X}L_t(x)=\frac{t}{2}-\frac12\|M_t\|_2.
$$

This identity is the computational hinge of the experiment. FTL, the comparator, and the SMART trace all reduce to maintaining $M_t$, computing one norm, and normalizing one vector. The cost is $O(Td)$ per trial.

Regret is always static regret against the best fixed action in hindsight:

$$
\mathrm{Reg}_T(A)=\sum_{t=1}^T\ell_t(x_t^A)-L_T^*.
$$

Adaptive algorithms such as FTRL or SMART can have negative static regret on some nonstationary synthetic streams. That is not a bug; it means the adaptive policy beat the best fixed comparator on that realized sequence.

### Diagnostic note: why FTRL can have negative regret

The dimension-robustness corruption check can show negative regret for FTRL. This is negative **static regret**, not negative loss. The regret is

$$
\mathrm{Reg}_T(\mathrm{FTRL})=L_T(\mathrm{FTRL})-L_T^*,
$$

where $L_T^*$ is the loss of the best single fixed vector over the whole horizon. In `strategic_corruption_suffix`, the reliable prefix, erosion phase, and alternating suffix can make the final signed sum $M_T$ small. Then the best fixed comparator is weak and has loss close to $T/2$. FTRL is adaptive: it can use the early signal, become cautious as the signal cancels, and end with cumulative loss slightly below the best fixed comparator. That produces a small negative static regret. This is allowed by the theory because worst-case regret bounds upper-bound how much worse FTRL can be than the comparator; they do not prevent FTRL from outperforming the best fixed action on a nonstationary sequence.

## Algorithms

### Exact FTL

The prefix leader after observing $t$ rounds is

$$
x_t^*
=
\begin{cases}
M_t/\|M_t\|_2, & M_t\ne0,\\
0, & M_t=0.
\end{cases}
$$

FTL plays the previous prefix leader:

$$
x_t^{\mathrm{FTL}}
=
\begin{cases}
M_{t-1}/\|M_{t-1}\|_2, & M_{t-1}\ne0,\\
0, & M_{t-1}=0.
\end{cases}
$$

This is implemented directly in `src/olc_exact.py`. No convex solver, generic prefix optimizer, or subgradient proxy is used.

### Robust FTRL Baseline

The robust baseline is Follow-the-Regularized-Leader on the exact linear losses

$$
c_t=-\frac12 y_tz_t,\qquad \ell_t(x)=\frac12+\langle c_t,x\rangle.
$$

With the time-varying quadratic regularizer

$$
\psi_t(x)=\frac{\sqrt{t}}{2\eta_0}\|x\|_2^2+\iota_X(x),
\qquad \eta_0=\sqrt{2},
$$

where $\iota_X$ is the indicator of the unit ball, FTRL is

$$
x_t^{\mathrm{FTRL}}
=
\arg\min_{x\in X}
\left\langle\sum_{i<t}c_i,x\right\rangle
+\frac{\sqrt{t}}{2\eta_0}\|x\|_2^2.
$$

Since $\sum_{i<t}c_i=-\frac12M_{t-1}$, the unconstrained optimum solves

$$
-\frac12M_{t-1}+\frac{\sqrt{t}}{\eta_0}x=0,
$$

so

$$
\tilde{x}_t=\frac{\eta_0}{2\sqrt{t}}M_{t-1}.
$$

The feasible set is the Euclidean unit ball, hence the implemented update is

$$
x_t^{\mathrm{FTRL}}
=
\Pi_X\left(\frac{\eta_0}{2\sqrt{t}}M_{t-1}\right).
$$

This is the correct robust baseline for this experiment because the losses are bounded linear losses on a bounded Euclidean domain, where quadratic FTRL is the standard $O(\sqrt{T})$ no-assumption fallback.

This matches the FTRL template in Orabona's Chapter 7:

$$
x_t\in\arg\min_{x\in V}\psi_t(x)+\sum_{i<t}\langle g_i,x\rangle,
$$

with $V=X$, $g_i=c_i$, and the increasing Euclidean quadratic regularizer above. It also matches Orabona's Chapter 8 online-linear-classification surrogate once the bounded-domain condition $|\langle z_t,x\rangle|\le1$ is enforced. The broader online-convex-optimization framing follows Zinkevich's static-regret setting, and the FTRL interpretation is standard in the McMahan FTRL/mirror-descent literature.

### SMART

SMART starts as exact FTL and computes the exact adapted FTL regret trace:

$$
\Sigma_t
=
\sum_{i=1}^t\ell_i(x_i^{\mathrm{FTL}})
-L_t^*
=
\sum_{i=1}^t\ell_i(x_i^{\mathrm{FTL}})
-\left(\frac{t}{2}-\frac12\|M_t\|_2\right).
$$

In this setting, $\Sigma_t$ is online-computable after round $t$, monotone nondecreasing by the FTL be-the-leader decomposition, and satisfies

$$
\Sigma_T=\mathrm{Reg}_T(\mathrm{FTL}).
$$

SMART switches after observing the first round where

$$
\Sigma_t\ge \theta.
$$

The theory threshold is $\theta=\sqrt{2T}$. The calibrated variant uses

$$
\theta=g_{\mathrm{emp}}(T),
$$

where $g_{\mathrm{emp}}(T)$ is the largest observed FTRL regret across designated hard calibration streams. After switching, FTRL is reset and run only on the suffix. The reset is intentional: it matches the SMART proof decomposition into an FTL prefix and a robust-policy suffix.

## What Experiments 2 and 4 Got Wrong

### Experiment 2: Wrong State for FTL

Experiment 2 treated the absolute-value objective as a generic prefix ERM problem and tracked a fast FTL state using played-point subgradients:

$$
\theta_t=\sum_{i\le t}g_i z_i,\qquad
g_i\in\partial_q \frac12|q-y_i|.
$$

At exact-fit boundary points $q=y_i$, the implementation used $g_i=0$. That is a valid subgradient choice for a linearized algorithm, but it is not the cumulative state of the restricted-domain OLC loss. The true state is always

$$
M_t=\sum_{i\le t}y_i z_i.
$$

On deterministic boundary-heavy streams such as `switching_leaders`, the old subgradient-state path dropped informative rounds. The resulting path was not true FTL on the original loss. It could even report negative FTL regret, which is mathematically impossible for exact FTL and was therefore a bug.

Experiment 2's CVXPY path reinforced the wrong conclusion that true FTL was computationally expensive. CVXPY was solving the generic absolute-loss problem for every prefix, but in this bounded OLC setting the same optimizer is available in closed form from $M_t$.

### Experiment 4: Solving a Nonexistent Bottleneck

Experiment 4 assumed that computing true FTL and the prefix comparator required solving $T$ optimization problems per run, roughly $O(T^2d)$. To avoid that cost, it reversed the construction: it designed a target leader path first, manufactured bounded increments that realized that path, and then converted those increments into feature-label pairs.

That machinery was unnecessary. The exact comparator and exact FTL action are available from

$$
M_t=\sum_{i\le t}y_i z_i,
\qquad
x_t^{\mathrm{FTL}}=\frac{M_{t-1}}{\|M_{t-1}\|_2},
\qquad
L_t^*=\frac{t}{2}-\frac12\|M_t\|_2.
$$

The correct complexity is $O(Td)$ per trial. Experiment 5 keeps the useful insight from experiment 4, namely closed-form true FTL, but removes the reverse-engineered leader-path synthesis as primary evidence.

### Why This Does Not Replace SVMs

The closed form applies only to this restricted regret-accounting objective:

$$
\min_{\|x\|_2\le1}\sum_{t=1}^T\frac12|\langle z_t,x\rangle-y_t|.
$$

Under $\|z_t\|\le1$ and $\|x\|\le1$, that objective collapses to a signed-average direction. SVMs and Pegasos-style methods solve different margin-regularized hinge-loss objectives, often with bias terms, kernels, and support-vector structure. Passive-Aggressive algorithms are also different: they are margin-driven online updates, not the quadratic-FTRL robust baseline used here.

## Sequence Design

### Behavior Claim

The experiment must show three SMART behaviors:

1. **Preserve optimism:** on benign predictable streams, SMART should remain close to FTL and avoid FTRL's robustness tax.
2. **Protect in hard regimes:** on high-variation or corrupted streams, SMART should switch and avoid FTL's large regret growth.
3. **Explain the switch:** in mixed streams, the switch should be readable from $\Sigma_t$ crossing the calibrated threshold after sustained deterioration begins.

### Literature Motivation

The sequence design uses application stories and individual-sequence examples from the literature.

Contextual pricing and allocation motivate the clean covariate-diverse stream and the weak-signal variant. Bastani, Bayati, and Khosravi show that greedy or mostly-greedy contextual-bandit policies can be effective when contexts have enough diversity, which is the operational analogue of a stable informative feature process. The weak-signal variant asks what happens when the same application has lower margin, lower effective covariate diversity, or noisier feedback.

Dynamic pricing and revenue-management models motivate the delayed-signal stream. Aviv and Pazgal study pricing under partially observed demand dynamics, illustrating why a sequential decision-maker may initially see weak or uninformative feedback before a stable response pattern emerges.

Online advertising and platform feedback motivate the strategic-corruption stream. Choi, Mela, Balseiro, and Leary survey online display-advertising markets, where feedback quality and strategic behavior matter; corruption-robust online learning, including Lykouris, Mirrokni, and Paes Leme, motivates stress tests with reliable prefixes and harmful suffixes.

Feder, Merhav, and Gutman, and later de Rooij, van Erven, Grunwald, and Koolen, motivate the OLC-FMG sequence. Their individual-sequence examples are designed to expose when it is safe to follow the leader and when a robust policy is needed. Here we build the vector-valued OLC analogue by controlling the signed feature sum $M_t$ through bounded feature-label pairs.

### OLC Construction Rule

All sequences must be valid OLC streams:

$$
\|z_t\|_2\le\rho,\qquad y_t\in\{-1,+1\}.
$$

The generator cannot directly specify arbitrary losses or arbitrary leader paths. It must specify feature-label pairs. The only lever that affects FTL and the comparator is the signed update

$$
y_tz_t.
$$

This is why every sequence below is described through how it shapes $M_t=\sum_{i\le t}y_i z_i$.

### Primary Sequence Families

The main regret-by-horizon figure uses five primary sequence families.

**`covariate_diverse_stationary`**

Purpose: benign, representative optimism case.

Real-world analogue: a mature contextual pricing, allocation, triage, or recommendation system where covariates are informative from the start and the relationship between features and outcomes is stable. This is the setting in which a greedy or FTL-like optimistic policy should plausibly work well.

Generation: sample a unit separator $u$, labels $y_t\sim\mathrm{Unif}\{-1,+1\}$, and orthogonal unit noise $v_t\perp u$. Then set

$$
z_t=\rho y_t\operatorname{unit}(0.72u+0.58v_t).
$$

The signed update is

$$
y_tz_t=\rho\operatorname{unit}(0.72u+0.58v_t),
$$

so $M_t$ has persistent positive drift toward $u$ with covariate variation around that direction.

Why it is appropriate: this is the OLC analogue of stable contextual structure with covariate diversity. It tests the no-unnecessary-tax claim: FTL should identify the stable direction quickly, and SMART should not switch.

**`weak_signal_low_margin`**

Purpose: benign but statistically harder optimism case.

Real-world analogue: the same contextual decision problem as above, but with weaker signal quality. Examples include sparse customer segments, low-margin treatment effects, noisier conversion labels, or feature sets that are only weakly predictive.

Generation: use the same class-conditional margin model as `covariate_diverse_stationary`, but reduce the signal margin and add light label noise:

$$
z_t=\rho y_t\operatorname{unit}(0.18u+0.98v_t),
$$

with each label independently flipped with probability $0.05$.

Why it is appropriate: this is still a stable applied setting, but the signed drift in $M_t$ is much weaker. It separates "easy benign" from "hard benign": SMART should still avoid switching, but regret should be visibly larger because FTL needs more samples to identify the stable direction.

**`delayed_signal_emergence`**

Purpose: cold-start or delayed-product-market-fit diagnostic.

Real-world analogue: a launch, new market, new advertising campaign, new clinical workflow, or newly instrumented platform where early observations are not yet aligned with the eventual stable response model. After enough deployment time, the signal becomes structured.

Generation: sample a latent separator $u$ and use a split at $0.45T$. Before the split, draw bounded covariates orthogonal to $u$ with independent random labels, so the prefix has no stable signed-feature drift toward the eventual separator. After the split, switch to the covariate-diverse margin model:

$$
z_t=\rho y_t\operatorname{unit}(0.72u+0.58v_t).
$$

Why it is appropriate: this models a launch or deployment where early feedback is uninformative but later feedback becomes structured. It is application-driven but graphically different from the stationary benign stream: FTL must recover from an unhelpful prefix, while SMART should avoid switching merely because the early data are noisy.

**`strategic_corruption_suffix`**

Purpose: paper-facing hardening sequence with an applied corruption story.

Real-world analogue: a deployed model whose early feedback is reliable, followed by a sustained period of low-quality or strategically distorted feedback. Examples include click-fraud bursts in advertising, bot traffic in recommendation systems, sensor degradation, data pipeline failure, or users strategically changing behavior after learning the policy.

Generation: sample a unit direction $u$. For the first $20\%$ of the horizon, set $z_t\approx\rho u$ and $y_t=+1$, so $y_tz_t$ builds a stable leader. For the next $20\%$, keep $z_t\approx\rho u$ but set $y_t=-1$, so $y_tz_t\approx-\rho u$ erodes the trusted signal. For the final $60\%$, keep $z_t\approx\rho u$ and alternate labels, keeping $M_t$ near the decision boundary and making FTL chase unstable leaders.

Concretely, the implementation uses

$$
z_t=\rho\operatorname{unit}(0.96u+0.04v_t),
$$

with $v_t\perp u$, and labels by phase:

$$
y_t=
\begin{cases}
+1, & t<0.20T,\\
-1, & 0.20T\le t<0.40T,\\
(-1)^{t-\lfloor0.40T\rfloor}, & t\ge0.40T.
\end{cases}
$$

Why it is appropriate: this is the OLC analogue of a reliable deployment followed by manipulation, click fraud, sensor failure, or strategic response. It is exogenous and pre-specified, but adversarially structured. It should make FTL unsafe and trigger SMART after the trusted prefix has been eroded.

**`olc_fmg_leader_gap`**

Purpose: literature bridge to individual-sequence examples.

Real-world analogue: none claimed as a calibrated application model. This is a deliberately stylized stress test that translates the classical individual-sequence "follow the leader if you can" construction into OLC feature-label form.

Generation: fix $z_t=\rho e_1$. For the first $40\%$ of the horizon, alternate labels $+1,-1,+1,-1,\ldots$ with an even prefix length. For the remaining $60\%$, use $y_t=+1$.

Why it is appropriate: this is the direct one-dimensional OLC analogue of alternating-then-stable individual sequences used to show why one should follow the leader only when the leader is stable. The alternating prefix repeatedly cancels $M_t$; the stable suffix eventually creates a reliable leader. It is illustrative and literature-facing rather than a calibrated market data model.

### Diagnostic Sequence

`switching_leaders` fixes $z_t=\rho e_1$ and uses label blocks of length 20 with alternating signs. This is included because the same style of deterministic boundary-heavy stream produced invalid negative regret in experiment 2. Under exact $M_t$-based FTL, the diagnostic trace verifies non-negative FTL regret and monotone $\Sigma_t$.

### Acceptance Criteria

A successful run should show:

1. In `covariate_diverse_stationary`, SMART and FTL have nearly identical regret and FTRL is higher.
2. In `weak_signal_low_margin`, SMART remains close to FTL but regret is visibly higher than in the clean benign case.
3. In `delayed_signal_emergence`, SMART should not switch during the uninformative prefix if FTL regret remains within the calibrated budget; the curve should show a distinct cold-start cost.
4. In `strategic_corruption_suffix`, SMART switches during the corrupted regime and materially reduces FTL's late-horizon regret.
5. In `olc_fmg_leader_gap`, SMART improves on FTL by responding to leader instability, matching the classical "follow the leader if you can" lesson.
6. The switch-diagnostic plot shows monotone $\Sigma_t$, the calibrated threshold, and interpretable switch timing.
7. `switching_leaders` confirms that experiment 2's negative regret was an implementation artifact.

## Evaluation Protocol

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

Default configuration:

- primary dimension: $d=20$
- feature radius: $\rho=0.8$
- horizons: $T=100,200,\ldots,1000$
- trials per horizon: 24
- empirical-threshold calibration trials: 8
- deterministic seed: 7
- dimension diagnostic: fixed horizon $T=10000$ and $d=1,2,\ldots,50$

Heavier paper-profile run:

```bash
python experiments/exp05_true_ftl_olc/run_experiment.py --paper-profile
```

Each horizon uses fresh sequences of exactly that length. Plots report mean regret with 95% confidence bands where applicable.

The dimension-sweep figure fixes the horizon at $T=10000$ and evaluates each primary sequence family at $d=1,2,\ldots,50$. It is a diagnostic for whether the qualitative ordering of FTL, FTRL, and calibrated SMART is stable as feature dimension changes.

## Outputs and Figure Provenance

Generated outputs:

- `outputs/figures/exp05_olc_regret_by_horizon.png`
- `outputs/figures/exp05_olc_empirical_threshold.png`
- `outputs/figures/exp05_olc_switch_diagnostics.png`
- `outputs/figures/exp05_olc_threshold_calibration.png`
- `outputs/figures/exp05_olc_dimension_sweep.png`
- `outputs/summary_regret.csv`
- `outputs/empirical_g.csv`
- `outputs/dimension_sweep.csv`

Curated dashboard figures:

- `figures/fig_exp05_olc_regret_by_horizon.png`
- `figures/fig_exp05_olc_empirical_threshold.png`
- `figures/fig_exp05_olc_switch_diagnostics.png`
- `figures/fig_exp05_olc_threshold_calibration.png`
- `figures/fig_exp05_olc_dimension_sweep.png`
- `figures/INDEX.md`

Regenerate the dashboard UI with:

```bash
python experiments/dashboard/generate_dashboard.py
```

## Latest Default Results

Latest default run produced the following mean final regrets at $T=1000$:

- `covariate_diverse_stationary`: `FTL=1.108`, `FTRL=1.500`, `SMART-theory=1.108`, `SMART-calibrated=1.108`, calibrated switch mean `1001.0` (no switch).
- `weak_signal_low_margin`: `FTL=5.572`, `FTRL=6.279`, `SMART-theory=5.572`, `SMART-calibrated=5.572`, calibrated switch mean `1001.0` (no switch).
- `delayed_signal_emergence`: `FTL=8.780`, `FTRL=10.500`, `SMART-theory=8.780`, `SMART-calibrated=8.780`, calibrated switch mean `1001.0` (no switch).
- `strategic_corruption_suffix`: `FTL=82.463`, `FTRL=-2.689`, `SMART-theory=48.387`, `SMART-calibrated=18.723`, calibrated switch mean `488.8`.
- `olc_fmg_leader_gap`: `FTL=80.400`, `FTRL=11.789`, `SMART-theory=52.739`, `SMART-calibrated=24.760`, calibrated switch mean `70.0`.

Interpretation:

1. SMART is indistinguishable from FTL in the clean, weak-signal, and delayed-signal regimes, avoiding FTRL's robustness tax even as the benign cases become progressively harder.
2. The first three panels now separate different benign mechanisms: immediate stable signal, weak low-margin signal, and cold-start signal emergence.
3. SMART sharply reduces FTL regret in the strategic-corruption suffix, switching after the reliable prefix has been eroded.
4. SMART improves substantially on FTL in the OLC-FMG leader-gap sequence, matching the classical individual-sequence intuition.
5. FTRL can have negative static regret in the corruption sequence because an adaptive policy can outperform the best fixed comparator on that nonstationary synthetic stream.

## Limits and Non-Claims

- The streams are synthetic and mechanism-oriented.
- The delayed-signal sequence is not a dynamic-regret benchmark; all reported regrets are static regret against the best fixed comparator.
- The strategic-corruption sequence is exogenous and pre-specified, but adversarially structured. It should be presented as a stress test, not as a calibrated empirical data model.
- The OLC-FMG sequence is illustrative and literature-facing rather than representative of a specific market process.
- Empirical threshold calibration is a modeling choice and should be reported alongside regret results.

## Code Map

- Exact algorithms and invariants: `src/olc_exact.py`
- Sequence families: `src/sequences.py`
- Figure generation and dashboard curation: `run_experiment.py`

## References

- Aviv, Y. and Pazgal, A. (2005). "A Partially Observed Markov Decision Process for Dynamic Pricing." *Management Science*. <https://doi.org/10.1287/mnsc.1050.0393>
- Bastani, H., Bayati, M., and Khosravi, K. (2021). "Mostly Exploration-Free Algorithms for Contextual Bandits." *Management Science*. <https://doi.org/10.1287/mnsc.2020.3605>
- Choi, H., Mela, C. F., Balseiro, S. R., and Leary, A. (2020). "Online Display Advertising Markets: A Literature Review and Future Directions." *Information Systems Research*. <https://doi.org/10.1287/isre.2019.0901>
- Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., and Singer, Y. (2006). "Online Passive-Aggressive Algorithms." *JMLR*. <https://www.jmlr.org/papers/v7/crammer06a.html>
- de Rooij, S., van Erven, T., Grunwald, P. D., and Koolen, W. M. (2014). "Follow the Leader If You Can, Hedge If You Must." *JMLR*. <https://jmlr.org/papers/v15/rooij14a.html>
- Feder, M., Merhav, N., and Gutman, M. (1992). "Universal Prediction of Individual Sequences." *IEEE Transactions on Information Theory*. <https://doi.org/10.1109/18.144705>
- Lykouris, T., Mirrokni, V., and Paes Leme, R. (2018). "Stochastic Bandits Robust to Adversarial Corruptions." *STOC*. <https://doi.org/10.1145/3188745.3188758>
- McMahan, B. (2011). "Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization." *AISTATS*. <https://proceedings.mlr.press/v15/mcmahan11b.html>
- Orabona, F. (2019). *A Modern Introduction to Online Learning*. <https://arxiv.org/abs/1912.13213>
- Shalev-Shwartz, S., Singer, Y., and Srebro, N. (2007). "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM." *ICML*. <https://doi.org/10.1145/1273496.1273598>
- Zinkevich, M. (2003). "Online Convex Programming and Generalized Infinitesimal Gradient Ascent." *ICML*. <https://martin.zinkevich.org/publications/ICML03.pdf>
