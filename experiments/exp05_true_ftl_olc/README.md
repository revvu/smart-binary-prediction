# Experiment 05: Online Linear Classification

## Objective

This experiment demonstrates SMART in a vector-valued online linear classification setting where:

1. FTL is computed exactly, not approximated by played-point subgradients.
2. The robust baseline is a concrete quadratic-FTRL algorithm with a standard online-convex-optimization interpretation.
3. The SMART switch statistic is the exact adapted FTL regret trace $\Sigma_t$.
4. The sequence families are bounded OLC feature-label streams designed to show clean benign, delayed-signal, corrupted, and slow-leader hard regimes.

The paper-facing claim is:

SMART preserves FTL's advantage on benign streams, switches when the exact FTL regret trace indicates sustained danger, and gives an interpretable one-switch compromise between optimism and worst-case robustness.

## Formal Setting

At each round $t$, the learner chooses a classifier vector in the Euclidean unit ball:

$$
\mathcal{W}=\{w\in\mathbb{R}^d:\|w\|_2\le 1\}.
$$

The learner then observes a bounded feature-label pair $(x_t,y_t)$ with

$$
\|x_t\|_2\le 1,\qquad y_t\in\{-1,+1\},
$$

and incurs the online linear classification surrogate

$$
\ell_t(w)=\frac12|\langle w,x_t\rangle-y_t|.
$$

Because $\|w\|_2\le1$ and $\|x_t\|_2\le1$, we have $|\langle w,x_t\rangle|\le1$. Therefore the absolute value collapses to a linear loss on the entire feasible set:

$$
\ell_t(w)=\frac12(1-y_t\langle w,x_t\rangle).
$$

Define the signed feature vector and cumulative signed feature sum:

$$
s_t=y_t x_t,\qquad M_t=\sum_{i=1}^t s_i=\sum_{i=1}^t y_i x_i.
$$

For any fixed comparator $u\in\mathcal{W}$, cumulative loss on a prefix is

$$
L_t(u)=\sum_{i=1}^t \ell_i(u)
=\frac{t}{2}-\frac12\langle M_t,u\rangle.
$$

The best fixed comparator loss is therefore

$$
\min_{\|u\|_2\le1}L_t(u)
=
\frac{t}{2}
-\frac12
\max_{\|u\|_2\le1}\langle M_t,u\rangle.
$$

By Cauchy-Schwarz,

$$
\max_{\|u\|_2\le1}\langle M_t,u\rangle=\|M_t\|_2,
$$

so the exact prefix comparator loss is

$$
L_t^*=\min_{u\in\mathcal{W}}L_t(u)=\frac{t}{2}-\frac12\|M_t\|_2.
$$

This identity is the computational hinge of the experiment. FTL, the comparator, and the SMART trace all reduce to maintaining $M_t$, computing one norm, and normalizing one vector. The cost is $O(Td)$ per trial.

Regret is always static regret against the best fixed action in hindsight:

$$
\mathrm{Reg}_T(A)=\sum_{t=1}^T\ell_t(w_t^A)-L_T^*.
$$

Adaptive algorithms such as FTRL or SMART can have negative static regret on some nonstationary synthetic streams. That is not a bug; it means the adaptive policy beat the best fixed comparator on that realized sequence.

## Algorithms

### Exact FTL

The prefix leader after observing $t$ rounds is

$$
w_t^*
=
\begin{cases}
M_t/\|M_t\|_2, & M_t\ne0,\\
0, & M_t=0.
\end{cases}
$$

FTL plays the previous prefix leader:

$$
w_t^{\mathrm{FTL}}
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
c_t=-\frac12 y_t z_t,\qquad \ell_t(w)=\frac12+\langle c_t,w\rangle.
$$

With the time-varying quadratic regularizer

$$
\psi_t(w)=\frac{\sqrt{t}}{2\eta_0}\|w\|_2^2+\iota_{\mathcal{W}}(w),
\qquad \eta_0=\sqrt{2},
$$

where $\iota_{\mathcal{W}}$ is the indicator of the unit ball, FTRL is

$$
w_t^{\mathrm{FTRL}}
=
\arg\min_{w\in \mathcal{W}}
\left\langle\sum_{i<t}c_i,w\right\rangle
+\frac{\sqrt{t}}{2\eta_0}\|w\|_2^2.
$$

Since $\sum_{i<t}c_i=-\frac12M_{t-1}$, the unconstrained optimum solves

$$
-\frac12M_{t-1}+\frac{\sqrt{t}}{\eta_0}w=0,
$$

so

$$
\tilde{w}_t=\frac{\eta_0}{2\sqrt{t}}M_{t-1}.
$$

The feasible set is the Euclidean unit ball, hence the implemented update is

$$
w_t^{\mathrm{FTRL}}
=
\Pi_{\mathcal{W}}\left(\frac{\eta_0}{2\sqrt{t}}M_{t-1}\right).
$$

This is the correct robust baseline for this experiment because the losses are bounded linear losses on a bounded Euclidean domain, where quadratic FTRL is the standard $O(\sqrt{T})$ no-assumption fallback.

This matches the FTRL template in Orabona's Chapter 7:

$$
w_t\in\arg\min_{w\in V}\psi_t(w)+\sum_{i<t}\langle g_i,w\rangle,
$$

with $V=\mathcal{W}$, $g_i=c_i$, and the increasing Euclidean quadratic regularizer above. It also matches Orabona's Chapter 8 online-linear-classification surrogate once the bounded-domain condition $|\langle w,x_t\rangle|\le1$ is enforced. The broader online-convex-optimization framing follows Zinkevich's static-regret setting, and the FTRL interpretation is standard in the McMahan FTRL/mirror-descent literature.

Orabona's FTRL regret bound for Euclidean $L$-Lipschitz losses with

$$
\psi_t(w)=\frac{L\sqrt{t}}{2\alpha}\|w\|_2^2
$$

is

$$
\mathrm{Reg}_T(u)
\le
L\sqrt{T}\left(\frac{\|u\|_2^2}{2\alpha}+\alpha\right).
$$

In this experiment $\|z_t\|_2\le1$ and $y_t\in\{-1,+1\}$, so $\|c_t\|_2\le1/2$ and $L=1/2$ for the implemented half-scaled OLC loss. The implemented regularizer has coefficient $\sqrt{t}/(2\eta_0)$ with $\eta_0=\sqrt{2}$, so it matches Orabona's form with $\alpha=L\eta_0=1/\sqrt{2}$. For any comparator $u\in\mathcal{W}$, $\|u\|_2\le1$, hence

$$
\mathrm{Reg}_T^{\mathrm{FTRL}}(u)
\le
\frac12\sqrt{T}
\left(\frac{1}{2(1/\sqrt{2})}+\frac{1}{\sqrt{2}}\right)
=
\sqrt{\frac{T}{2}}.
$$

Thus the paper-backed SMART robust-regret budget for this exact implementation is $g(T)=\sqrt{T/2}$. A threshold $\sqrt{2T}$ would still be a conservative valid bound for losses with $\|c_t\|_2\le1$, but it is not the scale-matched bound for the half-scaled loss used here.

### SMART

SMART starts as exact FTL and computes the exact adapted FTL regret trace:

$$
\Sigma_t
=
\sum_{i=1}^t\ell_i(w_i^{\mathrm{FTL}})
-L_t^*
=
\sum_{i=1}^t\ell_i(w_i^{\mathrm{FTL}})
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

The theory threshold is $\theta=g(T)=\sqrt{T/2}$. The calibrated variant uses

$$
\theta=g_{\mathrm{emp}}(T),
$$

where $g_{\mathrm{emp}}(T)$ is the largest observed FTRL regret across designated hard calibration streams. After switching, FTRL is reset and run only on the suffix. The reset is intentional: it matches the SMART proof decomposition into an FTL prefix and a robust-policy suffix.


## Sequence Design

### Behavior Claim

The experiment must show four SMART behaviors:

1. **Preserve optimism:** on benign predictable streams, SMART should remain close to FTL and avoid FTRL's robustness tax.
2. **Protect in hard regimes:** on high-variation or corrupted streams, SMART should switch and avoid FTL's large regret growth.
3. **Avoid unnecessary robustness tax:** on streams with one slowly emerging but persistent leader, SMART should remain close to FTL even when FTRL is much worse than FTL.
4. **Explain the switch:** in mixed streams, the switch should be readable from $\Sigma_t$ crossing the calibrated threshold after sustained deterioration begins.

### Literature Motivation

The sequence design uses application stories and individual-sequence examples from the literature.

Contextual pricing and allocation motivate the clean covariate-diverse stream. Bastani, Bayati, and Khosravi show that greedy or mostly-greedy contextual-bandit policies can be effective when contexts have enough diversity, which is the operational analogue of a stable informative feature process.

Dynamic pricing and revenue-management models motivate the delayed-signal stream. Aviv and Pazgal study pricing under partially observed demand dynamics, illustrating why a sequential decision-maker may initially see weak or uninformative feedback before a stable response pattern emerges.

Online advertising and platform feedback motivate the strategic-corruption stream. Choi, Mela, Balseiro, and Leary survey online display-advertising markets, where feedback quality and strategic behavior matter; corruption-robust online learning, including Lykouris, Mirrokni, and Paes Leme, motivates stress tests with reliable prefixes and harmful suffixes.

Feder, Merhav, and Gutman, and later de Rooij, van Erven, Grunwald, and Koolen, motivate the individual-sequence stress tests. Their examples are designed to expose when it is safe to follow the leader and when a robust policy is needed. Here we build OLC analogues by controlling the signed feature sum $M_t$ through bounded feature-label pairs, including a loss-based slow-leader stream inspired by the older AdaHedge stress sequence.

### OLC Construction Rule

All sequences must be valid OLC streams:

$$
\|x_t\|_2\le1,\qquad y_t\in\{-1,+1\}.
$$

The generator cannot directly specify arbitrary losses or arbitrary leader paths. It must specify feature-label pairs. The only lever that affects FTL and the comparator is the signed update

$$
y_t x_t.
$$

This is why every sequence below is described through how it shapes $M_t=\sum_{i\le t}y_i x_i$.

### Primary Sequence Families

The main regret-by-horizon figures use four primary sequence families.

**`covariate_diverse_stationary`**

Purpose: benign, representative optimism case.

Real-world analogue: a mature contextual pricing, allocation, triage, or recommendation system where covariates are informative from the start and the relationship between features and outcomes is stable. This is the setting in which a greedy or FTL-like optimistic policy should plausibly work well.

Generation: sample a unit separator $u$, labels $y_t\sim\mathrm{Unif}\{-1,+1\}$, and orthogonal unit noise $v_t\perp u$. Then set

$$
x_t=y_t\operatorname{unit}(0.72u+0.58v_t).
$$

The signed update is

$$
y_t x_t=\operatorname{unit}(0.72u+0.58v_t),
$$

so $M_t$ has persistent positive drift toward $u$ with covariate variation around that direction.

Why it is appropriate: this is the OLC analogue of stable contextual structure with covariate diversity. It tests the no-unnecessary-tax claim: FTL should identify the stable direction quickly, and SMART should not switch.

**`delayed_signal_emergence`**

Purpose: cold-start or delayed-product-market-fit diagnostic.

Real-world analogue: a launch, new market, new advertising campaign, new clinical workflow, or newly instrumented platform where early observations are not yet aligned with the eventual stable response model. After enough deployment time, the signal becomes structured.

Generation: sample a latent separator $u$ and use a split at $0.45T$. Before the split, draw bounded covariates orthogonal to $u$ with independent random labels, so the prefix has no stable signed-feature drift toward the eventual separator. After the split, switch to the covariate-diverse margin model:

$$
x_t=y_t\operatorname{unit}(0.72u+0.58v_t).
$$

Why it is appropriate: this models a launch or deployment where early feedback is uninformative but later feedback becomes structured. It is application-driven but graphically different from the stationary benign stream: FTL must recover from an unhelpful prefix, while SMART should avoid switching merely because the early data are noisy.

**`strategic_corruption_suffix`**

Purpose: paper-facing hardening sequence with an applied corruption story.

Real-world analogue: a deployed model whose early feedback is reliable, followed by a sustained period of low-quality or strategically distorted feedback. Examples include click-fraud bursts in advertising, bot traffic in recommendation systems, sensor degradation, data pipeline failure, or users strategically changing behavior after learning the policy.

Generation: sample a unit direction $u$. For the first $20\%$ of the horizon, set $x_t\approx u$ and $y_t=+1$, so $y_t x_t$ builds a stable leader. For the next $20\%$, keep $x_t\approx u$ but set $y_t=-1$, so $y_t x_t\approx-u$ erodes the trusted signal. For the final $60\%$, keep $x_t\approx u$ and alternate labels, keeping $M_t$ near the decision boundary and making FTL chase unstable leaders.

Concretely, the implementation uses

$$
x_t=\operatorname{unit}(0.96u+0.04v_t),
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

**`loss_based_slow_leader_gap`**

Purpose: primary FTRL-stress sequence with one slowly emerging leader.

Real-world analogue: none claimed as a calibrated application model. This is a deliberately stylized stress test that translates the older loss-based AdaHedge stress construction into OLC feature-label form.

Generation: fix $x_t=e_1$. Choose labels greedily so the signed one-dimensional leader gap

$$
S_t=\sum_{i=1}^t y_i
$$

stays close to the target curve $t^{0.4}$. Equivalently, because $M_t=S_t e_1$, the OLC stream keeps one persistent leader but makes that leader's lead emerge sublinearly and slowly. At each step the generator chooses $y_t=-1$ or $y_t=+1$ according to which update makes $S_t$ closer to $t^{0.4}$.

Why it is appropriate: this sequence separates FTL's optimism from FTRL's conservative shrinkage. Once the sign of $M_t$ is positive, FTL plays the stable leader direction and incurs only a small boundary cost. Quadratic FTRL instead plays with magnitude proportional to $\|M_{t-1}\|_2/\sqrt{t}\approx t^{-0.1}$, so it remains under-confident for a long time even though the leader is persistent. This tests whether SMART preserves FTL's advantage when the robust baseline is the poor choice.

### Diagnostic Sequence

`switching_leaders` fixes $x_t=e_1$ and uses label blocks of length 20 with alternating signs. This is included because the same style of deterministic boundary-heavy stream produced invalid negative regret in experiment 2. Under exact $M_t$-based FTL, the diagnostic trace verifies non-negative FTL regret and monotone $\Sigma_t$.

### Acceptance Criteria

A successful run should show:

1. In `covariate_diverse_stationary`, SMART and FTL have nearly identical regret and FTRL is higher.
2. In `delayed_signal_emergence`, SMART should not switch during the uninformative prefix if FTL regret remains within the calibrated budget; the curve should show a distinct cold-start cost.
3. In `strategic_corruption_suffix`, SMART switches during the corrupted regime and materially reduces FTL's late-horizon regret.
4. In `loss_based_slow_leader_gap`, FTL and SMART stay near each other while FTRL is materially worse, showing that SMART avoids the robust baseline when the exact FTL trace remains small.
5. The switch-diagnostic plot shows monotone $\Sigma_t$, the calibrated threshold, and interpretable switch timing.
6. `switching_leaders` confirms that experiment 2's negative regret was an implementation artifact.

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
- horizons: $T=100,200,\ldots,1000$
- trials per horizon: 24
- empirical-threshold calibration trials: 8
- deterministic seed: 7

Heavier paper-profile run:

```bash
python experiments/exp05_true_ftl_olc/run_experiment.py --paper-profile
```

Optional one-time dimension diagnostic:

```bash
python experiments/exp05_true_ftl_olc/run_experiment.py --dimension-sweep
```

Each horizon uses fresh sequences of exactly that length. Plots report mean regret with 95% confidence bands where applicable.

In the regret-by-horizon figure, line paths and uncertainty bands are drawn at the exact regret values. Marker glyphs are slightly offset in display-space points only where FTL and SMART mean-regret markers overlap in benign regimes.

The optional dimension-sweep figure fixes the horizon at $T=10000$ and evaluates each primary sequence family at $d=1,2,\ldots,50$. It is a slow one-time diagnostic for whether the qualitative ordering of FTL, FTRL, and calibrated SMART is stable as feature dimension changes; it is not part of the default run.

## Outputs and Figure Provenance

Generated outputs:

- `outputs/figures/exp05_olc_regret_by_horizon.png`
- `outputs/figures/exp05_olc_empirical_threshold.png`
- `outputs/figures/exp05_olc_loss_based_slow_leader_regret_vs_sqrt_bound.png`
- `outputs/figures/exp05_olc_calibrated_g_vs_sqrt_bound.png`
- `outputs/figures/exp05_olc_switch_diagnostics.png`
- `outputs/figures/exp05_olc_threshold_calibration.png`
- `outputs/summary_regret.csv`
- `outputs/empirical_g.csv`

Optional dimension diagnostic outputs:

- `outputs/figures/exp05_olc_dimension_sweep.png`
- `outputs/dimension_sweep.csv`

Curated dashboard figures:

- `figures/fig_exp05_olc_regret_by_horizon.png`
- `figures/fig_exp05_olc_empirical_threshold.png`
- `figures/fig_exp05_olc_loss_based_slow_leader_regret_vs_sqrt_bound.png`
- `figures/fig_exp05_olc_calibrated_g_vs_sqrt_bound.png`
- `figures/fig_exp05_olc_switch_diagnostics.png`
- `figures/fig_exp05_olc_threshold_calibration.png`
- `figures/INDEX.md`

Regenerate the dashboard UI with:

```bash
python experiments/dashboard/generate_dashboard.py
```

## Result Interpretation

The combined regret-by-horizon figure has four panels. The two benign panels cover immediate stable signal and cold-start signal emergence; in both, success means SMART remains visually close to FTL and avoids FTRL's robustness tax.

The two hard-regime panels test different stress modes. In `strategic_corruption_suffix`, success means SMART switches after the reliable prefix has been eroded and reduces late-horizon regret. In `loss_based_slow_leader_gap`, success means SMART does not pay FTRL's robustness tax on a stream where the best leader emerges slowly but persistently.

FTRL can have negative static regret in the corruption sequence because an adaptive policy can outperform the best fixed comparator on that nonstationary synthetic stream.

## Limits and Non-Claims

- The streams are synthetic and mechanism-oriented.
- The delayed-signal sequence is not a dynamic-regret benchmark; all reported regrets are static regret against the best fixed comparator.
- The strategic-corruption sequence is exogenous and pre-specified, but adversarially structured. It should be presented as a stress test, not as a calibrated empirical data model.
- The loss-based slow-leader sequence is illustrative and literature-facing rather than representative of a specific market process.
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
