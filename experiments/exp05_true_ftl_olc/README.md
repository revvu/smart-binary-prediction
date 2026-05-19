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

## Design gate

### SMART behavior claim

The experiment must show:

1. **Preserve optimism:** on benign predictable streams, SMART should remain visually close to true FTL and avoid FTRL's robustness tax.
2. **Protect in hard regimes:** on high-variation or adversarial streams, SMART should switch and avoid FTL's large regret growth.
3. **Interpretable switch:** in mixed streams, the switch should occur when $\Sigma_t$ crosses the calibrated threshold near the onset of sustained deterioration.

### Sequence families

The default sequence families are:

1. `iid_separable_margin`: i.i.d. bounded-margin separable stream.
2. `massart_10`: the same stream with independent 10% label flips.
3. `alternating_antileader`: fixed feature direction with alternating labels, used as an illustrative FTL failure mode.
4. `benign_to_hard_suffix`: separable prefix followed by an adaptive anti-leader suffix, the main SMART single-switch regime.
5. `separator_drift`: gradual rotation of the latent separator, included as a nonstationary diagnostic rather than the central proof example.

### Why these sequences are appropriate

The i.i.d. and Massart streams are natural online classification baselines where optimism should not be punished. The alternating anti-leader stream is deliberately illustrative: it creates repeated FTL cancellation/reversal and checks robust protection. The benign-to-hard suffix stream is the closest match to SMART's one-switch design because the FTL prefix is useful and the adversarial suffix creates sustained trace growth. The separator drift stream links the experiment to practical concept drift, but static regret to the best fixed comparator should not be interpreted as dynamic tracking regret.

### Acceptance criteria

A successful run should show:

1. In `iid_separable_margin`, SMART and FTL have nearly identical regret and FTRL is higher.
2. In `alternating_antileader`, FTL grows faster than FTRL and SMART switches.
3. In `benign_to_hard_suffix`, SMART is near or below the lower envelope of the fixed baselines by combining FTL's prefix and FTRL's suffix.
4. The diagnostic plot shows monotone $\Sigma_t$ and switch timing that is explainable from the threshold crossing.
5. The threshold sweep shows the cost of switching too early or too late.

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

## Limits and non-claims

- The streams are synthetic and mechanism-oriented.
- The anti-leader stream is illustrative rather than representative.
- The separator drift diagnostic is not a dynamic-regret benchmark.
- Empirical threshold calibration is itself a modeling choice and should be reported alongside the regret results.

## Code map

- Exact algorithms and invariants: `src/olc_exact.py`
- Sequence families: `src/sequences.py`
- Figure generation and dashboard curation: `run_experiment.py`
