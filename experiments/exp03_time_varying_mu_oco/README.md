# exp03_time_varying_mu_oco

Experiment 03 studies SMART for OCO quadratic losses with time-varying centers:

- $\ell_t(a) = \frac{1}{2}(a-\mu_t)^2$

This clean Python implementation was derived from the uploaded notebooks:

- `smart_oco.ipynb`
- `SMART_oco_anytime_lr.ipynb`
- `smart_binary.ipynb` (archived for context)

## Objective

Demonstrate SMART's core mechanism in a clean 1D OCO setting:

1. Preserve optimistic performance in benign regimes.
2. Protect against realistic hard/nonstationary regimes.
3. Show interpretable switching behavior (`Sigma_t` crossing threshold) rather than only final regret.

The key design variable is the input sequence `mu_t`. Sequence quality determines whether the plots are informative.

## What is implemented

- FTL baseline
- OGD baseline
  - fixed learning rate: $\eta = 2/\sqrt{n}$
  - optional anytime learning rate: $\eta_t = 1/\sqrt{t}$
- SMART switching policy with Eq. (6) style switching statistic and threshold $2\sqrt{n}$

## Paper-facing `\mu_t` sequence suite

Experiment 03 should focus on three realistic sequences:

1. `stable_benign` (FTL-dominant)
- AR(1)-style low-noise process around a stable center.
- Intended takeaway: SMART stays close to FTL and avoids unnecessary switching.

2. `corruption_burst` (adversarial-realistic)
- Stable baseline with transient windows where centers flip sign strongly.
- Intended takeaway: FTL degrades in burst windows; SMART switches and reduces damage.

3. `drift_plus_shift` (representative mixed regime)
- Gradual drift followed by one moderate structural shift.
- Intended takeaway: SMART transitions from optimistic to robust behavior with interpretable timing.

## Acceptance criteria (for paper use)

1. `stable_benign`: `SMART` is approximately `FTL` and usually does not switch.
2. `corruption_burst`: `FTL` degrades sharply; `SMART` switches and materially lowers regret.
3. `drift_plus_shift`: switch occurs around post-drift deterioration and `SMART` lands between optimistic and robust extremes.
4. Diagnostics: at least one sequence must show clear `Sigma_t` threshold crossing aligned with regime change.

## Threshold calibration

- In this setting, Eq.(6) `Sigma_t` values are often small relative to `2*sqrt(n)`.
- The runner therefore exposes `--threshold-scale` and uses a calibrated default (`0.0035`) so switch behavior is observable and comparable across the three sequences.

## Figures

Curated paper-candidate figures live in `figures/` with labels/titles mapped in `figures/INDEX.md`.

## Run

```bash
cd experiments/exp03_time_varying_mu_oco
python run_experiments.py --n 1000 --trials 30 --threshold-scale 0.0035
```

Anytime OGD variant:

```bash
python run_experiments.py --n 1000 --anytime-lr
```

Run selected scenarios only:

```bash
python run_experiments.py --scenario stable_benign corruption_burst drift_plus_shift
```

## Outputs

Figures are written to:

- `outputs/figures/*_mu.png`
- `outputs/figures/*_regret.png`
- `outputs/figures/*_diagnostics.png`

## Known issues

1. Threshold choice is still calibration-sensitive and should be reported alongside results.
2. These are synthetic but structured sequences; realism is controlled, not dataset-derived.
3. Conclusions depend on domain bounds, quadratic form, and robust baseline choice (OGD schedule).
