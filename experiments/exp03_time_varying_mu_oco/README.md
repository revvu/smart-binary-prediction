# exp03_time_varying_mu_oco

Experiment 03 studies SMART for OCO quadratic losses with time-varying centers:

- $\ell_t(a) = \frac{1}{2}(a-\mu_t)^2$

This clean Python implementation was derived from the uploaded notebooks:

- `smart_oco.ipynb`
- `SMART_oco_anytime_lr.ipynb`
- `smart_binary.ipynb` (archived for context)

## Objective

Implement and evaluate SMART when losses are
`ell_t(a)=0.5*(a-mu_t)^2` with time-varying `mu_t` (not only constant `mu_t=1/4`).

Main question:

- Does SMART adapt across nonstationary `mu_t` regimes while retaining robust behavior when optimistic tracking becomes unreliable?

## What is implemented

- FTL baseline
- OGD baseline
  - fixed learning rate: $\eta = 2/\sqrt{n}$
  - optional anytime learning rate: $\eta_t = 1/\sqrt{t}$
- SMART switching policy with Eq. (6) style switching statistic and threshold $2\sqrt{n}$

## Built-in `\mu_t` scenarios

- `constant_0.25` (existing baseline)
- `step_0.75_to_0.25` (requested: first half 3/4, second half 1/4)
- `sine`
- `uniform_random`

## Figures

Curated paper-candidate figures live in `figures/` with labels/titles mapped in `figures/INDEX.md`.

## Run

```bash
cd experiments/exp03_time_varying_mu_oco
python run_experiments.py --n 1000
```

Anytime OGD variant:

```bash
python run_experiments.py --n 1000 --anytime-lr
```

Run selected scenarios only:

```bash
python run_experiments.py --scenario constant_0.25 step_0.75_to_0.25
```

## Outputs

Figures are written to:

- `outputs/figures/*_mu.png`
- `outputs/figures/*_regret.png`

## Known issues

1. In current settings SMART often does not switch (`switch round = n+1`), effectively matching FTL.
2. The threshold `2*sqrt(n)` appears conservative for several generated `mu_t` schedules.
3. Current scenarios are limited (constant/step/sine/random), not fully adversarial stress tests.
4. Conclusions are sensitive to domain bounds, loss form, and OGD schedule choices.
