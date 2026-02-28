# exp03_time_varying_mu_oco

Experiment 03 studies SMART for OCO quadratic losses with time-varying centers:

- $\ell_t(a) = \frac{1}{2}(a-\mu_t)^2$

This clean Python implementation was derived from the uploaded notebooks:

- `smart_oco.ipynb`
- `SMART_oco_anytime_lr.ipynb`
- `smart_binary.ipynb` (archived for context)

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
