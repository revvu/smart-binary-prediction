# exp04_leader_path_synthesis

This experiment builds **realistic-looking online linear classification sequences** by steering toward target leader trajectories while keeping sampled points plausible.

It is designed to demonstrate the full SMART story without relying on simplistic handcrafted sequences.

## Key idea

Instead of manually hard-coding obvious label patterns, we:

1. Define target leader regimes (stable, drift, regime shift, bursty corruption).
2. Sample candidate points from a time-varying Gaussian feature prior.
3. Choose the next `(z_t, y_t)` that best balances:
   - making next realized FTL leader close to target,
   - realism (feature/margin plausibility),
   - limited label flipping from latent base labels.

This guarantees the resulting leader sequence is realizable by construction for the generated data.

## Files

- `src/synthesis.py`: regime generators + search-based sequence synthesis.
- `src/eval.py`: FTL/FTRL/SMART regret curves and switch-stat diagnostics.
- `run_experiment.py`: end-to-end runner producing figures.
- `objective.md`: experimental objective, setup, and known limitations.

## Run

```bash
cd experiments/exp04_leader_path_synthesis
python run_experiment.py --t-max 1000 --t-step 100 --runs 8 --d 5 --n-candidates 64
```

Notes:

- The runner uses a base threshold scale for SMART (`--threshold-scale`, default `0.2`), with a more aggressive scale during bursty-corruption regimes.
- This mirrors using an empirical threshold calibration for demonstration, rather than only the conservative theoretical threshold.

## Outputs

- `outputs/figures/regret_vs_horizon_by_regime.png`
- `outputs/figures/<regime>_diagnostics.png`
