# SMART Dashboard

Static dashboard for browsing SMART theory and experiment reports.

## What it shows

- `Overview` tab: sections 1-5 from `smart_algorithm.md`
- `Best-of-All-Worlds` tab: binary prediction report from `exp01_best_of_all_worlds`
- `OLC` tab: online linear classification report from `exp05_true_ftl_olc`
- Per-experiment report:
  - rendered `README.md`
  - figure gallery from `figures/`

## Regeneration

From repository root:

```bash
python experiments/dashboard/generate_dashboard.py
```

This writes:

- `experiments/dashboard/index.html`
- `experiments/dashboard/assets/` (copied experiment figures for self-contained deploys)

## Viewing

From repository root:

```bash
python -m http.server 8000
```

Then open:

- `http://localhost:8000/experiments/dashboard/`
