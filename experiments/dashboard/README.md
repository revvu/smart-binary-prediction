# SMART Dashboard

Static dashboard for browsing SMART theory and experiment reports.

## What it shows

- `Overview` tab: rendered `smart_algorithm.md`
- One tab per `experiments/expNN_*` directory
- Per-experiment report:
  - rendered `README.md`
  - figure gallery from `figures/`
  - rendered `figures/INDEX.md` (if present)

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
