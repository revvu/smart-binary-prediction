# SMART Dashboard

Static dashboard for browsing SMART theory and experiment reports.

## What it shows

- `Overview` tab: rendered `smart_algorithm.md`
- One tab per active `experiments/expNN_*` directory
- A combined `Legacy OLC` tab for `exp02_online_linear_classification` and `exp04_leader_path_synthesis`
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
