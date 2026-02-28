# smart binary prediction

This repository now separates experiment code from manuscript context.

## Layout

- `experiments/`: runnable experiment code and experiment-specific outputs
  - `exp01_best_of_all_worlds/`
  - `exp02_online_linear_classification/`
  - `exp03_time_varying_mu_oco/`
- `paper/`: manuscript and figure context for writing
  - `manuscripts/`: conference/journal versions (`aistats`, `colt`, `ms`, `arxiv`)
  - `shared/`: shared `.tex` and `.bib` files
  - `figures/`: figure assets referenced by manuscripts
- `archive/`: imported artifacts and metadata from repo merge

## Workflow

- Add new work as `experiments/expNN_<short_name>/`.
- Keep each experiment self-contained (code + local outputs + README).
- Promote finalized figures from experiment folders into `paper/figures/` only when ready for manuscript use.
- Treat `paper/` as read-only context unless explicitly working on paper edits.

## Notes

- `paper/` was set to read-only at the filesystem level to protect manuscript context.
- To intentionally edit paper files, restore write permission locally when needed:

```bash
chmod -R u+w paper
```
