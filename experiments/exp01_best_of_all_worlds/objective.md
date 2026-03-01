# Objective: Experiment 01 (Best-of-All-Worlds, Binary Prediction)

## What this experiment is trying to show

This experiment is the main empirical validation for the binary-prediction claims in the paper:

1. SMART (BoBW) tracks FTL on benign/stochastic sequences where FTL is strong.
2. SMART avoids FTL's linear-regret failures on adversarial/worst-case-style sequences.
3. SMART stays close to the better benchmark (FTL vs Cover-style robust baseline) up to a constant factor, consistent with the theory.
4. On loss-based sequences that hurt AdaHedge, SMART can remain competitive (including near-constant regret behavior in reported settings).

In paper terms, this is the empirical story behind instance-optimal adaptation via one switch from optimistic (FTL) to robust policy.

## Experiment design

The codebase centers on synthetic binary sequences that span easy-to-hard regimes.

- Easy regime:
  - i.i.d. Bernoulli sequences (`generate_iid_sequence`)
  - Goal: show FTL and SMART perform similarly; robust worst-case algorithms are overly conservative.
- Hard regime:
  - FMG-style sequences (`generate_FMG_sequence`, `generate_highlossFMG_ftl`)
  - Goal: increase lead changes/crossings and show FTL degrades while SMART transitions toward robust behavior.
- Loss-based regime:
  - Targeted loss-difference sequence generation (`loss_based_sequence_generation`), e.g. `n^0.4`.
  - Goal: reproduce settings where AdaHedge underperforms and SMART remains strong.

Primary scripts and components:

- `graph_driver.py`: quick entry script for loss-based sequence visualization.
- `Scripts/graphing.py`: plotting/comparison utilities.
- `Scripts/sequence_generation.py`: sequence families used in figures.
- `Scripts/algorithms.py`, `Scripts/classified_Hedge.py`: policy implementations including SMART variants.
- Existing figure artifacts under `Graphs/`.

## What counts as success

- Regret curves demonstrate the expected phase behavior:
  - SMART ≈ FTL on stochastic/low-crossing sequences.
  - SMART << FTL on high-crossing adversarial sequences.
  - SMART remains near min(FTL, robust baseline) trend.
- Reported comparisons against Cover/AdaHedge are directionally consistent with manuscript figures.

## Problems and limitations we are currently encountering

1. Legacy, non-modular structure:
- Core logic is spread across `Scripts/`, drivers, and historical plotting code.
- Reproducing a specific paper figure is not always a one-command path.

2. Reproducibility/noise control:
- Random seeds are not consistently managed across all drivers.
- Some plots are exploratory and rely on manual parameter tuning.

3. Mixed artifact quality:
- `Graphs/` includes both historical and paper-candidate outputs, making provenance unclear.

4. Benchmark alignment drift risk:
- Multiple SMART/algorithm variants exist; not all runs clearly map to the final paper's exact configuration.

## Immediate next steps for this experiment

1. Add a single canonical runner that regenerates only the paper-relevant binary figures.
2. Pin seeds and config in a small config file for deterministic reruns.
3. Separate archival plots from "paper-target" plots in distinct output subfolders.
