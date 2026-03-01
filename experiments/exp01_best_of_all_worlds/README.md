# exp01_best_of_all_worlds

Legacy SMART/binary-prediction experiment set from the original `Best-of-All-Worlds` repo.

## Objective

This experiment is the main empirical validation for the binary-prediction claims:

1. SMART tracks FTL on benign/stochastic sequences where FTL is strong.
2. SMART avoids FTL's linear-regret failures on adversarial/worst-case-style sequences.
3. SMART stays close to the better benchmark (FTL vs robust baseline) up to a constant factor.
4. On loss-based sequences that hurt AdaHedge, SMART remains competitive.

## Experimental design

The codebase uses synthetic binary sequences that span easy-to-hard regimes.

- Easy regime:
  - i.i.d. Bernoulli sequences (`generate_iid_sequence`)
  - Goal: show FTL and SMART perform similarly while robust worst-case algorithms are conservative.
- Hard regime:
  - FMG-style sequences (`generate_FMG_sequence`, `generate_highlossFMG_ftl`)
  - Goal: increase lead changes/crossings and show FTL degrades while SMART transitions toward robust behavior.
- Loss-based regime:
  - targeted loss-difference sequence generation (`loss_based_sequence_generation`), e.g. `n^0.4`.
  - Goal: reproduce settings where AdaHedge underperforms and SMART remains strong.

## Entry points

- `python graph_driver.py`
- `python blackwells_worst_experimentation.py`
- `python graphing_gui.py`

## Main code locations

- `Scripts/`: algorithm and sequence-generation logic
- `Graphs/`: saved plots
- `Playground/`: exploratory UI/testing scripts

## Known issues

1. Legacy structure is non-modular; reproducing specific figures is not always one-command.
2. Seeds/config are not consistently pinned across all drivers.
3. `Graphs/` contains mixed historical and paper-candidate outputs.
4. Multiple algorithm variants exist; some runs are not clearly aligned to final paper configs.

## Notes

- This experiment retains original layout for compatibility.
