# 130M Mix5B Results

## Run Summary

- Model: `130M` preset
- Data:
  - Stage A: `/data/mix5B/edu_*.bin,/data/mix5B/dclm_*.bin,/data/mix5B/code_*.bin,/data/mix5B/math_*.bin`
  - Stage B anneal: `/data/mix5B/anneal_*.bin`
  - Validation: original `fineweb_val_*.bin`
- Seed: `1337`
- Schedule horizon: `4770` steps
- Anneal rescue final checkpoint: `/data/runs/130M_mix5B/model.pt`
- Final validation loss: `4.2793`
- Spend recorded on Modal for the rescue + eval day: `$4.77`

## Final 7-Task Scoreboard

| Task | Mix5B 130M | Baseline | Delta |
| --- | ---: | ---: | ---: |
| hellaswag | 29.32 | 32.54 | -3.22 |
| openbookqa | 25.60 | 28.20 | -2.60 |
| winogrande | 51.54 | 52.17 | -0.63 |
| arc_challenge | 24.57 | 23.46 | +1.11 |
| arc_easy | 41.67 | 37.71 | +3.96 |
| boolq | 52.78 | 61.31 | -8.53 |
| piqa | 59.41 | 65.40 | -5.99 |
| avg | 40.70 | 42.97 | -2.27 |

## Bug

### Root Cause

The multi-pattern loader expanded `edu_*`, `dclm_*`, `code_*`, and `math_*` into ordered file lists and then consumed them family by family. Even after adding shard "shuffle", the implementation chose among non-empty families rather than uniformly across all remaining shards. That preserved family-level drift and let long families dominate early training.

### Symptoms

- Stage A consumed shards in practice as `edu -> dclm -> code`, with `math` never reached before the run degraded.
- Validation loss climbed from `3.74` at the last healthy checkpoint to `5.13` in the broken tail.
- The model drifted toward the currently active family instead of learning from a stable mixture.

### Fix

`hobbylm/data.py` now performs a seeded global shuffle over the full shard list for each dataset pass. With `seed=1337`, every epoch is reproducible, but shard selection is randomized across all remaining shards rather than stepping to the next family in list order.

## What We Learned Before 1B

- Multi-pattern data loading needs explicit global mixing guarantees; "multiple globs" is not enough.
- Cheap CPU verification is worth doing before expensive GPU runs.
- Rescue runs can salvage a bad training tail, but they do not replace a clean end-to-end validation run on the intended recipe.
- Checkpoint provenance matters: later checkpoints from a corrupted curriculum should not be reused blindly.

## Clean Rerun Still Needed

We still need one clean 130M rerun with the fixed loader:

- Stage A on `edu + dclm + code + math` with seeded shard shuffle enabled from step `0`
- Stage B anneal continuation on `anneal_*`
- Final lmeval on the clean rerun checkpoint

That rerun is the real validation for the Mix5B recipe before spending on the `1B` mission.
