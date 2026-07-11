# CLAUDE.md — Coding rules for this repo (HobbyLM 130M mission)

You are working inside the HobbyLM repo, executing `MISSION-130M.md`. That file owns the
WHAT (goal, phases, budget, gates). This file owns the HOW of writing code. These rules
follow Andrej Karpathy's engineering philosophy — the same ethos this codebase was built
with (it descends from nanoGPT/modded-nanogpt). Violating them is how GPU money gets
burned on silent bugs.

---

## 1. The prime directive: neural-net code fails SILENTLY

Ordinary software crashes when wrong. Training code does not — it trains anyway, just
worse, and you find out $60 later. Karpathy's core insight: most training bugs produce a
model that still learns, so a smooth loss curve proves nothing about correctness.

Therefore: **you may not trust any code you haven't watched fail and succeed.** Before
believing any component works, make it produce a result you can verify by hand.

## 2. The loop: smallest verifiable step, always

Work in this exact cycle, never skipping stages:

1. Make ONE small change (one file, one concern, ideally < 50 lines of diff).
2. Run the cheapest possible check that could catch a mistake in that change
   (CPU smoke test, tiny-shard dry run, single forward pass, unit test).
3. Look at the actual output with your own eyes — print it, decode it, plot it.
4. Only then move to the next change.

Never batch five changes and test once. When a run breaks after five changes, you've
learned nothing about which one did it. One change, one verification, every time.

## 3. Simplicity rules (nanoGPT ethos)

- **Don't be a hero.** Copy the proven pattern that already exists in this repo instead
  of inventing a new one. New abstractions require explicit justification; the default
  answer to "should I add a class/framework/config system?" is no.
- **Fewest moving parts that work.** No new dependencies unless truly unavoidable —
  this repo is deliberately plain PyTorch + tiktoken + numpy + modal. Every dependency
  is a liability you now own.
- **Readable > clever.** Explicit loops beat clever one-liners. Code should read
  top-to-bottom like the algorithm it implements. If a line needs a comment to be
  understood, first try rewriting the line.
- **Delete code when you can.** The best diff is a negative diff. Dead flags, unused
  branches, commented-out experiments: remove them.
- **Additive, minimal diffs to existing files.** This repo has a working, ablation-
  verified training stack. Add flags; do not restructure. If a change touches more
  than ~2 files, stop and reconsider the approach.

## 4. The ML-specific checklist (Karpathy's training recipe, adapted)

Before ANY GPU spend, and after ANY change to data or training code:

- [ ] **Become one with the data.** Decode and READ random samples from the actual
      shards the trainer will consume — not the source dataset, the shards. Look for:
      garbage text, wrong language, truncation, missing EOT boundaries, one source
      dominating. Ten minutes of reading data catches more bugs than any test.
- [ ] **Verify loss at init.** First logged loss should be ≈ ln(50304) ≈ 10.83 (uniform
      over vocab). If it's not within ~0.3 of that, initialization or data loading is
      broken. Stop.
- [ ] **Overfit one batch.** Train on a single fixed batch for a few hundred steps;
      loss must drive toward ~0. If the model can't memorize one batch, the training
      loop is broken — no amount of data will fix it.
- [ ] **Verify the input pipeline at the tensor level.** Print `x[0]` and `y[0]`,
      decode both, confirm y is x shifted by exactly one token. Off-by-one here trains
      a model that "works" and is subtly ruined.
- [ ] **Fix the seed (1337) and confirm two short runs produce identical losses.**
      Non-determinism you didn't choose is a bug you can't debug.
- [ ] **Check shapes and dtypes at the boundaries you touched.** uint16 shards, long
      tensors on device, bf16 compute, fp32 router logits — a silent cast is a silent bug.
- [ ] **Watch the first minutes of any real run.** Step time in the expected range
      (~750 ms for 130M/8×H100)? Loss falling? First checkpoint saves AND resumes?
      If any answer is no, kill the run — it's $2 now or $60 later.

## 5. Change discipline for experiments

- **One variable at a time.** The mission is a controlled experiment (data changes,
  everything else pinned). Never "improve" an unrelated setting mid-mission, even if
  it looks obviously better. Note the idea in RESULTS-130M.md instead.
- **Baselines before improvements.** Any new comparison needs the dumb baseline run
  first, or the number means nothing.
- **Write down every run.** run-name, config delta, cost, outcome — in
  RESULTS-130M.md, at the time you launch, not from memory later.
- **When a result looks surprisingly good, assume a bug first.** Leakage between train
  and val, eval on the wrong checkpoint, and duplicated data all masquerade as wins.

## 6. Debugging protocol

When something is wrong:
1. Reproduce it at the smallest possible scale (CPU, tiny shards, 10 steps).
2. Bisect: strip the change in half until the failure disappears; the bug lives in the
   last half you removed.
3. Print intermediate tensors/values; do not reason from imagination about what the
   code "should" be doing. Look.
4. Read the actual library code when the docs are ambiguous — this repo is small enough
   that reading the source is faster than guessing.
5. Never "fix" a bug you can't explain. A fix that works for unknown reasons is a
   second bug.

## 7. No-touch zones (from the mission — repeated here because they matter)

- `fineweb_val_*.bin` validation shards: read-only, forever.
- Architecture flags: QK-norm, router z-loss, aux-free bias, `norm_topk_prob=False`,
  top_k, expert counts — frozen.
- `fp8` / `all_max` opts: measured-broken in this repo. Never enable.
- The Muon/AdamW parameter split (router must stay on AdamW — Muon destroys routing).
- Budget gates and `[HUMAN GATE]` stops in MISSION-130M.md: absolute.

## 8. Git & housekeeping

- Small, single-purpose commits with messages that state WHY, not just what.
- Commit before every GPU launch, so any run maps to an exact code state.
- Never commit: weights, shards, tokens/keys, or anything in `data/` or `runs/`.
- If you changed repo code, run the full CPU test suite before the commit, not after
  the GPU job.

## 9. Honesty rules

- If you are not sure a step worked, say so and verify — do not narrate success you
  haven't observed.
- Report numbers exactly as measured, including the disappointing ones. A failed gate
  reported accurately is mission progress; a fudged pass is sabotage.
- If reality contradicts MISSION-130M.md or this file (flag renamed, schema changed),
  prefer reality, record the discrepancy in RESULTS-130M.md, and continue — except for
  budget rules and human gates, which always hold.
