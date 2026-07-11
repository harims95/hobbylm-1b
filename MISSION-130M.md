# MISSION-130M.md — HobbyLM 130M Quality-Data Run (end-to-end spec)

**This file is the single source of truth for this project.** It contains the goal, all
settled decisions with reasoning, the exact step-by-step plan, every command, the target
numbers, and hard guardrails. Execute top to bottom. Do not re-litigate decisions marked
LOCKED. Wherever a step needs a human (accounts, money), STOP and ask — marked
`[HUMAN GATE]`. This is a self-contained mission: nothing outside this file and the
HobbyLM repo is required.

---

## 1. GOAL

Train a ~130M-total-parameter sparse-MoE language model on **10B tokens of curated,
high-quality data** using the existing HobbyLM codebase
(https://github.com/harishsg993010/HobbyLM), and **beat the repo's own 130M flagship**
(trained on 10B raw FineWeb tokens) on downstream evals.

This run has two purposes:
1. A complete, publishable small model in its own right.
2. Proof that the curated data recipe works, de-risking a future 1B/100B run at 10× scale.

**Definition of success:**
- Training completes 9,537 steps (~10B tokens) without divergence.
- lm-eval-harness 7-task 0-shot average (hellaswag, openbookqa, winogrande,
  arc_challenge, arc_easy, boolq, piqa; acc_norm where defined) ≥ **44.5**,
  vs the repo baseline **42.97**.
- Total spend ≤ **$100**.

**Non-goals (do NOT do these):**
- Any architecture change. The config is ablation-settled (§3).
- Changing the tokenizer (GPT-2 BPE, vocab 50304, stays).
- Changing seq_len (stays 1024 for this run — comparability with the baseline).
- Touching or regenerating the FineWeb validation shards.
- Beating trillion-token models (SmolLM2 etc.). The claim is per-token efficiency.

---

## 2. CONTEXT — WHAT ALREADY EXISTS IN THE REPO (verified 2026-07-07)

- **Architecture (130M preset, `hobbylm/config.py`):** d_model 512, 12 layers (first
  dense), GQA 8Q/2KV heads, head_dim 64, QK-norm, RoPE θ=10k, 32 experts / top-8 /
  0 shared, expert_ffn 192, dense_ffn 1536, sigmoid gating, `norm_topk_prob=False`,
  aux-loss-free bias balancing + router z-loss, tied embeddings.
  ~140M total / ~62M active params.
- **Optimizer:** Muon (all 2-D/expert matrices; lr 0.02 base) + AdamW (router, embeds,
  lm_head, norms; lr 3e-4 base). Prior flagship used sqrt batch-scaled LR:
  **muon 0.04 / adam 6e-4** at 1.05M-token batches. WSD-style schedule via
  `warmup_steps=100`, `cooldown_frac=0.4`, `final_lr_frac=0.1`.
- **Baseline to beat (docs/ARCHITECTURE_RESEARCH.md §6c, §9):** 130M on 10B raw FineWeb,
  8×H100, 9,537 steps × 1.05M tokens, ~742 ms/step, ~2h wall-clock. Final FineWeb val
  loss **3.3016**. lmeval: hellaswag 32.54, obqa 28.20, wino 52.17, arc_c 23.46,
  arc_e 37.71, boolq 61.31, piqa 65.40 → **avg 42.97**.
- **Throughput flag:** `fused_ce` is a verified free win (−21% peak memory, bit-identical
  loss). **`fp8` is measured-BROKEN (zero-grad backward) — never enable.**
- **Data format (`hobbylm/data.py`):** .bin shards = 256-int32 header
  `[magic=20240520, version=1, num_tokens]` then uint16 GPT-2 tokens; each document
  prepended with EOT (50256). Loader glob-matches a pattern, e.g. `fineweb_train_*.bin`.
- **Training harness:** `training/modal_train.py` on Modal serverless H100s; checkpoints
  and data live on a Modal volume. Eval via `--action lmeval` (lm-eval-harness through
  the repo's `MoELMWrapper`).
- **Companion script (this folder): `prepare_mix10B.py`** — builds the data mix below.
  Its shard writer is already roundtrip-verified byte-identical against
  `hobbylm/data.py::load_shard`.

**Why this mission exists:** the repo's own comparison table shows model ranking sorts by
pretraining data quantity/quality, not parameter count. All prior runs used raw FineWeb.
Curated data + a decay-phase anneal is the highest-leverage improvement available, and it
requires zero architecture changes.

---

## 3. LOCKED DECISIONS (do not revisit)

| # | Decision | Why |
|---|----------|-----|
| D1 | Keep architecture, optimizer, LRs, seq_len 1024, and step count identical to the repo's 130M flagship | Single-variable experiment: only the data changes, so the result is attributable |
| D2 | Data = curated mix (§5): FineWeb-Edu backbone + DCLM + code + math | FineWeb-Edu lifts knowledge/reasoning evals at equal token count; blend beats any single source |
| D3 | Two-phase schedule: main mix for 8,100 steps (85%), then anneal on a high-quality slice for the final 1,437 steps (15%) | SmolLM2/OLMo decay-phase technique; cheap and reliably moves downstream scores |
| D4 | Validation = the repo's ORIGINAL `fineweb_val_*.bin` shards, seed 1337 | Otherwise the 3.3016 comparison is meaningless |
| D5 | Always `--opts fused_ce`; never `fp8` | fused_ce verified free; fp8 verified broken |
| D6 | Success is judged on the lmeval 7-task average; FineWeb val loss is a sanity check only | Mixed-source training shifts FineWeb val loss for distribution reasons unrelated to model quality |
| D7 | Code diffs to the repo must be minimal and additive (new CLI flags, not rewrites), CPU-smoke-tested before any GPU spend | Protect the working baseline stack |

---

## 4. PHASE 0 — SETUP & VERIFICATION

1. Clone and install:
   ```bash
   git clone https://github.com/harishsg993010/HobbyLM.git && cd HobbyLM
   pip install -r requirements.txt  # if absent: pip install torch tiktoken numpy modal datasets tqdm
   ```
2. Copy `prepare_mix10B.py` (companion file) into the repo root.
3. CPU sanity checks (must pass before spending anything):
   ```bash
   python -m hobbylm.count_params --smoke
   python -m pytest tests/ -x -q
   ```
4. Read `training/modal_train.py` end to end. Record: the Modal volume name, how
   `data_dir`/`train_pattern` are set, how checkpoints are saved/resumed, exact CLI flag
   names. **Adapt every command in this file to the real CLI** — commands below use the
   doc's conventions and may need flag-name adjustments.
5. `[HUMAN GATE]` Confirm with the human: Modal account authenticated
   (`modal token new`), HF token available (`HF_TOKEN`), and budget approved (**$100 cap**).
6. Verify the original FineWeb **val** shards exist on the Modal volume (e.g.
   `fineweb_val_*.bin`). If missing, regenerate val shards ONLY via the repo's original
   FineWeb pipeline. Never regenerate under any other circumstance.

---

## 5. PHASE 1 — BUILD THE 10B-TOKEN DATA MIX (~$10–20, CPU only)

`prepare_mix10B.py` streams the datasets, interleaves documents by weight, tokenizes with
GPT-2 BPE (EOT-prepended), and writes 100M-token shards in the exact loader format.

**Main phase mix → `mix_train_*.bin` (9.5B tokens):**

| Source | HF dataset (config) | Tokens | Weight |
|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 6.0B | 63.2% |
| DCLM-baseline | `mlfoundations/dclm-baseline-1.0` | 1.5B | 15.8% |
| Stack-Edu (code) | `HuggingFaceTB/stack-edu` (`python`) | 1.0B | 10.5% |
| FineMath 4+ | `HuggingFaceTB/finemath` (`finemath-4plus`) | 1.0B | 10.5% |

**Anneal slice → `anneal_train_*.bin` (0.5B tokens, kept as separate shards):**
70% `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`) + 30% finemath-4plus.

**Steps:**
1. **Smoke test first (MANDATORY):**
   ```bash
   python prepare_mix10B.py --out data/tiny --scale 0.001
   ```
   This surfaces dataset schema/config-name mismatches in minutes. If a source errors,
   fix its entry in `MAIN_SOURCES`/`ANNEAL_SOURCES` (a `content`-field fallback already
   exists) and rerun. Then decode ~200 tokens from a tiny shard with tiktoken and confirm
   readable text.
2. **Full build on Modal CPU** (tokenizing 10B tokens is CPU-hours — not a laptop job):
   write a minimal Modal wrapper that runs `prepare_mix10B.py --out /data/mix10B`,
   mounting the same volume the trainer reads. Expect several hours; ~20GB output.
3. **Verify:** ~95 `mix_train` shards + ~5 `anneal_train` shards; load 2 random shards
   with `hobbylm.data.load_shard`; decode samples; confirm English / code / math all
   appear. If any source exhausted early (script logs it), backfill the shortfall with
   extra FineWeb-Edu.

---

## 6. PHASE 2 — TRAIN (~$60–80, ~2h on 8×H100)

Mirror the repo's 130M flagship exactly, except the data. Total 9,537 steps at 1.05M
tokens/step, split into two stages.

**Stage A — main mix, steps 0 → 8,100:**
```bash
modal run training/modal_train.py --action train --preset 130M --gpus 8 \
  --steps 8100 --opts fused_ce --batch-tokens 1048576 \
  --data mix10B --run-name 130M_mix10B --save-every 1000
```
(Use sqrt-scaled LRs muon 0.04 / adam 6e-4 — the flagship's values — via whatever
mechanism the CLI provides. seq_len 1024, seed 1337, val on `fineweb_val_*.bin`.)

**Stage B — anneal, steps 8,100 → 9,537:** resume from the step-8,100 checkpoint with the
train pattern switched to `anneal_train_*.bin`. Requirements:
- The LR schedule must CONTINUE (the run is inside `cooldown_frac=0.4` decay), not restart.
- Total step count stays 9,537 so the schedule and baseline comparison line up.
- If the trainer lacks resume-with-different-data support, add minimal flags
  (`--resume <ckpt> --train-pattern <glob>`), per D7: small diff, CPU-test with the tiny
  shards from Phase 1 step 1 before launching on GPUs.

**Monitor:** loss should fall smoothly (QK-norm + Muon gave the baseline zero spikes).
Within the first 15 minutes, verify: (a) step time ≈ 700–800 ms (else something is
misconfigured — kill it), (b) the step-1000 checkpoint saves and can be loaded.
If loss diverges or an expert's routing traffic collapses toward 0%, stop, resume from
the last good checkpoint with `bias_update_rate` halved; `[HUMAN GATE]` if a restart
would push total spend past $100.

---

## 7. PHASE 3 — EVALUATE & JUDGE

1. Run the repo's eval on BOTH checkpoints (final 9,537 AND pre-anneal 8,100 — the pair
   isolates the anneal's contribution):
   ```bash
   modal run training/modal_train.py --action lmeval --run-name 130M_mix10B
   ```
2. Fill the scoreboard (baseline column from the repo's own table):

   | task | baseline (raw FineWeb) | ours pre-anneal | ours final |
   |---|---|---|---|
   | hellaswag | 32.54 | ___ | ___ |
   | openbookqa | 28.20 | ___ | ___ |
   | winogrande | 52.17 | ___ | ___ |
   | arc_challenge | 23.46 | ___ | ___ |
   | arc_easy | 37.71 | ___ | ___ |
   | boolq | 61.31 | ___ | ___ |
   | piqa | 65.40 | ___ | ___ |
   | **average** | **42.97** | ___ | ___ |
   | FineWeb val loss | 3.3016 | ___ | ___ |

3. **Gates:**
   - **G1 (primary): final avg ≥ 44.5** → mission success.
   - **G2:** hellaswag, arc_easy, sciq-class knowledge tasks each improve vs baseline.
   - **G3 (sanity):** FineWeb val loss ≤ 3.45. A modest rise vs 3.3016 is EXPECTED and
     fine (per D6) — it only flags a problem if it blows past 3.45.
   - **G4:** final avg ≥ pre-anneal avg (the anneal helped, or at least didn't hurt).
4. **Outcomes:**
   - G1 pass → write results (step 5), done; the recipe is validated for the 1B run.
   - Avg 43.0–44.5 → partial win; note that the 1B mix should raise FineWeb-Edu to 70%.
   - Avg < 43.0 → STOP. Debug in this order: decode random training shards (is the data
     what we think?), confirm the anneal switch actually happened at 8,100, confirm val
     shards/seed/steps matched the baseline. `[HUMAN GATE]` report findings; no further
     spend.
5. Write `RESULTS-130M.md` in the repo: final config, exact token counts achieved per
   source, the scoreboard, wall-clock, total cost, and any discrepancies between this
   file and reality.
6. Sample generations from the final checkpoint via the repo's `generate.py`
   (capital-of-France class facts, a short code snippet, simple arithmetic) and paste
   3–4 samples into RESULTS-130M.md.
7. Optional (ask first): export GGUF via `export/to_gguf.py` and verify it runs in
   `hobby-rs`; push weights to Hugging Face. `[HUMAN GATE]` before any public upload.

---

## 8. GUARDRAILS — ABSOLUTE RULES

1. **Never** modify or regenerate the FineWeb validation shards (except the missing-val
   recovery case in §4.6).
2. **Never** enable `fp8` or `all_max` — measured-broken in the repo.
3. **Never** change architecture flags: QK-norm, router z-loss, aux-free bias,
   `norm_topk_prob=False`, top_k=8 all stay exactly as the preset defines.
4. **Hard budget cap: $100 total.** Any single job projected over $80, or any restart
   pushing the total past $100 → `[HUMAN GATE]` first.
5. Every repo code change: minimal, additive, CPU-smoke-tested (`count_params --smoke`
   + pytest + a tiny-shard dry run) before any GPU job.
6. Seed 1337 everywhere (training and data mixing).
7. Unique `--run-name` per run; log every job, its cost, and its outcome in
   RESULTS-130M.md as you go.
8. If reality contradicts this file (flag names, dataset schemas, throughput numbers),
   prefer reality, record the discrepancy in RESULTS-130M.md, and continue toward the
   goal — but budget rules and `[HUMAN GATE]`s hold without exception.

---

## 9. BUDGET

| Item | Est. |
|---|---|
| Data prep (Modal CPU, 10B tokens) | $10–20 |
| Training (8×H100, ~2h, two stages) | $60–80 |
| Evals (2 checkpoints) | $5–10 |
| **Total (hard cap $100)** | **~$75–100** |
