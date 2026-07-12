# MISSION-1B.md — HobbyLM 1B/100B Flagship (end-to-end spec)

**Single source of truth for the 1B run.** Goal, locked decisions with reasoning, phases,
exact commands, gates, budget, and guardrails. Execute top to bottom. Do not re-litigate
LOCKED decisions. Wherever money, accounts, or irreversible actions are involved, STOP and
ask — marked `[HUMAN GATE]`. Prefer reality over this file if a flag/schema/number differs;
record the discrepancy in RESULTS-1B.md, but never bypass a `[HUMAN GATE]` or budget cap.

Read `CLAUDE.md` (or `AGENTS.md`) first — the Karpathy-style engineering discipline applies
to every step here, especially "neural-net code fails silently."

---

## 1. GOAL

Train the strongest ~1B-total-parameter sparse-MoE model we can on 100B tokens of our
curated mix, for roughly $600–900 cash (after brother's Vast credits), building on the
HobbyLM codebase and everything proven in the 130M runs.

**Definition of success:**
- Completes 100B tokens (main + anneal) without divergence or expert collapse.
- lm-eval 7-task 0-shot avg (hellaswag, openbookqa, winogrande, arc_challenge, arc_easy,
  boolq, piqa; acc_norm where defined) beats **pythia-410m (45.34)** and ideally clears
  **gemma-3-270m (48.09)** and **SmolLM2-135M (49.34)**.
- Honest public claim we can defend: *best open sub-1B model per training token / per
  dollar.* Not "beats Qwen3-0.6B / SmolLM2-360M outright" — they used 20–360× our tokens.

**Non-goals:** matching trillion-token models on absolute MMLU; architecture research;
long-context/multimodal; anything that breaks the one-variable-at-a-time discipline within
a phase.

---

## 2. WHAT'S ALREADY PROVEN (from the 130M runs — don't re-derive)

- **Data recipe works.** 130M on our 5B mix + anneal scored ~43.3 avg vs Harish's 42.97
  baseline — on HALF the tokens (5B vs 10B). The mix is more token-efficient; that's the
  property that scales.
- **The loader shuffle fix is essential.** v1 (sequential shards) → catastrophic
  forgetting, 40.70. v2 (seeded global shuffle, commit `ac55e3a`) → 43.3. The fix must be
  active here too — verify it.
- **Anneal phase helps.** Stage A only ~42.7 → +anneal ~43.3. Keep it.
- **fused_ce is free; fp8 is broken.** Settled.
- **Pipeline mechanics work:** shard writer format, resume-with-different-pattern,
  schedule_max_steps spanning stages, checkpoint/resume, lmeval — all validated.
- **Data-build failure modes seen & fixed:** stack-edu flaky (swapped to the-stack-smol +
  smollm-corpus python-edu); HF rate-limiting (pass HF_TOKEN as a secret); a source
  underfilling (the >90%-budget guard before writing `.done`).

---

## 3. LOCKED DECISIONS

| # | Decision | Why |
|---|----------|-----|
| D1 | Architecture = brother's 1B preset, unchanged except D2/D3 | Ablation-settled; our edge is data, not architecture |
| D2 | seq_len 2048 (up from 1024) | 1024 caps evals and real usability; 2048 safe at RoPE θ=10k |
| D3 | Consider top_k 8→12 ONLY if the Phase 2 ablation says so | 1B preset is ~7× sparse (~140M active); evals compare vs dense 360–600M. Prior ablation favored more active experts. Gated on evidence, not vibes |
| D4 | **Tokenizer = GPT-2 (vocab 50,304), NOT SmolLM2** | Phase 0.3 measured only 4.19% fewer tokens with SmolLM2 on a ~200MB mix sample, below the 5% threshold. Keep 130M comparability and avoid vocab/eval/Rust/GGUF churn. IDs still fit uint16 |
| D5 | Data = 100B curated mix: 60% FineWeb-Edu, 15% DCLM, 10% code (Stack-Edu, multi-language), 10% FineMath, 5% anneal | 10× scale of the validated 5B mix; code source upgraded to Stack-Edu (see §5a) |
| D6 | Fixed seeded-shuffle loader (commit `ac55e3a`), verified before spend | The single most expensive bug we know of |
| D7 | Two-phase training: main 85% + anneal 15% | Validated at 130M |
| D8 | Data build + backup + evals + SFT on **Modal**; **flagship training on Vast** | Modal convenient/cheap for small jobs; Vast ~35% cheaper for the one big job + brother's $370 credit |
| D9 | Every shard uploaded to a HF dataset repo as it's built | Backup + data-transfer bridge to Vast + reproducibility |
| D10 | A cheap 130M shakedown run ON VAST before the flagship | De-risk the SSH/torchrun/resume path before betting $600 on it |
| D11 | Always `--opts fused_ce`; never fp8 | Measured |
| D12 | Judge on lm-eval avg; val loss is sanity-only | Val loss still depends on data/order, but GPT-2 keeps tokenizer comparability with the 130M runs |

---

## 4. PHASE 0 — PREP & DECISIONS TO CONFIRM

1. Confirm the 130M validation is accepted as passed (data recipe green-lit). If not, stop.
2. `[HUMAN GATE]` Confirm budgets & accounts:
   - Modal: your account (data build, evals, SFT) — est. ~$120 needed.
   - Vast: brother's account, ~$370 credit + ability to add ~$250 cash if needed.
   - HF: token with write access (for the shard backup repo).
3. **Tokenizer compression sanity check complete:** GPT-2 vs SmolLM2 on the same ~200MB
   sample measured only a 4.19% SmolLM2 token-count reduction, below the 5% threshold.
   Decision: keep GPT-2 for D4.
4. No tokenizer code migration is needed for Phase 1:
   - Keep `prepare_mix10B.py` on GPT-2/tiktoken.
   - Keep model/eval/generate vocab size and EOT handling unchanged.
   - Keep Rust engine / GGUF export assumptions unchanged.

---

## 5. PHASE 1 — BUILD & BACK UP THE 100B DATA (Modal, ~$10 + upload, ~8–9h parallel)

1. Fix the two known issues up front: add `HF_TOKEN` as a Modal secret (rate limits) and
   keep the >90%-budget `.done` guard.
2. **Parallel build** — one Modal CPU function per source, running simultaneously
   (FineWeb-Edu is the ~8h bottleneck; others finish sooner). For speed, optionally split
   FineWeb-Edu into 4 sub-workers over dataset slices → whole build ~2–3h. cpu=4,
   memory=32768, timeout=86400, **detached**.
   - FineWeb-Edu 60B (switch config from `sample-10BT` to full / `sample-100BT`)
   - DCLM 15B, code 10B (the-stack-smol + smollm-corpus python-edu), FineMath 10B
   - anneal 5B (cosmopedia-v2 70% + finemath-4plus 30%), separate shards
   - If a source exhausts early, backfill the shortfall from FineWeb-Edu; log it.
3. **Upload each shard to HF as it's written** (D9): private repo
   `harims95/hobbylm-mix100b-gpt2`. Non-blocking/resumable — a failed upload logs and
   continues, never crashes the build; track uploaded shards in a manifest. README:
   tokenizer (GPT-2, 50,304), shard format, mix ratios, per-source token counts, seed 1337.
4. **Verify (free):** total ≈ 200GB / ~1,000 shards; decode ~200 tokens each from an edu,
   code, and math shard (English/code/math readable); run the shuffle-verification script
   over the 1B patterns — first 40 shards must be interleaved, not grouped.

### 5a. CODE SOURCE — Stack-Edu, done right (10B tokens)

At 5B we fell back to the-stack-smol (a small *sample* set — Python alone only had ~274M
tokens, forcing a second source). For the 1B, upgrade to **Stack-Edu**
(`HuggingFaceTB/stack-edu`): the educational-quality filtered slice of The Stack v2, the
exact code source SmolLM2 used, and it ships text directly (no Software Heritage / S3
download layer — that's why we're NOT using raw The Stack v2, which only ships blob
pointers and would add a whole download tier).

**Language mix (10B total): ~70% Python, ~30% spread across JavaScript, Java, C++, Go.**
Multiple languages avoid exhausting any single split (the Python-underfill trap from 5B)
and add cheap diversity. Our evals test reasoning, not coding, so Python-heavy is fine.

**MANDATORY streaming fix before spending (the 5B failure was config/field naming, not a
real limitation):**
1. Stack-Edu is configured per-language. Confirm the EXACT config names by listing the
   dataset's configs, then stream each language as its own config (e.g. `python`,
   `javascript`, `java`, `cpp`, `go` — verify spelling/casing; casing bit us before, e.g.
   `Python` vs `python`).
2. Before launching any Modal job: stream 3 docs from EACH chosen language config and print
   the row keys + first 200 chars. Confirm the text field name (fallback to `content`
   already handled). Only launch once every language decodes cleanly.
3. Build each language as its own sub-source with its own token budget (e.g. Python 7B; the
   other four ~0.75B each) and the >90%-budget `.done` guard per language. If one language
   underfills, backfill from Python (largest split), and log it.
4. Write all code languages into the shared `code_*.bin` family so the loader's shard-level
   shuffle mixes them with edu/dclm/math exactly as validated at 130M.

---

## 6. PHASE 2 — OPTIONAL top_k ABLATION (Modal, ~$40, only if pursuing D3)

One 130M-scale run on the new GPT-2-tokenized mix, single knob: top_k 8 vs 12, ~3,000
steps, same val protocol. If top_k=12 improves val ≥0.01 at equal steps, adopt it for the
flagship (active ~140M→~190M; budget the flagship up to the higher end). Else keep top_k=8.
Skip entirely if budget is tight — it's optimization, not a requirement.

---

## 7. PHASE 3 — VAST SHAKEDOWN (Vast, ~$5, MANDATORY before flagship) [HUMAN GATE to spend]

Prove the Vast execution path cheaply before the $600 job rides on it.

1. **Pick a host** (verified datacenter, on-demand, reliability ≥99.5%, max-duration ≥1
   month, 4×H100 SXM on ONE machine, disk ≥500GB NVMe). Never interruptible for the
   flagship; never split across two hosts (no InfiniBand on Vast marketplace).
2. `vastai` CLI: create SSH key, search/filter, `create instance`, SSH in, `tmux`.
3. Environment: clone the fork (`feat/curated-data-mix-130m` + 1B mission updates), install
   deps (Docker image or pip), download a SMALL slice of shards from the HF backup (D9) to
   the box.
4. Launch a tiny run with **plain torchrun** (not modal_train.py):
   `torchrun --nproc_per_node=4 training/train.py --preset 130M --max-steps 200 ...`
   with checkpoints every 50 steps to a persistent location (Vast volume or push to HF).
5. **Shakedown gates:** first loss ≈ ln(50304) ≈ 10.8; loss falls; checkpoint saves AND
   resumes after a deliberate kill; step time sane. If all pass, the Vast path is trusted.
   If not, fix on the cheap run — never debug on the flagship.
6. Destroy the shakedown instance (stop paying).

---

## 8. PHASE 4 — THE FLAGSHIP (Vast, ~$600–850, ~45h) [HUMAN GATE before launch]

1. **Recompute the run:** 100B tokens ÷ (batch tokens/step) = total steps; main = 85%,
   anneal = 15%; `schedule_max_steps` = total. At seq 2048, re-probe throughput with a
   short speed test on the rented host BEFORE committing — do not trust the 1024 numbers.
   Adjust micro-batch down if OOM (fused_ce headroom helps).
2. `[HUMAN GATE]` Present measured ms/step @2048, projected wall-clock, projected cost
   (cash + credit split), and confirm the host meets all Phase 3 criteria. Get explicit go.
3. **Rent the flagship host** (same criteria as Phase 3; verified/on-demand). Download the
   full 200GB shard set from HF backup to the box. Verify shard count + a decode sample.
4. **Sanity run first (~$3):** 200 steps attached in tmux; confirm loss ~10.8 falling,
   checkpoint resume works, step time matches the probe. Only then the real run.
5. **Launch main phase, detached in tmux, torchrun 4×H100:** train pattern = all main-mix
   shards SHUFFLED (loader fix), val on a held slice, `schedule_max_steps` = total,
   `fused_ce` on, **checkpoint every 15–30 min** to a persistent Vast volume AND pushed to
   HF (marketplace hosts can vanish). Log run-name, config, start time to RESULTS-1B.md.
6. **Monitor** (SSH, every few hours): loss smooth (Muon+QK-norm → no spikes); expert load
   balanced (no expert at ~0% traffic); run lmeval on intermediate checkpoints every ~10%
   of steps — the avg should climb. **Standing rules:** on crash/host-loss, resume from the
   latest good checkpoint; **max 2 auto-resumes**, then STOP and wait for human. Divergence
   or expert collapse → stop, resume with `bias_update_rate` halved. `[HUMAN GATE]` if any
   restart burns >$50.
7. **Anneal phase (final 15%):** resume from the main-phase-end checkpoint with train
   pattern = anneal shards only, continue to the end. LR continues its cooldown (don't
   restart); bias freeze near the end stays on.

---

## 9. PHASE 5 — EVAL, POST-TRAIN, PUBLISH

1. Final lmeval on the flagship (Modal or Vast). Refresh reference models via `lmeval_hf`
   on the identical 7-task 0-shot protocol.
2. Scoreboard in RESULTS-1B.md:

   | model | tokens | 7-task avg | target |
   |---|---|---|---|
   | OURS-1B (100B, GPT-2 tok) | 100B | ___ | beat 45.34 / aim 48–49 |
   | pythia-410m | 300B | 45.34 | beat ✅? |
   | gemma-3-270m | 6T | 48.09 | beat ✅? |
   | SmolLM2-135M | 2T | 49.34 | beat ✅? |
   | SmolLM2-360M | 4T | 56.29 | stretch (not expected) |

3. **Post-training (Modal, ~$100, separate `[HUMAN GATE]`):** SFT on a SmolTalk-style mix
   + one DPO round, reusing the repo's chat-tuning path → publish a `-Chat` variant. Base
   results ship first regardless.
4. Export GGUF (GPT-2 vocab), verify it loads/generates in the Rust engine. Sanity
   prompts: a fact, a short code completion, a simple math word problem.
5. `[HUMAN GATE]` before any public upload. Then flip the HF data repo public (D9), push
   weights + a results README (scoreboard, recipe, cost, per-token-efficiency framing).

---

## 10. GUARDRAILS — ABSOLUTE

1. Verify the loader shuffle fix (Phase 1.4) before any GPU spend. Never run the sequential
   loader.
2. Never enable fp8/all_max. Never remove QK-norm, router z-loss, aux-free bias.
3. Never launch a job projected >$50 without a `[HUMAN GATE]`. Flagship needs its own gate.
4. Flagship on Vast: verified datacenter + on-demand + reliability ≥99.5% + single machine.
   Never interruptible, never multi-host.
5. Checkpoint every 15–30 min on the flagship; push checkpoints off-box (HF/persistent
   volume). Verify resume works within the first hour.
6. Max 2 auto-resumes on any failure, then STOP for a human.
7. Minimal additive code diffs, CPU-smoke-tested before GPU spend. Seed 1337 everywhere.
8. Log every run (name, config, cost, outcome) in RESULTS-1B.md as you go.
9. One variable at a time within a phase. If reality contradicts this file, prefer reality,
   log it, continue — but budget rules and `[HUMAN GATE]`s are absolute.

---

## 11. BUDGET

| Phase | Item | Est. |
|---|---|---|
| 1 | 100B data build + HF backup (Modal CPU) | ~$10 + storage |
| 2 | Optional top_k ablation (Modal) | ~$40 |
| 3 | Vast shakedown (130M, ~$5) | ~$5 |
| 4 | Flagship (Vast, ~45h, 4×H100) | ~$600–850 |
| 5 | Evals + GGUF (Modal) | ~$30 |
| 5 | Post-train SFT+DPO (Modal, optional) | ~$100 |
| — | HF storage (200GB, few months) | ~$0–20 |
| — | Buffer (~10%) | ~$100 |
| **Total** | | **~$900–1,150 gross; ~$530–780 cash after brother's $370 Vast credit** |

Hard cap without new human approval: **$1,200 gross.**
