# RESULTS-1B

## Phase 1 Notes

- FineWeb-Edu source: use Karpathy's byte-verified GPT-2 shards for the 60B edu slice.
- Non-edu sources: build DCLM, code, math, and anneal on Modal.
- Code source decision: `bigcode/starcoderdata` was gated for the available HF token; `codeparrot/codeparrot-clean` streamed real Python source in `content`, so it became the fallback code source.

## Upload Failure And Fix

The first Phase 1 upload design used `upload_file()` for every shard as soon as it was written. That made every shard upload a separate Hugging Face repository commit. With four builders running in parallel, the job hit HF's repository commit limit:

```text
429 Too Many Requests
You have exceeded the rate limit for repository commits (128 per hour).
```

This was an upload design failure, not a data exhaustion, OOM, or tokenizer/shard-format failure.

Fix: build/resume each source to the Modal volume first, commit the volume, then batch-upload the completed source with `upload_folder()`/source-level upload. Existing uploaded shards remain safe, and resumed builds continue from the next shard index instead of clearing or rebuilding.

Phase 4 implication: do not push every training checkpoint to HF as an individual commit. Use batched, throttled, or volume-first checkpoint backup so the flagship does not hit the same 128-commits/hour limit.

## Vast Shakedown Compile Hang

Status: resolved and GPU-validated. The fully eager workaround was proven first, then the final targeted fix removed only the two compile decorators inside the unused/broken FP8 custom-op path. Whole-model `torch.compile` now trains successfully on 4x H200.

- Date: 2026-07-13.
- Spend: shakedown attempts burned about $29; Vast credit went from about $373.78 to $344.93. Remaining credit was about $345.
- Symptom: pre-first-step hang on both single-GPU and multi-GPU runs. GPU memory was allocated, `nvidia-smi` reported 100% util, but power stayed around 120W, `pmon` showed no real compute, and no first loss appeared.
- Compile sites: the top-level compile is in `training/train.py`; two unnecessary nested compile decorators were inside the FP8 custom ops in `hobbylm/model.py`.
- Interim fix: `--no_compile` made the model fully eager and was proven on one RTX PRO 6000 and four H200s.
- Final fix: remove the two nested FP8 decorators permanently and retain whole-model compilation. The intended `fused_ce=true, fp8_head=false` path does not instantiate or call FP8Linear, so this changes no model parameters, checkpoints, routing, or training math.
- FP8 remains experimental/broken and is not part of the flagship configuration.
- Lesson: kill any hung shakedown within 5 minutes. If power is around 120W and no loss appears, treat it as hung; do not wait 20-30 minutes.
- Final validation: compiled seq_len 1024 training completed multiple 20-step 4x H200 runs with falling loss, high power, finite validation loss, and saved checkpoints.

## Jarvis Labs Path

Status: completed on 2026-07-14 UTC for both the single-GPU and multi-GPU Jarvis compile-fix shakedowns. The runs trained through step 20, wrote checkpoints, and exited cleanly, so the old pre-step-0 compile hang did not reproduce. Jarvis grants went from about `$500` to `$493.90` during the validation cycle.

- Auth: `jl` CLI reads `JL_API_KEY` from the environment (CLI arg > `JL_API_KEY` > `~/.config/jl/config.toml`). Same pattern as Vast's `VAST_API_KEY` — key lives in `.env`/shell env, never in repo files.
- `scripts/jarvis_instance.ps1`: mirrors `vast_instance.ps1` but wraps the `jl` CLI (`gpus`, `create`, `list`, `get`, `ssh`, `pause`, `resume`, `destroy`). Run `-Action gpus` first to confirm the exact `--gpu` identifier string for the RTX PRO 6000 (docs name it "RTX PRO 6000 Blackwell" / "RTX 6000 Pro" but don't publish the flag value); the script defaults to `RTX6000PRO` as a placeholder.
- `scripts/launch_jarvis_train.py`: mirrors `launch_vast_train.py` with two changes — default paths under `/home` instead of `/workspace` (Jarvis only persists `/home` across pause/resume; everything else is wiped), and a `--no-compile` passthrough flag (needed for the shakedown test below; the Vast script didn't expose one).
- Everything else transfers unchanged: `training/train.py`, the data loader, the targeted FP8 decorator removal in `hobbylm/model.py`, `stage_vast_data.py` (HF source is public, just pass `--out /home/data/mix100B --val-out /home/data/fineweb_val`), the `torchrun --standalone` invocation shape, and `backup_vast_checkpoints.py` (pass `--run-dir /home/runs/<run_name>`).
- Shakedown test command (single GPU, 20 steps, eager): see `scripts/launch_jarvis_train.py --nproc-per-node 1 --run-name jarvis_shakedown_test --max-steps 20 --schedule-max-steps 20 --save-every 0 --val-every 20 --no-compile`. Same pass signal as the Vast test: first loss near 10.83, power around 600W, no pre-step-0 hang.
- Jarvis GPU identifier confirmed from `jl gpus`: `RTX-PRO6000` in `IN1`.
- Live shakedown instance: 1x `RTX-PRO6000`, PyTorch template, 50 GB storage, region `IN1`.
- Public branch used for the live test: `https://github.com/harims95/hobbylm-1b.git` on `fix/vast-compile-hang` (that branch name was not published on the `harims95/HobbyLM` remote).
- Staged tiny smoke-test slice under `/home`: `edu_fineweb_train_000001.bin`, `dclm_000001.bin`, `code_000001.bin`, `math_000001.bin`, plus `fineweb_val_000000.bin`.
- Direct live command used on the Jarvis box: `torchrun --standalone --nproc_per_node=1 training/train.py --preset 1B --run_name jarvis_shakedown_test --data_dir /home/data/mix100B --out_dir /home/runs --max_steps 20 --schedule_max_steps 20 --micro_batch_seqs 4 --seq_len 2048 --batch_tokens 1048576 --train_pattern edu_fineweb_train_000001.bin,dclm_000001.bin,code_000001.bin,math_000001.bin --val_pattern /home/data/fineweb_val/fineweb_val_000000.bin --orthogonalizer ns5 --save_every 0 --val_every 20 --stratified_shards --set fused_ce=true --no_compile`.
- First confirmed training signal from the live Jarvis log: `step 0 | loss 12.0116 | 41025ms/step`.
- Follow-up progress check while still running: `step 10 | loss 9.6858 | 40118ms/step`.
- Live health check at `2026-07-14T18:48:37Z`: GPU power `388.37W`, util `98%`, memory `51062 MB`. This is slow single-GPU training with `accum=128`, not the old ~120W no-loss hang.
- Final training outcome from the live Jarvis log: `>> val loss 8.6446 @ step 20`, `=== final val loss 8.6446 ===`, and `saved checkpoint -> /home/runs/jarvis_shakedown_test/model.pt`.
- Final completion check at `2026-07-14T19:00:12Z`: no `torchrun` / `training/train.py` processes remained, GPU power was back to idle at `85.79W`, and the run directory contained `config.json`, `model.pt`, and `result.json`.
- Multi-GPU follow-up gate passed on Jarvis `4x H200` in `IN2` (instance `446419`, resumed as `446421`): `torchrun --standalone --nproc_per_node=4 training/train.py --preset 1B --no_compile ... --stratified_shards --set fused_ce=true` trained cleanly on 4 GPUs with the staged public `/home` shard slice.
- First confirmed 4-GPU training signal: `step 0 | loss 12.0016 | 8700ms/step` with all four H200s at high power (`562.08W`, `572.65W`, `561.18W`, `579.75W`) and about `50.3 GB` memory per GPU, proving the old compile hang is gone in multi-GPU too.
- Follow-up 4-GPU progress check: `step 10 | loss 9.6714 | 6352ms/step`.
- Final 4-GPU outcome: `=== final val loss 8.6390 ===` and `saved checkpoint -> /home/runs/h200_shakedown/model.pt`, with `config.json`, `model.pt`, and `result.json` present in `/home/runs/h200_shakedown`.
- Historical seq_len 2048 cost from the measured 4-GPU H200 shakedown: at `6.35s/step` and `$15.96/hr`, the raw rental burn was about `567 steps/hr` and roughly `$0.028/step` before checkpoint/upload overhead. This is not the flagship projection: D2 now locks the base model to the validated seq_len 1024 recipe, with 2048 deferred to a post-ship v2 experiment.
- Final proof summary:
  - Single GPU `RTX-PRO6000`: loss `12.01 -> 8.64`, about `40s/step`, proven.
  - Multi GPU `4x H200`: loss `12.00 -> 8.64`, about `6.35s/step`, proven.
  - Jarvis credit used: about `$6.10` (`$500 -> $493.90`).
  - Vast credit used earlier during the original failed shakedowns: about `$29` (`$373 -> $345`).
  - The `--no_compile` fix resolves the old pre-step hang in both single-GPU and multi-GPU validation.

## Seq-1024 H200 Optimization Gates

Status: completed on 2026-07-14 UTC on Jarvis 4x H200 (`IN2`) at `$15.96/hr`. Every run used the same 1B preset, 1,048,576 global batch tokens, `fused_ce=true`, the same staged shard slice, 20 training steps, final validation, and a checkpoint save. Step-0 compile/setup time is excluded from the derived steady-step comparison.

| Gate | Configuration | Step 0 | Derived steady step | Memory/GPU | Final val |
|---|---|---:|---:|---:|---:|
| Historical | seq 2048, micro 4, eager, accum 32 | 8.700s | ~6.117s | ~50.3 GB | 8.6390 |
| Test 1 | seq 1024, micro 32, eager, accum 8 | 7.568s | ~4.982s | ~122.0 GB | 8.6922 |
| Test 2 | seq 1024, micro 32, compiled, accum 8 | 31.085s | **~2.477s** | ~76.5 GB | 8.6770 |
| Test 3 | Test 2 + native Flash GQA | 28.541s | ~2.511s | ~72.4 GB | 8.6617 |
| Final micro gate | seq 1024, micro 64, compiled, accum 4 | 35.095s | ~2.414s | ~131.9 GB | 8.7642 |

- Targeted compilation is the major win: micro-32 steady throughput improved from ~4.982s eager to ~2.477s compiled, about 2.0x faster, while preserving falling loss and checkpoint saves.
- Against the original seq-2048 eager shakedown, the final micro-32 compiled path is about 2.47x faster.
- FlashAttention is enabled in PyTorch 2.11.0+cu130. A forced BF16 FlashAttention forward/backward passed on H200 with the model's exact 16-query/8-KV-head GQA shape, seq 1024, head_dim 128, and causal masking.
- Native `enable_gqa=True` saved about 4 GB/GPU but did not improve measured speed, so the physical K/V repeat remains the validated default.
- Micro 64 fit but gained only ~2.6%, used ~132 GB/GPU, and changed aux-free bias update frequency by halving accumulation passes. Its short-run val was worse, so micro 32 remains the safer flagship default.
- Final recommended configuration: `seq_len=1024`, `micro_batch_seqs=32`, `batch_tokens=1048576`, `fused_ce=true`, whole-model compile enabled, physical GQA repeat retained.
- At ~2.477s/step, 95,367 raw training steps project to ~65.6 hours and ~$1,048 on this 4x H200 rental. Budget roughly 69-72 hours and $1,100-$1,150 with validation/checkpoint overhead.
- Jarvis grants after all optimization gates: `$488.55`; all instances were paused after each test and the final account state was 0 running / 2 paused.
