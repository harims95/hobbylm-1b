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

Status: compile-hang fix is committed on `fix/vast-compile-hang` (`d4874cb`) but is not proven yet. Do not merge it into `main` or `feat/1b-flagship` until a real rental shows the model trains.

- Date: 2026-07-13.
- Spend: shakedown attempts burned about $29; Vast credit went from about $373.78 to $344.93. Remaining credit was about $345.
- Symptom: pre-first-step hang on both single-GPU and multi-GPU runs. GPU memory was allocated, `nvidia-smi` reported 100% util, but power stayed around 120W, `pmon` showed no real compute, and no first loss appeared.
- Root cause: there were three `torch.compile` sites, not one. The top-level compile was in `training/train.py`, and two nested `@torch.compile` decorators lived inside the FP8 custom op in `hobbylm/model.py`. The old `--no_compile` skipped only the top-level compile, so the nested FP8 compiles still fired and hung before step 0.
- Fix: `_maybe_compile()` wraps the nested custom-op compiles and is gated by `HOBBYLM_NO_COMPILE=1`, which is set by `--no_compile`. This makes `--no_compile` fully eager.
- Note: the FP8 head path is unused for the intended run (`fused_ce` is used) and was already flagged as broken in review, but its nested compile decorators were still reachable.
- Lesson: kill any hung shakedown within 5 minutes. If power is around 120W and no loss appears, treat it as hung; do not wait 20-30 minutes.
- Next validation: run one cheap single-GPU rental with `--no_compile`. Passing signal is first loss near 10.83 and real training power around 600W. Merge only after that succeeds.

## Jarvis Labs Path

Status: completed on 2026-07-14 UTC for both the single-GPU and multi-GPU Jarvis compile-fix shakedowns. The runs trained through step 20, wrote checkpoints, and exited cleanly, so the old pre-step-0 compile hang did not reproduce. Jarvis grants went from about `$500` to `$493.90` during the validation cycle.

- Auth: `jl` CLI reads `JL_API_KEY` from the environment (CLI arg > `JL_API_KEY` > `~/.config/jl/config.toml`). Same pattern as Vast's `VAST_API_KEY` — key lives in `.env`/shell env, never in repo files.
- `scripts/jarvis_instance.ps1`: mirrors `vast_instance.ps1` but wraps the `jl` CLI (`gpus`, `create`, `list`, `get`, `ssh`, `pause`, `resume`, `destroy`). Run `-Action gpus` first to confirm the exact `--gpu` identifier string for the RTX PRO 6000 (docs name it "RTX PRO 6000 Blackwell" / "RTX 6000 Pro" but don't publish the flag value); the script defaults to `RTX6000PRO` as a placeholder.
- `scripts/launch_jarvis_train.py`: mirrors `launch_vast_train.py` with two changes — default paths under `/home` instead of `/workspace` (Jarvis only persists `/home` across pause/resume; everything else is wiped), and a `--no-compile` passthrough flag (needed for the shakedown test below; the Vast script didn't expose one).
- Everything else transfers unchanged: `training/train.py`, the data loader, the `HOBBYLM_NO_COMPILE` compile fix in `hobbylm/model.py`, `stage_vast_data.py` (HF source is public, just pass `--out /home/data/mix100B --val-out /home/data/fineweb_val`), the `torchrun --standalone` invocation shape, and `backup_vast_checkpoints.py` (pass `--run-dir /home/runs/<run_name>`).
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
- Cost implication from the measured 4-GPU H200 run: at `6.35s/step` and `$15.96/hr`, the raw rental burn is about `567 steps/hr` and roughly `$0.028/step` before checkpoint/upload overhead. Seq-len choice remains a [HUMAN GATE] decision and is intentionally not committed here.
- Final proof summary:
  - Single GPU `RTX-PRO6000`: loss `12.01 -> 8.64`, about `40s/step`, proven.
  - Multi GPU `4x H200`: loss `12.00 -> 8.64`, about `6.35s/step`, proven.
  - Jarvis credit used: about `$6.10` (`$500 -> $493.90`).
  - Vast credit used earlier during the original failed shakedowns: about `$29` (`$373 -> $345`).
  - The `--no_compile` fix resolves the old pre-step hang in both single-GPU and multi-GPU validation.
