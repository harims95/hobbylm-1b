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
