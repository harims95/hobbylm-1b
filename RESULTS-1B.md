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
