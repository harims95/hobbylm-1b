"""
prepare_mix100B.py - build Phase 1 non-edu shards for the 1B run.

FineWeb-Edu comes from byte-verified Karpathy shards. This script builds only:
  dclm_*.bin   -> 15B tokens
  code_*.bin   -> 10B tokens (requires a text-bearing source)
  math_*.bin   -> 10B tokens
  anneal_*.bin -> 5B tokens

Builds resume from existing local/HF shards and upload each completed source in one batch.
"""
from __future__ import annotations

import argparse
import math
import os
import re

import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

MAGIC, VERSION = 20240520, 1
SHARD_TOKENS = 100_000_000
BATCH_DOCS = 500
REPO_ID = "harims95/hobbylm-mix100b-gpt2"
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token

MAIN_SOURCES = {
    "dclm": ("mlfoundations/dclm-baseline-1.0", None, "text", 15_000_000_000),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus", "text", 10_000_000_000),
}

ANNEAL_PARTS = [
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 3_500_000_000),
    ("HuggingFaceTB/finemath", "finemath-4plus", "text", 1_500_000_000),
]

CODE_PARTS = [
    ("codeparrot/codeparrot-clean", None, "content", 10_000_000_000),
]


def doc_stream(dataset: str, config: str | None, field: str):
    ds = load_dataset(dataset, config, split="train", streaming=True)
    for row in ds:
        text = row.get(field) or row.get("content") or row.get("text") or ""
        if text and len(text) > 80:
            yield text


def tokenize_batch(texts: list[str]) -> list[np.ndarray]:
    out = []
    encoded = ENC.encode_ordinary_batch(texts)
    for toks in encoded:
        arr = np.asarray([EOT, *toks], dtype=np.uint32)
        assert (arr < 2**16).all(), "token id out of uint16 range"
        out.append(arr.astype(np.uint16))
    return out


def prefix_files(out_dir: str, prefix: str) -> list[str]:
    if not os.path.isdir(out_dir):
        return []
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{6}})\.bin$")
    return sorted(name for name in os.listdir(out_dir) if pat.match(name))


def local_prefix_stats(out_dir: str, prefix: str) -> tuple[int, int]:
    total = 0
    next_idx = 1
    for name in prefix_files(out_dir, prefix):
        path = os.path.join(out_dir, name)
        header = np.fromfile(path, dtype=np.int32, count=256)
        if len(header) != 256 or int(header[0]) != MAGIC or int(header[1]) != VERSION:
            raise RuntimeError(f"bad shard header: {path}")
        total += int(header[2])
        next_idx = max(next_idx, int(name.split("_")[-1].split(".")[0]) + 1)
    return total, next_idx


def remote_prefix_stats(repo_id: str, prefix: str) -> tuple[int, int]:
    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{6}})\.bin$")
    files = sorted(f for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset") if pat.match(f))
    if not files:
        return 0, 1
    max_idx = max(int(pat.match(f).group(1)) for f in files)
    return len(files) * SHARD_TOKENS, max_idx + 1


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, shard_idx_start: int,
                 shard_tokens: int = SHARD_TOKENS):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.n = 0
        self.shard_idx = shard_idx_start
        self.total = 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, toks: np.ndarray):
        i = 0
        while i < len(toks):
            take = min(len(toks) - i, self.shard_tokens - self.n)
            self.buf[self.n:self.n + take] = toks[i:i + take]
            self.n += take
            i += take
            if self.n == self.shard_tokens:
                self.flush()

    def flush(self):
        if self.n == 0:
            return
        header = np.zeros(256, dtype=np.int32)
        header[0], header[1], header[2] = MAGIC, VERSION, self.n
        fname = f"{self.prefix}_{self.shard_idx:06d}.bin"
        path = os.path.join(self.out_dir, fname)
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(self.buf[:self.n].tobytes())
        self.total += self.n
        print(f"WROTE {path} ({self.n:,} tokens)", flush=True)
        self.shard_idx += 1
        self.n = 0

    def close(self):
        self.flush()


def upload_source_folder(out_dir: str, prefix: str, repo_id: str):
    files = prefix_files(out_dir, prefix)
    if not files:
        raise RuntimeError(f"no local {prefix}_*.bin files to upload from {out_dir}")
    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print(f"BATCH_UPLOAD {prefix}: {len(files)} local shard(s) -> {repo_id}", flush=True)
    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{prefix}_*.bin", f".{prefix}.done"],
        commit_message=f"upload {prefix} shards",
    )
    print(f"BATCH_UPLOADED {prefix}: {len(files)} shard(s) -> {repo_id}", flush=True)


def build_from_streams(out_dir: str, prefix: str, streams: list[tuple[str, object, int]],
                       budget: int, repo_id: str, upload: bool):
    local_total, local_next = local_prefix_stats(out_dir, prefix)
    remote_total, remote_next = remote_prefix_stats(repo_id, prefix)
    resume_tokens = max(local_total, remote_total)
    next_idx = max(local_next, remote_next)
    if resume_tokens > budget:
        raise RuntimeError(f"{prefix} existing tokens {resume_tokens:,} exceed budget {budget:,}")

    print(
        f"RESUME {prefix}: local={local_total:,} remote={remote_total:,} "
        f"resume={resume_tokens:,} next_idx={next_idx:06d}",
        flush=True,
    )
    writer = ShardWriter(out_dir, prefix, next_idx)
    pbar = tqdm(total=budget, initial=resume_tokens, unit="tok", unit_scale=True, desc=prefix)
    count = resume_tokens
    seen = 0
    for label, stream, part_budget in streams:
        local = 0
        print(f"OPEN {prefix}:{label} budget={part_budget:,}", flush=True)
        while local < part_budget and count < budget:
            texts = []
            for _ in range(BATCH_DOCS):
                try:
                    texts.append(next(stream))
                except StopIteration:
                    break
            if not texts:
                print(f"EXHAUSTED {prefix}:{label} after {local:,} tokens", flush=True)
                break
            for toks in tokenize_batch(texts):
                ntok = len(toks)
                start = 0
                if seen + ntok <= resume_tokens:
                    seen += ntok
                    local += ntok
                    continue
                if seen < resume_tokens:
                    start = resume_tokens - seen
                seen += ntok
                local += ntok
                piece = toks[start:]
                if len(piece) > budget - count:
                    piece = piece[:budget - count]
                writer.add(piece)
                count += len(piece)
                pbar.update(len(piece))
                if local >= part_budget or count >= budget:
                    break
    pbar.close()
    writer.close()
    min_ok = math.ceil(0.9 * budget)
    if count < min_ok:
        raise RuntimeError(f"{prefix} underfilled: got {count:,} tokens, need at least {min_ok:,} / {budget:,}")
    with open(os.path.join(out_dir, f".{prefix}.done"), "w", encoding="utf-8") as f:
        f.write(str(count))
    print(f"DONE {prefix}: {count:,} / {budget:,} tokens", flush=True)
    if upload:
        upload_source_folder(out_dir, prefix, repo_id)


def build_single(out_dir: str, source: str, repo_id: str, upload: bool):
    dataset, config, field, budget = MAIN_SOURCES[source]
    streams = [(source, doc_stream(dataset, config, field), budget)]
    build_from_streams(out_dir, source, streams, budget, repo_id, upload)


def build_anneal(out_dir: str, repo_id: str, upload: bool):
    streams = [
        (f"{dataset}:{config}", doc_stream(dataset, config, field), budget)
        for dataset, config, field, budget in ANNEAL_PARTS
    ]
    build_from_streams(out_dir, "anneal", streams, sum(b for *_, b in ANNEAL_PARTS), repo_id, upload)


def build_code(out_dir: str, repo_id: str, upload: bool):
    streams = [
        (dataset, doc_stream(dataset, config, field), budget)
        for dataset, config, field, budget in CODE_PARTS
    ]
    build_from_streams(out_dir, "code", streams, sum(b for *_, b in CODE_PARTS), repo_id, upload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/mix100B")
    ap.add_argument("--source", choices=["dclm", "code", "math", "anneal"], required=True)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--upload-only", action="store_true")
    args = ap.parse_args()

    if args.upload_only:
        upload_source_folder(args.out, args.source, args.repo_id)
        return
    upload = not args.skip_upload
    if args.source == "anneal":
        build_anneal(args.out, args.repo_id, upload)
    elif args.source == "code":
        build_code(args.out, args.repo_id, upload)
    else:
        build_single(args.out, args.source, args.repo_id, upload)


if __name__ == "__main__":
    main()
