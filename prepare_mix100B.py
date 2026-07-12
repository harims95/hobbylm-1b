"""
prepare_mix100B.py - build Phase 1 non-edu shards for the 1B run.

FineWeb-Edu comes from byte-verified Karpathy shards. This script builds only:
  dclm_*.bin   -> 15B tokens
  code_*.bin   -> 10B tokens (requires a text-bearing source)
  math_*.bin   -> 10B tokens
  anneal_*.bin -> 5B tokens

Every flushed shard is uploaded to harims95/hobbylm-mix100b-gpt2 immediately.
"""
from __future__ import annotations

import argparse
import math
import os

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
    ("Python", 7_000_000_000),
    ("JavaScript", 750_000_000),
    ("Java", 750_000_000),
    ("Cpp", 750_000_000),
    ("Go", 750_000_000),
]


def doc_stream(dataset: str, config: str | None, field: str):
    ds = load_dataset(dataset, config, split="train", streaming=True)
    for row in ds:
        text = row.get(field) or row.get("content") or row.get("text") or ""
        if text and len(text) > 80:
            yield text


def stack_edu_stream(config: str):
    ds = load_dataset("HuggingFaceTB/stack-edu", config, split="train", streaming=True)
    for row in ds:
        text = row.get("content") or row.get("text") or row.get("source") or ""
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


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, repo_id: str, shard_tokens: int = SHARD_TOKENS):
        self.out_dir = out_dir
        self.prefix = prefix
        self.repo_id = repo_id
        self.shard_tokens = shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.n = 0
        self.shard_idx = 1
        self.total = 0
        self.api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
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
        self.api.upload_file(
            path_or_fileobj=path,
            path_in_repo=fname,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        print(f"UPLOADED {fname} -> {self.repo_id}", flush=True)
        self.shard_idx += 1
        self.n = 0

    def close(self):
        self.flush()


def clear_prefix(out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.startswith(prefix + "_") and name.endswith(".bin"):
            os.remove(os.path.join(out_dir, name))
    marker = os.path.join(out_dir, f".{prefix}.done")
    if os.path.exists(marker):
        os.remove(marker)


def build_from_streams(out_dir: str, prefix: str, streams: list[tuple[str, object, int]], budget: int, repo_id: str):
    clear_prefix(out_dir, prefix)
    writer = ShardWriter(out_dir, prefix, repo_id)
    pbar = tqdm(total=budget, unit="tok", unit_scale=True, desc=prefix)
    count = 0
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
                writer.add(toks)
                local += len(toks)
                count += len(toks)
                pbar.update(len(toks))
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


def build_single(out_dir: str, source: str, repo_id: str):
    dataset, config, field, budget = MAIN_SOURCES[source]
    streams = [(source, doc_stream(dataset, config, field), budget)]
    build_from_streams(out_dir, source, streams, budget, repo_id)


def build_anneal(out_dir: str, repo_id: str):
    streams = [
        (f"{dataset}:{config}", doc_stream(dataset, config, field), budget)
        for dataset, config, field, budget in ANNEAL_PARTS
    ]
    build_from_streams(out_dir, "anneal", streams, sum(b for *_, b in ANNEAL_PARTS), repo_id)


def build_code(out_dir: str, repo_id: str):
    streams = [(lang, stack_edu_stream(lang), budget) for lang, budget in CODE_PARTS]
    build_from_streams(out_dir, "code", streams, sum(b for _, b in CODE_PARTS), repo_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/mix100B")
    ap.add_argument("--source", choices=["dclm", "code", "math", "anneal"], required=True)
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    if args.source == "anneal":
        build_anneal(args.out, args.repo_id)
    elif args.source == "code":
        build_code(args.out, args.repo_id)
    else:
        build_single(args.out, args.source, args.repo_id)


if __name__ == "__main__":
    main()
