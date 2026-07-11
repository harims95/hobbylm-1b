"""
prepare_mix10B.py - build streaming shards for the 130M run, one source at a time.

Output sets for scale=0.5:
  edu_*.bin    -> 3.0B tokens
  dclm_*.bin   -> 0.75B tokens
  code_*.bin   -> 0.50B tokens
  math_*.bin   -> 0.50B tokens
  anneal_*.bin -> 0.25B tokens (70% cosmopedia, 30% finemath-4plus), sequentially written
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

MAGIC, VERSION = 20240520, 1
SHARD_TOKENS = 100_000_000
BATCH_DOCS = 500
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token

MAIN_SOURCES = {
    "edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 6.0e9),
    "dclm": ("mlfoundations/dclm-baseline-1.0", None, "text", 1.5e9),
    "code": ("codeparrot/codeparrot-clean", None, "content", 1.0e9),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus", "text", 1.0e9),
}
ANNEAL_PARTS = [
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 0.35e9),
    ("HuggingFaceTB/finemath", "finemath-4plus", "text", 0.15e9),
]


def doc_stream(dataset, config, field):
    ds = load_dataset(dataset, config, split="train", streaming=True)
    for row in ds:
        text = row.get(field) or row.get("content") or ""
        if text and len(text) > 80:
            yield text


def tokenize_batch(texts: list[str]) -> list[np.ndarray]:
    out = []
    for text in texts:
        toks = [EOT] + ENC.encode_ordinary(text)
        arr = np.asarray(toks, dtype=np.uint32)
        assert (arr < 2**16).all(), "token id out of uint16 range"
        out.append(arr.astype(np.uint16))
    return out


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, shard_tokens: int = SHARD_TOKENS, shard_idx_start: int = 1):
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
                self._flush()

    def _flush(self):
        if self.n == 0:
            return
        header = np.zeros(256, dtype=np.int32)
        header[0], header[1], header[2] = MAGIC, VERSION, self.n
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.shard_idx:06d}.bin")
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(self.buf[:self.n].tobytes())
        self.total += self.n
        print(f"  wrote {path} ({self.n:,} tokens)", flush=True)
        self.shard_idx += 1
        self.n = 0

    def close(self):
        self._flush()


def clear_prefix(out_dir: str, prefix: str):
    for name in os.listdir(out_dir):
        if name.startswith(prefix + "_") and name.endswith(".bin"):
            os.remove(os.path.join(out_dir, name))


def done_marker(out_dir: str, prefix: str) -> str:
    return os.path.join(out_dir, f".{prefix}.done")


def existing_prefix_stats(out_dir: str, prefix: str) -> tuple[int, int]:
    total = 0
    next_idx = 1
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith(prefix + "_") and name.endswith(".bin")):
            continue
        path = os.path.join(out_dir, name)
        header = np.fromfile(path, dtype=np.int32, count=256)
        total += int(header[2])
        try:
            shard_idx = int(name.split("_")[-1].split(".")[0])
            next_idx = max(next_idx, shard_idx + 1)
        except ValueError:
            pass
    return total, next_idx


def build_single(out_dir: str, prefix: str, dataset: str, config: str | None, field: str, budget: int):
    clear_prefix(out_dir, prefix)
    writer = ShardWriter(out_dir, prefix)
    stream = doc_stream(dataset, config, field)
    pbar = tqdm(total=budget, unit="tok", unit_scale=True, desc=prefix)
    count = 0
    while count < budget:
        texts = []
        for _ in range(BATCH_DOCS):
            try:
                texts.append(next(stream))
            except StopIteration:
                break
        if not texts:
            break
        for toks in tokenize_batch(texts):
            writer.add(toks)
            count += len(toks)
            pbar.update(len(toks))
            if count >= budget:
                break
    pbar.close()
    writer.close()
    min_ok = math.ceil(0.9 * budget)
    if count < min_ok:
        raise RuntimeError(f"{prefix} underfilled: got {count:,} tokens, need at least {min_ok:,} / {budget:,}")
    with open(done_marker(out_dir, prefix), "w", encoding="utf-8") as f:
        f.write(str(count))
    print(f"{prefix}: {count:,} / {budget:,} tokens", flush=True)


def build_code(out_dir: str, scale: float):
    dataset, config, field, budget_raw = MAIN_SOURCES["code"]
    target_total = int(budget_raw * scale)
    min_total = math.ceil(0.9 * target_total)
    existing_total, next_idx = existing_prefix_stats(out_dir, "code")
    if existing_total >= min_total:
        with open(done_marker(out_dir, "code"), "w", encoding="utf-8") as f:
            f.write(str(existing_total))
        print(f"code: existing shards already satisfy guard ({existing_total:,} tokens)", flush=True)
        return

    writer = ShardWriter(out_dir, "code", shard_idx_start=next_idx)
    stream = doc_stream(dataset, config, field)
    pbar = tqdm(total=target_total, initial=existing_total, unit="tok", unit_scale=True, desc="code")
    count = existing_total
    while count < target_total:
        texts = []
        for _ in range(BATCH_DOCS):
            try:
                texts.append(next(stream))
            except StopIteration:
                break
        if not texts:
            break
        for toks in tokenize_batch(texts):
            writer.add(toks)
            count += len(toks)
            pbar.update(len(toks))
            if count >= target_total:
                break
    pbar.close()
    writer.close()
    if count < min_total:
        raise RuntimeError(f"code underfilled: got {count:,} total tokens, need at least {min_total:,} / {target_total:,}")
    with open(done_marker(out_dir, "code"), "w", encoding="utf-8") as f:
        f.write(str(count))
    print(f"code: {count:,} / {target_total:,} tokens", flush=True)


def build_anneal(out_dir: str, prefix: str, scale: float):
    clear_prefix(out_dir, prefix)
    writer = ShardWriter(out_dir, prefix)
    total_budget = int((0.35e9 + 0.15e9) * scale)
    pbar = tqdm(total=total_budget, unit="tok", unit_scale=True, desc=prefix)
    count = 0
    for dataset, config, field, budget_raw in ANNEAL_PARTS:
        budget = int(budget_raw * scale)
        stream = doc_stream(dataset, config, field)
        local = 0
        while local < budget:
            texts = []
            for _ in range(BATCH_DOCS):
                try:
                    texts.append(next(stream))
                except StopIteration:
                    break
            if not texts:
                break
            for toks in tokenize_batch(texts):
                writer.add(toks)
                local += len(toks)
                count += len(toks)
                pbar.update(len(toks))
                if local >= budget:
                    break
    pbar.close()
    writer.close()
    min_ok = math.ceil(0.9 * total_budget)
    if count < min_ok:
        raise RuntimeError(f"{prefix} underfilled: got {count:,} tokens, need at least {min_ok:,} / {total_budget:,}")
    with open(done_marker(out_dir, prefix), "w", encoding="utf-8") as f:
        f.write(str(count))
    print(f"{prefix}: {count:,} / {total_budget:,} tokens", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mix10B")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--source", choices=["edu", "dclm", "code", "math", "anneal"], required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.source == "anneal":
        build_anneal(args.out, "anneal", args.scale)
        return
    if args.source == "code":
        build_code(args.out, args.scale)
        return
    dataset, config, field, budget_raw = MAIN_SOURCES[args.source]
    build_single(args.out, args.source, dataset, config, field, int(budget_raw * args.scale))


if __name__ == "__main__":
    main()
