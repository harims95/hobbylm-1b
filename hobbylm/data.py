"""FineWeb .bin data loader (modded-nanogpt format).

Each shard: 256-int32 header (magic 20240520, version 1, num_tokens), then uint16 GPT-2 tokens.
Yields (inputs, targets) of shape (B, S), next-token aligned per row. Shards by DDP rank.
"""
from __future__ import annotations

import glob
import random
import torch


def load_shard(path: str) -> torch.Tensor:
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0].item() == 20240520, f"bad magic in {path}"
    assert header[1].item() == 1, f"bad version in {path}"
    ntok = int(header[2].item())
    tokens = torch.empty(ntok, dtype=torch.uint16)
    with open(path, "rb", buffering=0) as f:
        f.seek(256 * 4)
        nread = f.readinto(tokens.numpy())
    assert nread == ntok * 2, f"short read in {path}"
    return tokens


def expand_patterns(patterns: str) -> list[str]:
    files = []
    seen = set()
    for pattern in (p.strip() for p in patterns.split(",") if p.strip()):
        for fp in sorted(glob.glob(pattern)):
            if fp not in seen:
                seen.add(fp)
                files.append(fp)
    return files


def shard_family(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if name.startswith("edu_fineweb_train_"):
        return "edu"
    return name.split("_", 1)[0]


def stratified_shard_order(files: list[str], seed: int = 1337, cycle: int = 0) -> list[str]:
    by_family: dict[str, list[str]] = {}
    for fp in files:
        by_family.setdefault(shard_family(fp), []).append(fp)

    rng = random.Random(seed + cycle)
    for family_files in by_family.values():
        rng.shuffle(family_files)

    total = len(files)
    targets = {family: len(family_files) for family, family_files in by_family.items()}
    used = {family: 0 for family in by_family}
    order = []
    for pos in range(total):
        # Pick the family most behind its ideal cumulative count. This is a small
        # "carryover" scheduler: exact totals, but near-target ratios in every window.
        family = max(
            (f for f in by_family if used[f] < targets[f]),
            key=lambda f: (targets[f] * (pos + 1) / total) - used[f],
        )
        order.append(by_family[family][used[family]])
        used[family] += 1
    return order


def shard_order(patterns: str, shuffle_shards: bool = False, seed: int = 1337,
                cycle: int = 0, stratified: bool = False) -> list[str]:
    files = expand_patterns(patterns)
    assert files, f"no data files match {patterns}"
    if stratified:
        return stratified_shard_order(files, seed=seed, cycle=cycle)
    if not shuffle_shards or len(files) <= 1:
        return files

    rng = random.Random(seed + cycle)
    mixed = list(files)
    rng.shuffle(mixed)
    return mixed


def data_generator(pattern: str, B: int, S: int, device, rank: int = 0, world: int = 1,
                   to_device: bool = True, shuffle_shards: bool = False, seed: int = 1337,
                   stratified_shards: bool = False):
    """Yields (x, y) of shape (B, S). If to_device is False, yields pinned CPU long tensors
    (for an async prefetcher to copy H2D while the GPU computes)."""
    files = expand_patterns(pattern)
    assert files, f"no data files match {pattern}"
    block = B * S
    pin = (not to_device) and torch.cuda.is_available()
    cycle = 0
    while True:
        order = shard_order(pattern, shuffle_shards=shuffle_shards, seed=seed,
                            cycle=cycle, stratified=stratified_shards)
        for fp in order:
            toks = load_shard(fp)
            n_blocks = (len(toks) - 1) // block
            # interleave blocks across ranks so each rank sees distinct data
            for i in range(rank, n_blocks, world):
                chunk = toks[i * block: i * block + block + 1]
                if to_device:
                    buf = chunk.to(device, dtype=torch.long, non_blocking=True)
                else:
                    buf = chunk.to(torch.long)
                    if pin:
                        buf = buf.pin_memory()
                x = buf[:-1].view(B, S)
                y = buf[1:].view(B, S)
                yield x, y
        cycle += 1


class CUDAPrefetcher:
    """Double-buffers the host->device copy on a side stream so it overlaps with compute.
    Wrap a data_generator(..., to_device=False); .next() returns GPU tensors."""
    def __init__(self, gen, device):
        self.gen = gen
        self.device = device
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._preload()

    def _preload(self):
        x, y = next(self.gen)
        if self.stream is None:
            self._next = (x.to(self.device), y.to(self.device))
            return
        with torch.cuda.stream(self.stream):
            self._next = (x.to(self.device, non_blocking=True),
                          y.to(self.device, non_blocking=True))

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self._next
        # ensure the consumer doesn't race the side-stream copy being recycled
        if self.stream is not None:
            x.record_stream(torch.cuda.current_stream())
            y.record_stream(torch.cuda.current_stream())
        self._preload()
        return x, y


def count_tokens(pattern: str) -> int:
    total = 0
    for fp in expand_patterns(pattern):
        header = torch.from_file(str(fp), False, 256, dtype=torch.int32)
        total += int(header[2].item())
    return total
