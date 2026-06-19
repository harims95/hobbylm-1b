"""FineWeb .bin data loader (modded-nanogpt format).

Each shard: 256-int32 header (magic 20240520, version 1, num_tokens), then uint16 GPT-2 tokens.
Yields (inputs, targets) of shape (B, S), next-token aligned per row. Shards by DDP rank.
"""
from __future__ import annotations

import glob
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


def data_generator(pattern: str, B: int, S: int, device, rank: int = 0, world: int = 1,
                   to_device: bool = True):
    """Yields (x, y) of shape (B, S). If to_device is False, yields pinned CPU long tensors
    (for an async prefetcher to copy H2D while the GPU computes)."""
    files = sorted(glob.glob(pattern))
    assert files, f"no data files match {pattern}"
    block = B * S
    pin = (not to_device) and torch.cuda.is_available()
    while True:
        for fp in files:
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
    for fp in sorted(glob.glob(pattern)):
        header = torch.from_file(str(fp), False, 256, dtype=torch.int32)
        total += int(header[2].item())
    return total
