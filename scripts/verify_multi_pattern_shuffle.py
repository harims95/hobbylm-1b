from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hobbylm.data import shard_order


def family_name(path: str) -> str:
    return Path(path).stem.split("_", 1)[0]


def make_demo_shards(root: Path) -> str:
    counts = {"edu": 31, "dclm": 8, "code": 6, "math": 6}
    for family, count in counts.items():
        for idx in range(1, count + 1):
            (root / f"{family}_{idx:06d}.bin").write_bytes(b"")
    return ",".join(
        str(root / pattern)
        for pattern in ("edu_*.bin", "dclm_*.bin", "code_*.bin", "math_*.bin")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--stratified", action="store_true")
    args = ap.parse_args()

    if args.patterns:
        patterns = args.patterns
    else:
        tmp = tempfile.TemporaryDirectory()
        patterns = make_demo_shards(Path(tmp.name))

    order = shard_order(patterns, shuffle_shards=True, seed=1337, stratified=args.stratified)
    families = [family_name(path) for path in order[:args.count]]
    print(" ".join(families))


if __name__ == "__main__":
    main()
