from pathlib import Path
import random

import torch

import hobbylm.data as data_mod
from hobbylm.data import expand_patterns, shard_order
from hobbylm.data import data_generator
from training.train import resolve_pattern


def test_resolve_pattern_joins_relative_glob():
    assert Path(resolve_pattern("data/mix10B", "mix_train_*.bin")) == Path("data/mix10B/mix_train_*.bin")


def test_resolve_pattern_keeps_absolute_glob():
    pattern = str(Path("data/fineweb10B/fineweb_val_*.bin").resolve())
    assert Path(resolve_pattern("data/mix10B", pattern)) == Path(pattern)


def test_expand_patterns_supports_comma_list(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"")
    b.write_bytes(b"")
    got = expand_patterns(f"{tmp_path}\\a*.bin,{tmp_path}\\b*.bin")
    assert got == [str(a), str(b)]


def test_data_generator_can_shuffle_shards_deterministically(tmp_path, monkeypatch):
    files = []
    token_map = {}
    for idx, name in enumerate(("a.bin", "b.bin", "c.bin"), start=1):
        path = tmp_path / name
        path.write_bytes(b"")
        files.append(str(path))
        token_map[str(path)] = torch.tensor([idx, idx + 10], dtype=torch.uint16)

    def fake_load_shard(path: str) -> torch.Tensor:
        return token_map[path]

    monkeypatch.setattr(data_mod, "load_shard", fake_load_shard)
    pattern = f"{tmp_path}\\*.bin"

    plain = data_generator(pattern, 1, 1, torch.device("cpu"), shuffle_shards=False)
    plain_order = [int(next(plain)[0][0, 0].item()) for _ in range(3)]
    assert plain_order == [1, 2, 3]

    expected_paths = expand_patterns(pattern)
    random.Random(1337).shuffle(expected_paths)
    expected = [int(token_map[path][0].item()) for path in expected_paths]

    shuffled = data_generator(pattern, 1, 1, torch.device("cpu"), shuffle_shards=True, seed=1337)
    shuffled_order = [int(next(shuffled)[0][0, 0].item()) for _ in range(3)]
    assert shuffled_order == expected


def test_shard_order_shuffles_across_all_remaining_shards(tmp_path):
    files = []
    for prefix in ("edu", "dclm", "code"):
        for idx in range(3):
            path = tmp_path / f"{prefix}_{idx:06d}.bin"
            path.write_bytes(b"")
            files.append(str(path))

    pattern = f"{tmp_path}\\edu_*.bin,{tmp_path}\\dclm_*.bin,{tmp_path}\\code_*.bin"
    expected = expand_patterns(pattern)
    random.Random(1337).shuffle(expected)

    order = shard_order(pattern, shuffle_shards=True, seed=1337)
    assert order == expected
