from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


EDU_REPO = "karpathy/fineweb-edu-100B-gpt2-token-shards"
MIX_REPO = "harims95/hobbylm-mix100b-gpt2"
VAL_REPO = "kjj0/fineweb10B-gpt2"
VAL_FILES = ["fineweb_val_000000.bin"]
SOURCE_PREFIXES = ("dclm_", "code_", "math_", "anneal_")
SHARD_BYTES = 100_000_000 * 2 + 256 * 4


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def select_files(api: HfApi) -> tuple[list[str], dict[str, list[str]]]:
    edu = sorted(
        f
        for f in api.list_repo_files(EDU_REPO, repo_type="dataset")
        if Path(f).name.startswith("edu_fineweb_train_") and f.endswith(".bin")
    )
    edu = [f for f in edu if "edu_fineweb_train_000001.bin" <= Path(f).name <= "edu_fineweb_train_000600.bin"]

    mix_files = api.list_repo_files(MIX_REPO, repo_type="dataset")
    by_source = {
        prefix[:-1]: sorted(
            f for f in mix_files if f.startswith(prefix) and f.endswith(".bin")
        )
        for prefix in SOURCE_PREFIXES
    }
    return edu, by_source


def copy_hf_file(repo_id: str, filename: str, out_dir: Path, token: str | None) -> None:
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    dst = out_dir / Path(filename).name
    if dst.exists() and dst.stat().st_size == Path(cached).stat().st_size:
        return
    shutil.copy2(cached, dst)


def print_group(name: str, files: list[str], list_files: bool) -> None:
    print(f"{name}: {len(files)} shard(s)")
    if files:
        print(f"  first: {Path(files[0]).name}")
        print(f"  last:  {Path(files[-1]).name}")
    if list_files:
        for fp in files:
            print(f"  {Path(fp).name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage the 1B Vast shard set from Hugging Face.")
    ap.add_argument("--out", default="/workspace/data/mix100B",
                    help="Target directory on the Vast box; all shard families land here.")
    ap.add_argument("--val-out", default="/workspace/data/fineweb_val",
                    help="Target directory for the original fineweb_val shard(s).")
    ap.add_argument("--download", action="store_true",
                    help="Actually download/copy files. Omit for dry-run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List/verify selected shards without downloading. This is the default.")
    ap.add_argument("--list-files", action="store_true",
                    help="Print every selected shard filename.")
    ap.add_argument("--env-file", default=".env",
                    help="Optional .env file with HF_TOKEN/HUGGING_FACE_HUB_TOKEN.")
    args = ap.parse_args()

    load_dotenv(Path(args.env_file))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    edu, by_source = select_files(api)
    main_mix = by_source["dclm"] + by_source["code"] + by_source["math"]
    all_mix = main_mix + by_source["anneal"]
    all_files = edu + all_mix

    print_group("edu", edu, args.list_files)
    for name in ("dclm", "code", "math", "anneal"):
        print_group(name, by_source[name], args.list_files)
    print_group("val", VAL_FILES, args.list_files)

    print(f"main_phase_shards: {len(edu) + len(main_mix)} (edu+dclm+code+math)")
    print(f"total_shards: {len(all_files)} (edu+dclm+code+math+anneal)")
    approx_gb = (len(all_files) + len(VAL_FILES)) * SHARD_BYTES / 1e9
    print(f"approx_dataset_size_gb: {approx_gb:.1f}")
    print(f"fits_4tb_host: {approx_gb < 4000}")
    print(f"out_dir: {args.out}")
    print(f"val_out_dir: {args.val_out}")

    expected = {
        "edu": 600,
        "dclm": 150,
        "code": 100,
        "math": 100,
        "anneal": 50,
    }
    actual = {"edu": len(edu), **{name: len(files) for name, files in by_source.items()}}
    bad = {name: (actual[name], want) for name, want in expected.items() if actual[name] != want}
    if bad:
        raise SystemExit(f"bad shard counts: {bad}")

    if not args.download:
        print("dry_run: no files downloaded")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in edu:
        copy_hf_file(EDU_REPO, filename, out_dir, token)
    for filename in all_mix:
        copy_hf_file(MIX_REPO, filename, out_dir, token)
    val_out_dir = Path(args.val_out)
    val_out_dir.mkdir(parents=True, exist_ok=True)
    for filename in VAL_FILES:
        copy_hf_file(VAL_REPO, filename, val_out_dir, token)
    print("download_complete")


if __name__ == "__main__":
    main()
