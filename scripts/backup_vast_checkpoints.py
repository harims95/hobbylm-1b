from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


ALLOW_PATTERNS = ["ckpt_*.pt", "model.pt", "config.json", "result.json"]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def snapshot(run_dir: Path) -> tuple[int, float]:
    files = [p for p in run_dir.iterdir() if p.is_file() and any(p.match(pat) for pat in ALLOW_PATTERNS)]
    if not files:
        return 0, 0.0
    return len(files), max(p.stat().st_mtime for p in files)


def backup_once(args: argparse.Namespace, token: str | None) -> None:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    n_files, newest = snapshot(run_dir)
    print(f"checkpoint_files={n_files} newest_mtime={newest:.0f} run_dir={run_dir}", flush=True)
    if n_files == 0:
        return
    if args.dry_run:
        print("dry_run: no upload", flush=True)
        return

    HfApi(token=token).create_repo(args.repo_id, repo_type=args.repo_type,
                                  private=args.private, exist_ok=True)
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(run_dir),
        path_in_repo=args.path_in_repo,
        allow_patterns=ALLOW_PATTERNS,
        commit_message=args.commit_message,
        token=token,
    )
    print(f"uploaded {n_files} file(s) to {args.repo_id}/{args.path_in_repo}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch checkpoint backups from a Vast training box to HF.")
    ap.add_argument("--run-dir", default="/workspace/runs/1b_main_vast")
    ap.add_argument("--repo-id", default="harims95/hobbylm-1b-checkpoints")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--path-in-repo", default="1b_main_vast")
    ap.add_argument("--commit-message", default="backup: batch upload Vast checkpoints")
    ap.add_argument("--interval-minutes", type=float, default=30.0)
    ap.add_argument("--watch", action="store_true",
                    help="Keep running and upload at most once per interval when files changed.")
    ap.add_argument("--upload", action="store_true",
                    help="Actually upload. Omit for dry-run.")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()
    args.dry_run = not args.upload
    args.private = not args.public

    load_dotenv(Path(args.env_file))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not args.watch:
        backup_once(args, token)
        return

    last_uploaded = 0.0
    while True:
        _, newest = snapshot(Path(args.run_dir))
        if newest > last_uploaded:
            backup_once(args, token)
            last_uploaded = newest
        else:
            print("no checkpoint changes; skipping upload", flush=True)
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
