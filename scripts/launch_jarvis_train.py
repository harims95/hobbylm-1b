from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


MAIN_PATTERN = "edu_fineweb_train_*.bin,dclm_*.bin,code_*.bin,math_*.bin"
ANNEAL_PATTERN = "anneal_*.bin"


def latest_checkpoint(run_dir: str) -> str:
    root = Path(run_dir)
    candidates = []
    model = root / "model.pt"
    if model.exists():
        candidates.append((model.stat().st_mtime, model))
    for ckpt in root.glob("ckpt_*.pt"):
        candidates.append((ckpt.stat().st_mtime, ckpt))
    if not candidates:
        raise SystemExit(f"no checkpoint found in {root}")
    return str(max(candidates, key=lambda item: item[0])[1])


def build_command(args: argparse.Namespace) -> list[str]:
    train_pattern = args.train_pattern
    if train_pattern == "main":
        train_pattern = MAIN_PATTERN
    elif train_pattern == "anneal":
        train_pattern = ANNEAL_PATTERN

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "training/train.py",
        "--preset", args.preset,
        "--run_name", args.run_name,
        "--data_dir", args.data_dir,
        "--out_dir", args.out_dir,
        "--max_steps", str(args.max_steps),
        "--schedule_max_steps", str(args.schedule_max_steps),
        "--micro_batch_seqs", str(args.micro_batch_seqs),
        "--seq_len", str(args.seq_len),
        "--batch_tokens", str(args.batch_tokens),
        "--train_pattern", train_pattern,
        "--val_pattern", args.val_pattern,
        "--orthogonalizer", args.orthogonalizer,
        "--save_every", str(args.save_every),
        "--val_every", str(args.val_every),
        "--stratified_shards",
        "--set", "fused_ce=true",
    ]
    if args.no_compile:
        cmd.append("--no_compile")
    resume = args.resume
    if resume == "latest":
        resume = latest_checkpoint(f"{args.out_dir.rstrip('/')}/{args.run_name}")
    if resume:
        cmd += ["--resume", resume]
    if args.init_from:
        cmd += ["--init_from", args.init_from]
    for item in args.set:
        cmd.append(item)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch HobbyLM training on a Jarvis Labs box with torchrun.")
    ap.add_argument("--nproc-per-node", type=int, default=4)
    ap.add_argument("--preset", default="1B")
    ap.add_argument("--run-name", default="1b_main_jarvis")
    # Jarvis Labs only persists /home across pause/resume; anything outside it is lost.
    ap.add_argument("--data-dir", default="/home/data/mix100B")
    ap.add_argument("--out-dir", default="/home/runs")
    ap.add_argument("--train-pattern", default="main",
                    help="'main', 'anneal', or an explicit comma-separated shard pattern.")
    ap.add_argument("--val-pattern", default="/home/data/fineweb_val/fineweb_val_*.bin")
    ap.add_argument("--max-steps", type=int, default=81_062,
                    help="Main-phase stop step for 85B tokens at 1,048,576 tokens/step.")
    ap.add_argument("--schedule-max-steps", type=int, default=95_367,
                    help="Full 100B-token schedule horizon at 1,048,576 tokens/step.")
    ap.add_argument("--micro-batch-seqs", type=int, default=32,
                    help="Validated seq_len=1024 throughput setting; reduce only if the host OOMs.")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-tokens", type=int, default=1_048_576)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--orthogonalizer", default="ns5", choices=["ns5", "polar"])
    ap.add_argument("--no-compile", action="store_true",
                    help="run fully eager by skipping whole-model torch.compile.")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--set", nargs="*", default=[], help="Additional ModelConfig overrides key=value.")
    ap.add_argument("--run", action="store_true", help="Actually execute. Omit to print only.")
    args = ap.parse_args()

    cmd = build_command(args)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if args.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
