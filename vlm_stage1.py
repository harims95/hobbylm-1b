"""Stage-1 alignment (DDP): train ONLY the projector on LAION-CC-SBU-558K. LLM + SigLIP2 frozen.

  torchrun --standalone --nproc_per_node=8 vlm_stage1.py --backbone ... --json ... --images ... --out ...
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from generate import load_model
from multimodal import MoEVLM
from vision import SiglipVision
from vlm_data import LlavaPretrain, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        dev = torch.device("cuda", local)
        torch.cuda.set_device(dev)
    else:
        rank, world, local = 0, 1, 0
        dev = torch.device("cuda")
    master = rank == 0

    def log(*a):
        if master:
            print(*a, flush=True)

    torch.manual_seed(1234 + rank)
    torch.set_float32_matmul_precision("high")

    enc = SiglipVision(device=dev)                       # frozen SigLIP2 (one copy per rank)
    llm, cfg, vloss, _ = load_model(args.backbone, dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    vlm.set_llm_trainable(False)                         # stage 1: projector only
    llm.set_bias_update_rate(0.0)                        # freeze MoE routing bias (keep backbone fixed)
    n_proj = sum(p.numel() for p in vlm.projector_parameters())
    log(f"backbone d{cfg.d_model} val={vloss:.3f} | SigLIP2 hidden={enc.hidden} | projector={n_proj/1e6:.2f}M")

    raw = vlm
    if ddp:
        vlm = DDP(vlm, device_ids=[local])               # only the projector has grad -> DDP syncs just that

    opt = torch.optim.AdamW(raw.projector_parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    ds = LlavaPretrain(args.json, args.images)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=args.micro, sampler=sampler, shuffle=(sampler is None),
                    num_workers=6, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    log(f"dataset={len(ds)} | micro={args.micro} x world={world} = eff batch {args.micro*world} | max_steps={args.max_steps}")

    vlm.train()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step, t0, run, last, epoch, done = 0, time.time(), 0.0, float("nan"), 0, False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for imgs, ids, tgt in dl:
            ids, tgt = ids.to(dev), tgt.to(dev)
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
            with torch.no_grad(), amp:
                feats = enc.encode_pixels(enc.preprocess(imgs))
            with amp:
                loss, parts = vlm(ids, image_features=feats, targets=tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.projector_parameters(), 1.0)
            opt.step()
            last = loss.item()
            run += last
            if master and step % args.log_every == 0:
                dt = (time.time() - t0) / (step + 1)
                log(f"step {step:5d} | loss {last:.4f} | avg {run/(step+1):.4f} | "
                    f"lr {opt.param_groups[0]['lr']:.2e} | {dt*1000:.0f}ms/step")
            step += 1
            if step >= args.max_steps:
                done = True
                break
        epoch += 1

    if master:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"projector": raw.mm_projector.state_dict(), "vision_dim": enc.hidden,
                    "backbone": args.backbone, "steps": step}, f"{args.out}/projector.pt")
        log(f"saved projector -> {args.out}/projector.pt  (final loss {last:.4f})")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
