"""Stage-2 SFT (DDP): instruction-tune projector + LLM on LLaVA-Instruct-150K. SigLIP2 frozen.

  torchrun --standalone --nproc_per_node=8 vlm_stage2.py --backbone ... --stage1 ... --json ... --zip ... --out ...
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path for the `hobbylm` package
from hobbylm.generate import load_model
from hobbylm.multimodal import MoEVLM
from hobbylm.vision import SiglipVision
from hobbylm.vlm_data import LlavaSFT, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--stage1", required=True, help="stage-1 projector.pt to initialize from")
    ap.add_argument("--json", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--proj_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=400)
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

    enc = SiglipVision(device=dev)
    llm, cfg, vloss, _ = load_model(args.backbone, dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    ck = torch.load(args.stage1, map_location=dev, weights_only=False)
    vlm.mm_projector.load_state_dict(ck["projector"])         # init projector from stage 1
    vlm.set_llm_trainable(True)                               # stage 2: projector + LLM both train
    llm.set_bias_update_rate(0.0)                            # freeze MoE balancing bias (avoid DDP buffer desync)
    log(f"backbone d{cfg.d_model} val={vloss:.3f} | stage1 projector steps={ck.get('steps')} | "
        f"trainable={sum(p.numel() for p in vlm.parameters() if p.requires_grad)/1e6:.0f}M")

    raw = vlm
    if ddp:
        vlm = DDP(vlm, device_ids=[local])     # MoE experts use whole grouped weight -> no unused params

    # two groups: projector (higher lr) + LLM (low SFT lr)
    proj_ids = {id(p) for p in raw.mm_projector.parameters()}
    proj_params = [p for p in raw.parameters() if p.requires_grad and id(p) in proj_ids]
    llm_params = [p for p in raw.parameters() if p.requires_grad and id(p) not in proj_ids]
    opt = torch.optim.AdamW([
        {"params": llm_params, "lr": args.lr},
        {"params": proj_params, "lr": args.proj_lr},
    ], betas=(0.9, 0.95), weight_decay=0.0)
    base_lrs = [args.lr, args.proj_lr]

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))            # cosine -> 0

    ds = LlavaSFT(args.json, args.zip)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=args.micro, sampler=sampler, shuffle=(sampler is None),
                    num_workers=6, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    log(f"dataset={len(ds)} | micro={args.micro} x world={world} = {args.micro*world} | max_steps={args.max_steps}")

    def save(tag):
        if not master:
            return
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": raw.llm.state_dict(), "projector": raw.mm_projector.state_dict(),
                    "config": {**cfg.to_dict(), "preset": "500M"}, "vision_dim": enc.hidden,
                    "backbone": args.backbone}, f"{args.out}/{tag}")
        log(f"saved -> {args.out}/{tag}")

    vlm.train()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step, t0, run, last, epoch, done = 0, time.time(), 0.0, float("nan"), 0, False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for imgs, ids, tgt in dl:
            ids, tgt = ids.to(dev), tgt.to(dev)
            m = lr_at(step)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * m
            with torch.no_grad(), amp:
                feats = enc.encode_pixels(enc.preprocess(imgs))
            with amp:
                loss, parts = vlm(ids, image_features=feats, targets=tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad], 1.0)
            opt.step()
            last = loss.item()
            run += last
            if master and step % args.log_every == 0:
                dt = (time.time() - t0) / (step + 1)
                log(f"step {step:5d} | loss {last:.4f} | avg {run/(step+1):.4f} | "
                    f"lr {opt.param_groups[0]['lr']:.2e} | {dt*1000:.0f}ms/step")
            step += 1
            if args.save_every and step % args.save_every == 0:
                save(f"ckpt_{step}.pt")
            if step >= args.max_steps:
                done = True
                break
        epoch += 1

    save("model.pt")
    log(f"stage-2 done (final loss {last:.4f})")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
