"""Muon (hidden + expert matrices) + AdamW (router, embeddings, head, norms).

Muon orthogonalizes the momentum via Newton-Schulz. Works on 2D weights and, batched over the
leading dim, on 3D expert stacks (E, d_in, d_out). The router gate, embeddings, lm_head, and all
1D params (norms/biases) go to AdamW — Muon would destroy the router's routing signal.
"""
from __future__ import annotations

import torch
from torch import Tensor


# Polar Express coefficient schedule (num_iters=5, safety_factor=2e-2, cushion=2), from
# modded-nanogpt / https://arxiv.org/abs/2505.16932. A tuned per-iteration (a,b,c) schedule
# that converges the orthogonalization faster than fixed Newton-Schulz coefficients.
_POLAR_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7, schedule: str = "ns5") -> Tensor:
    """Orthogonalize over the last two dims; batched over any leading dims (e.g. experts).

    schedule="ns5": fixed Newton-Schulz coefficients (baseline, `steps` iterations).
    schedule="polar": Polar Express per-iteration coefficient schedule (5 iterations; `steps` ignored).
    """
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    if schedule == "polar":
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
        for a, b, c in _POLAR_COEFFS:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    else:
        a, b, c = 3.4445, -4.7750, 2.0315
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    if transpose:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.1, ns_steps=5, schedule="ns5"):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      ns_steps=ns_steps, schedule=schedule))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            mom, wd, lr, steps = group["momentum"], group["weight_decay"], group["lr"], group["ns_steps"]
            schedule = group["schedule"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "mom" not in state:
                    state["mom"] = torch.zeros_like(g)
                buf = state["mom"]
                buf.lerp_(g, 1 - mom)
                g = g.lerp(buf, mom)                       # Nesterov
                o = newton_schulz(g, steps, schedule=schedule).to(p.dtype)
                m, n = p.shape[-2], p.shape[-1]
                scale = 0.2 * max(m, n) ** 0.5             # update RMS ~ AdamW (Moonlight)
                p.mul_(1 - lr * wd)                        # decoupled weight decay
                p.add_(o, alpha=-lr * scale)


def build_optimizers(model, tc):
    """Return (muon, adamw). Dedupes tied params; routes by name/shape."""
    muon_p, adam_p, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_router = name.endswith("gate.weight")
        is_embed = ("embed" in name) or ("lm_head" in name)
        if p.ndim >= 2 and not is_router and not is_embed:
            muon_p.append(p)
        else:
            adam_p.append(p)
    muon = Muon(muon_p, lr=tc.muon_lr, momentum=tc.muon_momentum,
                weight_decay=tc.muon_wd, ns_steps=tc.muon_ns_steps,
                schedule=getattr(tc, "orthogonalizer", "ns5"))
    adamw = torch.optim.AdamW(adam_p, lr=tc.adam_lr, betas=tc.adam_betas,
                              weight_decay=tc.adam_wd, eps=1e-8)
    return muon, adamw, (len(muon_p), len(adam_p))
