# -*- coding: utf-8 -*-
"""Re-run the training loop locally (CPU) with the same model and data code.

Useful when the original archives are unavailable. torch uses different random
streams on CPU and CUDA, so results are statistically rather than bit-wise
comparable with the GPU runs.

    python pilot_repro.py --iters 30000 --n-train 40 --radius 1.5 --cap 2 --out ./runs
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"
CFG = dict(K=8, sigma=0.3, n_train=1000, T=100, iters=3000, batch=256,
           lr=1e-3, n_sample=2000, n_traj=10, tdim=32)
DISTANCES = [1.0, 1.5, 2.0, 3.0, 4.5]
CAPACITIES = [(1, 8), (2, 16), (3, 64), (4, 256), (5, 512)]


def make_gmm(n, K, radius, sigma, seed):
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(K) / K
    means = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    comp = rng.integers(0, K, size=n)
    x = means[comp] + sigma * rng.standard_normal((n, 2))
    return x.astype(np.float32), comp, means


class Denoiser(nn.Module):
    def __init__(self, depth, width, tdim=32):
        super().__init__()
        self.tdim = tdim
        layers, d_in = [], 2 + tdim
        for i in range(depth):
            layers += [nn.Linear(d_in if i == 0 else width, width), nn.SiLU()]
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def temb(self, t):
        half = self.tdim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], dim=1))


class DDPM:
    def __init__(self, T, device):
        self.T = T
        s = 1000.0 / T
        self.betas = torch.linspace(1e-4 * s, 0.02 * s, T, device=device)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return ((model(xt, t) - eps) ** 2).mean()


def build_model(depth, width, cfg):
    model = Denoiser(depth, width, cfg["tdim"]).to(DEVICE)
    model.ddpm = DDPM(cfg["T"], DEVICE)
    return model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(depth, width, radius, seed, cfg):
    set_seed(seed)
    x_np, _, _ = make_gmm(cfg["n_train"], cfg["K"], radius, cfg["sigma"], seed)
    x = torch.tensor(x_np, device=DEVICE)
    model = build_model(depth, width, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    bs = min(cfg["batch"], len(x))
    losses = []
    model.train()
    for it in range(cfg["iters"]):
        idx = torch.randint(0, len(x), (bs,), device=DEVICE)
        loss = model.ddpm.loss(model, x[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return model, losses, x_np


@torch.no_grad()
def sample(model, n, return_trajectory=True, seed=0, n_traj=10):
    model.eval()
    d = model.ddpm
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(n, 2, device=DEVICE, generator=g)
    keep = set(np.linspace(d.T - 1, 0, n_traj - 1).round().astype(int).tolist())
    traj = [x.cpu().numpy()]
    for t in range(d.T - 1, -1, -1):
        tt = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        eps = model(x, tt)
        a, ab, b = d.alphas[t], d.abar[t], d.betas[t]
        mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
        if t > 0:
            x = mean + b.sqrt() * torch.randn(x.shape, device=DEVICE, generator=g)
        else:
            x = mean
        if t in keep:
            traj.append(x.cpu().numpy())
    final = x.cpu().numpy()
    return (final, np.stack(traj)) if return_trajectory else final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, nargs="*", default=None)
    ap.add_argument("--cap", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    a = ap.parse_args()

    cfg = dict(CFG, iters=a.iters)
    if a.n_train:
        cfg["n_train"] = a.n_train
    if a.K:
        cfg["K"] = a.K
    radii = a.radius if a.radius else DISTANCES
    caps = a.cap if a.cap is not None else list(range(len(CAPACITIES)))
    os.makedirs(a.out, exist_ok=True)

    t_all = time.time()
    for r in radii:
        for ci in caps:
            di = DISTANCES.index(r) if r in DISTANCES else -1
            depth, width = CAPACITIES[ci]
            name = f"re_pro_d{di}_c{ci}_s{a.seed}"
            path = os.path.join(a.out, name + ".npz")
            if os.path.exists(path):
                print("skip", name)
                continue
            t0 = time.time()
            model, losses, x_train = train(depth, width, r, a.seed, cfg)
            final, traj = sample(model, cfg["n_sample"], True, seed=a.seed,
                                 n_traj=cfg["n_traj"])
            chord = 2 * r * math.sin(math.pi / cfg["K"])
            np.savez_compressed(path, final=final, traj=traj, train=x_train,
                                losses=np.array(losses), radius=r, dist=chord,
                                dist_idx=di, cap_idx=ci, seed=a.seed,
                                depth=depth, width=width, n_train=cfg["n_train"],
                                sigma=cfg["sigma"], k=cfg["K"])
            print(f"done {name}: r={r} cap={depth}x{width} "
                  f"loss={np.mean(losses[-200:]):.4f} {time.time()-t0:.1f}s", flush=True)
    print(f"total {time.time()-t_all:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
