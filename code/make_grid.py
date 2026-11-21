# -*- coding: utf-8 -*-
"""Build or check the list of runs for one half of the grid.

plan   writes the expected rows in the same schema as the runner
check  compares that list with the archives on disk and the metrics file

    python make_grid.py plan  --dist-idx 0,1,2 --out results_A.csv
    python make_grid.py check --results results_A.csv --metrics results/metrics.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# columns produced by the runner, in the same order
FIELDS = ["tag", "dist_idx", "cap_idx", "radius", "depth", "width", "params", "seed",
          "n_train", "T", "K", "final_loss", "seconds", "npz_path",
          "mode_coverage", "mmd", "nn_ratio"]


def n_params(depth, width, tdim=32):
    """Denoiser layout: 2+tdim inputs, depth hidden layers, 2 outputs."""
    total = (2 + tdim) * width + width
    total += (depth - 1) * (width * width + width)
    total += width * 2 + 2
    return total


def parse_caps(s):
    out = []
    for part in s.split(","):
        d, w = part.split(":")
        out.append((int(d), int(w)))
    return out


def plan(a):
    dists = [float(x) for x in a.dists.split(",")]
    caps = parse_caps(a.caps)
    seeds = [int(x) for x in a.seeds.split(",")]
    idx = [int(x) for x in a.dist_idx.split(",")]
    rows = []
    for di in idx:
        for ci, (depth, width) in enumerate(caps):
            for s in seeds:
                rows.append(dict(
                    tag=a.tag, dist_idx=di, cap_idx=ci, radius=dists[di],
                    depth=depth, width=width, params=n_params(depth, width),
                    seed=s, n_train=a.n_train, T=a.T, K=8,
                    npz_path=os.path.join(a.npz_root, a.tag,
                                          f"{a.tag}_d{di}_c{ci}_s{s}.npz"),
                    **{c: np.nan for c in ("final_loss", "seconds",
                                           "mode_coverage", "mmd", "nn_ratio")}))
    df = pd.DataFrame(rows, columns=FIELDS)
    df.to_csv(a.out, index=False)
    print(f"已写出 {a.out}: {len(df)} 行 "
          f"({len(idx)} 距离 × {len(caps)} 容量 × {len(seeds)} 种子)")
    print(f"参数量: {[n_params(d, w) for d, w in caps]}")
    return df


def check(a):
    res = pd.read_csv(a.results)
    print(f"{a.results}: {len(res)} 行")
    print("每个 (dist_idx, cap_idx) 的行数:")
    print(res.groupby(["dist_idx", "cap_idx"]).size().unstack("cap_idx").to_string())

    missing = [p for p in res["npz_path"]
               if not os.path.exists(os.path.join(a.npz_root,
                                                  os.path.basename(p)))]
    print(f"npz 缺失: {len(missing)}")
    if missing[:3]:
        print("  例:", missing[:3])

    if os.path.exists(a.metrics):
        m = pd.read_csv(a.metrics)
        m = m[m.tag == res.tag.iloc[0]]
        key = ["dist_idx", "cap_idx", "seed"]
        cols = ["mode_coverage", "mmd", "nn_ratio"]
        if len(m):
            merged = res.merge(m[key + cols], on=key, how="left")
        else:
            merged = res.assign(**{c: np.nan for c in cols})
        print(f"指标未算: {int(merged[cols].isna().any(axis=1).sum())} / {len(merged)}")
    else:
        print(f"{a.metrics} 还不存在")
    dup = res.duplicated(["dist_idx", "cap_idx", "seed"]).sum()
    print(f"重复行: {dup}")
    loss = res["final_loss"]
    print(f"final_loss 范围: {loss.min():.4f} ~ {loss.max():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dists", default="1.0,1.5,2.0,3.0,4.5")
    p.add_argument("--caps", default="1:8,2:16,3:64,4:256,5:512")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--dist-idx", default="0,1,2")
    p.add_argument("--tag", default="main")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--npz-root", default="diffusion_project")
    p.add_argument("--out", default="../results_A.csv")
    c = sub.add_parser("check")
    c.add_argument("--results", default="../results_A.csv")
    c.add_argument("--metrics", default="../metrics.csv")
    c.add_argument("--npz-root", default="diffusion_project")
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else check(a)
