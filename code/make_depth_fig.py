# -*- coding: utf-8 -*-
"""画「深度 × 宽度」的 nn_ratio 折线：两个半径各一个面板，两条线（宽度 256 / 512）。

宽度 512 的 depth=5 用 mainN40 的 (5,512) 补（同配置 n=40、30000 步，5 个种子）。
用法： python make_depth_fig.py --csv ../results/metrics.csv --out ../figures/depth_width.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = {1.5: 0.048, 3.0: 0.049}


def collect(csv):
    m = pd.read_csv(csv)
    dp = m[m.group == "depthsweep"][["radius", "depth", "width", "seed", "nn_ratio"]].copy()
    fill = m[(m.tag == "mainN40") & (m.cap_idx == 4) & (m.seed <= 2)][
        ["radius", "seed", "nn_ratio"]].copy()
    fill["depth"] = 5
    fill["width"] = 512
    dp = pd.concat([dp, fill], ignore_index=True)
    dp = dp.drop_duplicates(["radius", "depth", "width", "seed"], keep="last")
    return dp.groupby(["radius", "width", "depth"]).nn_ratio.agg(["mean", "std", "size"]).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results/metrics.csv")
    ap.add_argument("--out", default="../figures/depth_width.png")
    a = ap.parse_args()
    g = collect(a.csv)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, r in zip(axes, (1.5, 3.0)):
        sub = g[g.radius == r]
        for w, style in ((256, dict(marker="o", color="#1f77b4")),
                         (512, dict(marker="s", color="#d62728"))):
            s = sub[sub.width == w].sort_values("depth")
            if s.empty:
                continue
            ax.errorbar(s.depth, s["mean"], yerr=s["std"], capsize=3, lw=1.6,
                        label=f"width {w}", **style)
            for _, row in s.iterrows():
                ax.annotate(f"{row['mean']:.3f}", (row.depth, row["mean"]),
                            textcoords="offset points", xytext=(0, 7),
                            fontsize=6.5, ha="center")
        ax.axhline(BASE[r], ls="--", lw=1, color="0.4")
        ax.text(6.3, BASE[r] - 0.009, "no-memorisation baseline", fontsize=6.5, color="0.35", ha="right")
        ax.set_title(f"r = {r}", fontsize=9)
        ax.set_xlabel("depth (hidden layers)", fontsize=8)
        ax.set_xticks(range(1, 7))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("nn_ratio (mean ± sd over seeds)", fontsize=8)
    fig.suptitle("Memorisation vs depth at fixed width (n_train=40, 30000 steps)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print("saved", a.out)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
