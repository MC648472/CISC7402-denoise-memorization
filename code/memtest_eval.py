# -*- coding: utf-8 -*-
"""memtest 评估：每个 n_train 单独校准 τ，算邻近度，并给出 B 用的中位距离比做对照。

用法：
  python memtest_eval.py --root .\_memtest --out ..\metrics_memtest.csv

对每个文件输出：
  n, radius, tau, baseline(真实新样本的 P(nn<tau)，应≈0.05),
  nn_ratio(主指标), nn_med_ratio(B 的中位比), coverage8, coverage_present, mmd
coverage_present 的分母是"训练集里实际出现过的模式数"——n=16 时会有模式没有任何训练点，
按 8 个算的话即使完美模型也拿不到 1.0。
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import eval_toolkit as ev

N_FRESH = 2000


def parse(name):
    m = re.search(r"_d(\d+)_c(\d+)_s(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def coverage_present(final, centers, train, sigma, radius_k=ev.MODE_RADIUS_K,
                     min_frac=ev.MODE_MIN_FRAC):
    """只统计训练集里出现过的模式；分母是出现过的模式数。"""
    present = np.unique(ev.assign_modes(train, centers))
    D = ev._sqdist(final, centers)
    lab = D.argmin(1)
    inside = np.sqrt(np.maximum(D.min(1), 0)) <= radius_k * sigma
    counts = np.bincount(lab[inside], minlength=len(centers))
    ok = counts[present] >= min_frac * len(final) / len(centers)
    return float(np.mean(ok)), int(len(present))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="metrics_memtest.csv")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "**", "*.npz"), recursive=True))
    files = [f for f in files if "memtest" in os.path.basename(f)]
    if not files:
        print("没找到 memtest_*.npz"); return

    rows = []
    for f in files:
        z = np.load(f)
        final, train = z["final"], z["train"]
        radius = float(z["radius"]) if "radius" in z.files else float(z["means_radius"])
        sigma = float(z["sigma"]) if "sigma" in z.files else 0.3
        n = len(train)
        p = parse(os.path.basename(f))
        seed = p[2] if p else 0
        cap = p[1] if p else -1
        dist = ev.chord(radius)
        centers = ev.gmm_centers(dist)

        calib = ev.gmm_sample(ev.N_CALIB, ev.seed_calib(seed), radius=radius, sigma=sigma)
        tau = ev.nn_threshold(calib, train)
        fresh = ev.gmm_sample(N_FRESH, ev.seed_target(seed), radius=radius, sigma=sigma)

        d_f, _ = ev.nn_two(final, train)
        d_n, _ = ev.nn_two(fresh, train)
        cov8 = ev.mode_coverage(final, centers, sigma)
        covp, n_present = coverage_present(final, centers, train, sigma)
        target = ev.gmm_sample(ev.N_TARGET, 777, radius=radius, sigma=sigma)
        rows.append(dict(n=n, cap=cap, seed=seed, radius=radius, tau=tau,
                         baseline=float(np.mean(d_n < tau)),
                         nn_ratio=float(np.mean(d_f < tau)),
                         nn_med_ratio=float(np.median(d_f) / max(np.median(d_n), 1e-12)),
                         coverage8=cov8, coverage_present=covp, n_present=n_present,
                         mmd=ev.mmd2(final, target), file=os.path.basename(f)))
        print(f"n={n:<5} cap={cap} cov8={cov8:.3f} cov_p={covp:.3f} "
              f"nn={rows[-1]['nn_ratio']:.3f} med_ratio={rows[-1]['nn_med_ratio']:.2f} "
              f"base={rows[-1]['baseline']:.3f}", flush=True)

    df = pd.DataFrame(rows).sort_values(["n", "cap"])
    df.to_csv(a.out, index=False)
    print("\n" + df.drop(columns=["file"]).round(4).to_string(index=False))
    print("\n按 n 汇总:")
    print(df.groupby("n")[["baseline", "nn_ratio", "nn_med_ratio",
                           "coverage8", "coverage_present"]].mean().round(4).to_string())
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
