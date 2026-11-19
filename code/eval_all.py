# -*- coding: utf-8 -*-
"""把 B 交来的各个文件夹统一算成一张 tidy 表（metrics_all.csv）。

支持的文件名：
  mainN40_d{di}_c{ci}_s{seed}.npz                  n=40, T=100, 冻结五档容量
  memtest_n{N}_d{di}_c{ci}_s{seed}.npz             n=N,  T=100
  ablN{N}_r{radius}_d{depth}_w{width}_s{seed}.npz  n=N,  T=100
  abl_T{T}_d{di}_c{ci}_s{seed}.npz                 n=1000, T=T

用法： python eval_all.py --roots _B_new _A_mainN40 --out metrics_all.csv
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

CAPS = [(1, 8), (2, 16), (3, 64), (4, 256), (5, 512)]
DISTANCES = [1.0, 1.5, 2.0, 3.0, 4.5]


def parse(path):
    b = os.path.basename(path)
    m = re.match(r"main_d(\d)_c(\d)_s(\d+)", b)
    if m:
        di, ci, s = map(int, m.groups())
        return dict(tag="main", group="main", dist_idx=di, cap_idx=ci, seed=s,
                    n_train=1000, T=100, depth=CAPS[ci][0], width=CAPS[ci][1])
    m = re.match(r"depthsweep512_r([\d.]+)_d(\d+)_w(\d+)_s(\d+)", b)
    if m:
        r, d, w, s = m.groups()
        return dict(tag="depthsweep512", group="depthsweep", radius=float(r), seed=int(s),
                    dist_idx=DISTANCES.index(float(r)) if float(r) in DISTANCES else -1,
                    cap_idx=-1, n_train=40, T=100, depth=int(d), width=int(w))
    m = re.match(r"depthsweep_r([\d.]+)_d(\d+)_w(\d+)_s(\d+)", b)
    if m:
        r, d, w, s = m.groups()
        return dict(tag="depthsweep", group="depthsweep", radius=float(r), seed=int(s),
                    dist_idx=DISTANCES.index(float(r)) if float(r) in DISTANCES else -1,
                    cap_idx=-1, n_train=40, T=100, depth=int(d), width=int(w))
    m = re.match(r"widthsweep_r([\d.]+)_d(\d+)_w(\d+)_s(\d+)", b)
    if m:
        r, d, w, s = m.groups()
        return dict(tag="widthsweep", group="widthsweep", radius=float(r), seed=int(s),
                    dist_idx=DISTANCES.index(float(r)) if float(r) in DISTANCES else -1,
                    cap_idx=-1, n_train=40, T=100, depth=int(d), width=int(w))
    m = re.match(r"mainN40_d(\d)_c(\d)_s(\d+)", b)
    if m:
        di, ci, s = map(int, m.groups())
        return dict(tag="mainN40", group="mainN40", dist_idx=di, cap_idx=ci,
                    seed=s, n_train=40, T=100,
                    depth=CAPS[ci][0], width=CAPS[ci][1])
    m = re.match(r"memtest_n(\d+)_d(\d)_c(\d)_s(\d+)", b)
    if m:
        n, di, ci, s = map(int, m.groups())
        return dict(tag=f"memtest_n{n}", group="memtest", dist_idx=di, cap_idx=ci,
                    seed=s, n_train=n, T=100,
                    depth=[5, 4][ci - 4], width=[512, 1024][ci - 4])
    m = re.match(r"ablN(\d+)_r([\d.]+)_d(\d+)_w(\d+)_s(\d+)", b)
    if m:
        n, r, d, w, s = m.groups()
        cap = (int(d), int(w))
        return dict(tag=f"ablN{n}", group="ablN", radius=float(r), seed=int(s),
                    dist_idx=DISTANCES.index(float(r)) if float(r) in DISTANCES else -1,
                    cap_idx=CAPS.index(cap) if cap in CAPS else -1,
                    n_train=int(n), T=100, depth=int(d), width=int(w))
    m = re.match(r"abl_T(\d+)_d(\d)_c(\d)_s(\d+)", b)
    if m:
        T, di, ci, s = map(int, m.groups())
        return dict(tag=f"ablT{T}", group="ablT", dist_idx=di, cap_idx=ci, seed=s,
                    n_train=1000, T=T, depth=CAPS[ci][0], width=CAPS[ci][1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="metrics_all.csv")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()

    files = []
    for root in a.roots:
        files += sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))

    done = set()
    if a.append and os.path.exists(a.out):
        old = pd.read_csv(a.out)
        done = {(r.tag, r.get("dist_idx"), r.get("cap_idx"), r.seed) for r in old.itertuples()}

    rows = []
    for i, f in enumerate(files, 1):
        meta = parse(f)
        if meta is None:
            print("跳过无法识别的文件名:", os.path.basename(f)); continue
        key = (meta["tag"], meta.get("dist_idx"), meta.get("cap_idx"), meta["seed"])
        if key in done:
            continue
        z = np.load(f)
        radius = meta.get("radius", float(z["means_radius"]) if "means_radius" in z.files else np.nan)
        m = ev.evaluate_run(f, radius=radius, seed=meta["seed"])
        rows.append(dict(meta, radius=radius, mode_coverage=m["mode_coverage"],
                         mmd=m["MMD"], nn_ratio=m["nn_ratio"]))
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    if not rows:
        print("没有新文件"); return
    new = pd.DataFrame(rows)
    if a.append and os.path.exists(a.out):
        new = pd.concat([pd.read_csv(a.out), new], ignore_index=True)
    new = new.drop_duplicates(["tag", "dist_idx", "cap_idx", "seed", "radius",
                                "depth", "width", "n_train", "T"], keep="last")
    new = new.sort_values(["group", "tag", "radius", "n_train", "seed", "cap_idx"])
    new.to_csv(a.out, index=False)
    print(f"已写出 {a.out}: {len(new)} 行")
    print(new.groupby(["group", "tag"]).size().to_string())


if __name__ == "__main__":
    main()
