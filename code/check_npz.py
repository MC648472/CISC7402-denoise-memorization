# -*- coding: utf-8 -*-
"""Validate a run archive before it enters the statistics.

Checks array shapes, finiteness, the sampling contract (traj[0] is noise,
traj[-1] equals final) and, when present, the metadata scalars.
Exit code 0 if every file passes, 1 otherwise.

    python check_npz.py RUN.npz              # validate
    python check_npz.py RUN.npz --infer      # also infer k / radius / sigma
    python check_npz.py runs/ --quiet        # whole directory, failures only

Expected arrays: final (n,2), traj (T,n,2), train (m,2), losses (L,).
Expected scalars: dist and seed; width/depth/n_train/sigma recommended.
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED_ARRAYS = ["final", "traj", "train", "losses"]
REQUIRED_SCALARS = ["dist", "seed"]
MIN_SNAPSHOTS = 8


# ---------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------
def validate(path):
    """返回 (errors, warnings)。errors = 这份数据不能用；warnings = 能算但建议补。"""
    errors, warnings = [], []
    z = np.load(path)
    files = set(z.files)

    for k in REQUIRED_ARRAYS:
        if k not in files:
            errors.append(f"缺少数组 {k}")
    for k in REQUIRED_SCALARS:
        if k not in files:
            warnings.append(f"缺少标量 {k}（只能靠 CSV 对号，建议补）")

    final = z["final"] if "final" in files else None
    train = z["train"] if "train" in files else None
    for name, a in (("final", final), ("train", train)):
        if a is None:
            continue
        if a.ndim != 2 or a.shape[1] != 2:
            errors.append(f"{name} 形状 {a.shape}，应为 (n,2)")
        elif not np.all(np.isfinite(a)):
            errors.append(f"{name} 含 NaN/Inf")
    if "losses" in files:
        if z["losses"].ndim != 1:
            warnings.append(f"losses 形状 {z['losses'].shape}，应为 (L,)")
        elif not np.all(np.isfinite(z["losses"])):
            errors.append("losses 含 NaN/Inf")

    if "traj" in files and final is not None:
        traj = z["traj"]
        if traj.ndim != 3 or traj.shape[1:] != final.shape:
            errors.append(f"traj 形状 {traj.shape}，应为 (T,{final.shape[0]},2)")
        elif traj.shape[0] < MIN_SNAPSHOTS:
            errors.append(f"traj 只有 {traj.shape[0]} 个快照，约定 >= {MIN_SNAPSHOTS}")
        else:
            if not np.all(np.isfinite(traj)):
                errors.append("traj 含 NaN/Inf")
            if not np.array_equal(traj[-1], final):
                errors.append("traj[-1] 不等于 final"
                              f"（max|diff|={np.abs(traj[-1] - final).max():.2e}）")
            sd = traj[0].std(0)
            if not np.all(np.abs(sd - 1.0) < 0.15):
                errors.append(f"traj[0] 不是标准正态噪声（每轴 sd={np.round(sd, 3)}）")

    if final is not None and train is not None and len(final) == len(train):
        if np.array_equal(final, train):
            errors.append("final 和 train 完全相同，采样没有生效")
    return errors, warnings


# ---------------------------------------------------------------------
# 反推几何参数（只在缺 metadata 时用）
# ---------------------------------------------------------------------
def kmeans(X, k, iters=150, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j]
                         for j in range(k)])
        if np.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    ss = sum(((X[lab == j] - C[j]) ** 2).sum() for j in range(k))
    return C, lab, float(ss)


def ring_centers(dist, k):
    r = dist / (2 * np.sin(np.pi / k))
    ang = 2 * np.pi * np.arange(k) / k
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def infer_geometry(train, k_max=10, k_min=1, rel_tol=0.15):
    """从训练集反推 k、环半径、相邻中心距离、分量 sigma。
    选 k：组内平方和已降到最优值的 1+rel_tol 以内，取其中最小的 k。"""
    rows = []
    for k in range(k_min, k_max + 1):
        C, lab, ss = kmeans(train, k, seed=0)
        r = float(np.linalg.norm(C, axis=1).mean())
        ang = np.sort(np.arctan2(C[:, 1], C[:, 0]))
        d = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        chord = float(np.mean(2 * r * np.sin(d / 2))) if k > 1 else float("nan")
        rows.append((k, r, chord, ss))
    ss_min = min(r[3] for r in rows)
    k = next(r[0] for r in rows if (r[3] - ss_min) <= rel_tol * max(ss_min, 1e-9))
    _, radius, chord, _ = next(r for r in rows if r[0] == k)
    C, lab, _ = kmeans(train, k, seed=0)
    centers = ring_centers(chord, k) if k > 1 else np.zeros((1, 2))
    lab2 = ((train[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    sigma = float(np.sqrt(((train - centers[lab2]) ** 2).sum(1).mean() / 2))
    return {"k": k, "radius": radius, "dist": chord, "sigma": sigma,
            "sizes": np.bincount(lab2, minlength=k), "k_scan": rows}


# ---------------------------------------------------------------------
def check_one(path, infer=False, quiet=False):
    z = np.load(path)
    files = set(z.files)
    errors, warnings = validate(path)
    info = {"file": path, "errors": errors, "warnings": warnings}

    if not quiet:
        print(f"\n=== {os.path.basename(path)} ===")
        print("  数组:", ", ".join(f"{k}{list(z[k].shape)}" for k in z.files
                                  if z[k].ndim >= 1))
        scal = [f"{k}={z[k].item()}" for k in z.files if z[k].ndim == 0]
        if scal:
            print("  标量:", ", ".join(scal))
        for p in errors:
            print(f"  [FAIL] {p}")
        for p in warnings:
            print(f"  [WARN] {p}")

    have_data = "final" in files and "train" in files
    if not have_data:
        return info

    if not quiet:
        if "losses" in files:
            print(f"  losses 首50={z['losses'][:50].mean():.4f} "
                  f"末50={z['losses'][-50:].mean():.4f}")
        if "traj" in files:
            print(f"  traj[0] 每轴 sd={np.round(z['traj'][0].std(0), 3)} "
                  f"| traj[-1]==final: {np.array_equal(z['traj'][-1], z['final'])}")

    if infer or "dist" not in files:
        g = infer_geometry(z["train"])
        info["inferred"] = g
        if not quiet:
            print(f"  [反推] k={g['k']}  环半径={g['radius']:.3f}  "
                  f"相邻中心距离(dist)={g['dist']:.3f}  sigma={g['sigma']:.4f}")
            print(f"  [反推] 各模式样本数={g['sizes']}")
            print("  k 扫描 (k, 半径, dist, 组内SS):")
            for k, r, c, ss in g["k_scan"]:
                print(f"    {k:2d}  {r:6.3f}  {c:6.3f}  {ss:9.1f}")

    if "dist" in files:
        d = float(z["dist"])
        c = ring_centers(d, int(z["k"]) if "k" in files else 8)
        sigma = (float(z["sigma"]) if "sigma" in files
                 else float(np.sqrt(((z["train"] - c[((z["train"][:, None, :]
                                 - c[None]) ** 2).sum(-1).argmin(1)]) ** 2)
                                    .sum(1).mean() / 2)))
        info["declared"] = {"dist": d, "sigma": sigma}
        if not quiet:
            print(f"  [声明] dist={d}  sigma={sigma:.3f}  环半径="
                  f"{np.linalg.norm(c, axis=1).mean():.3f}")
    elif "means_radius" in files:
        r = float(z["means_radius"])
        info["declared"] = {"radius": r, "dist": 2 * r * np.sin(np.pi / 8)}
        if not quiet:
            print(f"  [声明] means_radius={r}  相邻中心间距="
                  f"{2 * r * np.sin(np.pi / 8):.3f}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--infer", action="store_true", help="从 train 反推 k/dist/sigma")
    ap.add_argument("--quiet", action="store_true", help="只打印失败项")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.npz")))
        else:
            files.append(p)

    results = [check_one(f, a.infer, a.quiet) for f in files]
    bad = [r for r in results if r["errors"]]
    warn = [r for r in results if r["warnings"] and not r["errors"]]
    print(f"\n共 {len(results)} 个文件：可用 {len(results) - len(bad)}"
          f"（其中 {len(warn)} 个带 WARN），失败 {len(bad)}")
    for r in bad:
        print(f"  FAIL {os.path.basename(r['file'])}: {'; '.join(r['errors'])}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
