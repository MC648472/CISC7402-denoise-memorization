"""
eval_toolkit.py  —— A 同学的评估工具包（v2）
数据几何结构与模型容量对去噪模型记忆和泛化的影响

和 v1 的区别（都是为了 B 的 selfcheck_samples.npz 实测出来的问题）：
  1. SIGMA 不再硬编码为 1.0。B 的数据实测 sigma≈0.298（见 2026-11-17 的 selfcheck），
     硬编码 1.0 会让 mode_coverage 的半径从 0.9 变成 3.0，小距离档直接饱和、
     轨迹图的 mode_hit 在 t=0 就变成 0.3。
     现在按优先级取：npz 里的 sigma 字段 > 从 train 估出来 > SIGMA 默认值。
  2. .npz 缺 metadata 也能算（用 train 反推），但 check_npz.py 会把缺字段判成 fail，
     逼 B 补齐 dist / seed / losses，否则 CSV 对不上号。
  3. 指标仍然是约定的三项：mode_coverage、MMD、nn_ratio。不额外加列。
  4. calib / target 按 (dist, sigma, seed) 缓存，125 次网格不会重复生成。

用法：
  python eval_toolkit.py selftest
  python eval_toolkit.py score  selfcheck_samples.npz --dist 3.44
  python eval_toolkit.py fill   results_A.csv
  python eval_toolkit.py pilot  results_pilot.csv --cap-col width
"""
import argparse
import math
import os
import sys
import time
import numpy as np
import pandas as pd

try:                      # Windows 控制台默认 GBK，中文会乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# =====================================================================
# 0. 全局约定（和 B 对齐后只改这里）
# =====================================================================
K_MODES = 8            # 高斯混合分量数（B 的 selfcheck 实测 k=8）
SIGMA = 0.30           # 各分量标准差；B 的 selfcheck 实测 0.298。仅作兜底
N_CALIB = 10000        # 密度校准集大小
N_TARGET = 10000       # 目标参考集大小
NN_QUANTILE = 0.05     # 近邻阈值分位数
MODE_RADIUS_K = 3.0    # 样本落入某模式的半径 = 3 sigma
MODE_MIN_FRAC = 0.1    # 某模式至少拿到"期望份额"的 10% 才算被覆盖
MMD_MAX_N = 3000       # MMD 每边最多取多少点

METRIC_COLS = ["mode_coverage", "MMD", "nn_ratio"]      # 进 results_*.csv 的三列


def seed_train(s):  return s
def seed_calib(s):  return 10000 + s
def seed_target(s): return 20000 + s


# =====================================================================
# 1. 高斯混合生成器（与 B 的 make_gmm 同一族）
#    B 的 make_gmm(n, K, radius, sigma, seed) 里，radius 是"环半径"；
#    相邻两个模式中心的间距（弦长）chord = 2*radius*sin(pi/K)。
#    K=8 时 chord = 0.7654 * radius。本文件内部一律用 chord，称 dist。
# =====================================================================
def chord(radius, k=K_MODES):
    return 2.0 * float(radius) * math.sin(math.pi / k)


def gmm_centers(dist, k=K_MODES):
    r = dist / (2 * np.sin(np.pi / k))
    ang = 2 * np.pi * np.arange(k) / k
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def gmm_sample(n, seed, dist=None, k=K_MODES, sigma=None, radius=None):
    """dist = 相邻中心间距（chord）；也可以直接给 B 的 radius。"""
    if radius is not None:
        dist = chord(radius, k)
    rng = np.random.default_rng(seed)
    c = gmm_centers(dist, k)
    lab = rng.integers(0, k, size=n)
    s = SIGMA if sigma is None else sigma
    return c[lab] + s * rng.standard_normal((n, 2))


def make_sets(dist, seed, n_train, gen=gmm_sample, sigma=None):
    return {"train": gen(n_train, seed_train(seed), dist, sigma=sigma),
            "calib": gen(N_CALIB, seed_calib(seed), dist, sigma=sigma),
            "target": gen(N_TARGET, seed_target(seed), dist, sigma=sigma)}


# =====================================================================
# 2. 基础距离工具
# =====================================================================
def _sqdist(A, B):
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T


def nn_two(X, T, chunk=2000):
    """X 中每个点到 T 的最近、次近距离 (d1, d2)。"""
    d1 = np.empty(len(X)); d2 = np.empty(len(X))
    for i in range(0, len(X), chunk):
        D = np.maximum(_sqdist(X[i:i + chunk], T), 0)
        part = np.partition(D, 1, axis=1)[:, :2]
        part.sort(axis=1)
        d1[i:i + chunk] = np.sqrt(part[:, 0])
        d2[i:i + chunk] = np.sqrt(part[:, 1])
    return d1, d2


def nn_threshold(calib, train, q=NN_QUANTILE):
    d1, _ = nn_two(calib, train)
    return float(np.quantile(d1, q))


def nn_ratio(final, train, tau):
    d1, _ = nn_two(final, train)
    return float(np.mean(d1 < tau))


# =====================================================================
# 3. 指标
# =====================================================================
def mode_coverage(final, centers, sigma=None, radius_k=MODE_RADIUS_K,
                  min_frac=MODE_MIN_FRAC):
    s = SIGMA if sigma is None else sigma
    k = len(centers)
    D = _sqdist(final, centers)
    lab = D.argmin(1)
    inside = np.sqrt(np.maximum(D.min(1), 0)) <= radius_k * s
    counts = np.bincount(lab[inside], minlength=k)
    return float(np.mean(counts >= min_frac * len(final) / k))


def _rbf(A, B, bw):
    return np.exp(-np.maximum(_sqdist(A, B), 0) / (2 * bw ** 2))


def median_bandwidth(Y, max_n=2000, seed=0):
    rng = np.random.default_rng(seed)
    Z = Y[rng.choice(len(Y), min(max_n, len(Y)), replace=False)]
    D = np.sqrt(np.maximum(_sqdist(Z, Z), 0))
    return float(np.median(D[np.triu_indices(len(Z), 1)]))


def mmd2(X, Y, bw_scales=(0.5, 1.0, 2.0), max_n=MMD_MAX_N, seed=0):
    """无偏 MMD^2，RBF 核，带宽 = 目标集中位数距离 × bw_scales，取平均。"""
    rng = np.random.default_rng(seed)
    if len(X) > max_n: X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n: Y = Y[rng.choice(len(Y), max_n, replace=False)]
    base = median_bandwidth(Y)
    n, m = len(X), len(Y)
    vals = []
    for s in bw_scales:
        bw = base * s
        Kxx, Kyy, Kxy = _rbf(X, X, bw), _rbf(Y, Y, bw), _rbf(X, Y, bw)
        vals.append((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
                    + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
                    - 2 * Kxy.mean())
    return float(np.mean(vals))


# =====================================================================
# 4. 几何量：从数据本身反推 sigma / 认领 centers
# =====================================================================
def assign_modes(X, centers):
    return np.argmin(_sqdist(X, centers), axis=1)


def estimate_sigma(X, centers):
    """各点相对最近中心的“每轴标准差”。各向同性高斯下 = 生成器的 sigma。"""
    lab = assign_modes(X, centers)
    dev = X - centers[lab]
    return float(np.sqrt((dev ** 2).sum(1).mean() / 2))


def local_radius(train, k=10):
    """每个训练点到它第 k 个邻居的距离 = 局部密度尺度 rho（诊断用）。"""
    D = np.maximum(_sqdist(train, train), 0)
    np.fill_diagonal(D, np.inf)
    return np.sqrt(np.partition(D, k - 1, axis=1)[:, k - 1])


def proximity(X, train, rho):
    """到最近训练点的距离 / 那个训练点的局部 k-NN 半径（诊断用，不进 CSV）。"""
    D = np.maximum(_sqdist(X, train), 0)
    j = D.argmin(1)
    return np.sqrt(D[np.arange(len(X)), j]) / np.maximum(rho[j], 1e-12)


def overdispersion(final, centers, train):
    """诊断：生成样本的 sigma / 训练集的 sigma。1.0 = 没抹平。"""
    lab = assign_modes(final, centers)
    sig_eff = float(np.sqrt(((final - centers[lab]) ** 2).sum(1).mean() / 2))
    return sig_eff / max(estimate_sigma(train, centers), 1e-9)


# =====================================================================
# 5. 轨迹追踪   traj: (T, n, 2)，traj[0] 纯噪声，traj[-1] 最终样本
# =====================================================================
def traj_metrics(traj, train, tau, centers, sigma=None, max_pts=1000, seed=0):
    s = SIGMA if sigma is None else sigma
    rng = np.random.default_rng(seed)
    idx = rng.choice(traj.shape[1], min(max_pts, traj.shape[1]), replace=False)
    out = {"mean_nn_dist": [], "nn_ratio": [], "mode_hit": [], "radius": []}
    for t in range(traj.shape[0]):
        X = traj[t][idx]
        d1, _ = nn_two(X, train)
        D = _sqdist(X, centers)
        dm = np.sqrt(np.maximum(D.min(1), 0))
        out["mean_nn_dist"].append(d1.mean())
        out["nn_ratio"].append(np.mean(d1 < tau))
        out["mode_hit"].append(np.mean(dm <= MODE_RADIUS_K * s))
        out["radius"].append(np.linalg.norm(X, axis=1).mean())
    return {k: np.array(v) for k, v in out.items()}


# =====================================================================
# 6. 读一个 .npz，解析几何，做完整评估
# =====================================================================
def load_run(npz_path):
    z = np.load(npz_path)
    meta = {k: (z[k].item() if z[k].ndim == 0 else z[k])
            for k in z.files if k not in ("final", "traj", "train", "losses")}
    return {"final": z["final"] if "final" in z.files else None,
            "train": z["train"] if "train" in z.files else None,
            "traj": z["traj"] if "traj" in z.files else None,
            "losses": z["losses"] if "losses" in z.files else None,
            "meta": meta, "path": npz_path}


_REF_CACHE = {}


def reference_sets(dist, sigma, seed, k=K_MODES):
    """(dist, sigma, seed) 只生成一次 calib / target。"""
    key = (round(float(dist), 6), round(float(sigma), 6), int(seed), int(k))
    if key not in _REF_CACHE:
        _REF_CACHE[key] = {
            "calib": gmm_sample(N_CALIB, seed_calib(seed), dist, k=k, sigma=sigma),
            "target": gmm_sample(N_TARGET, seed_target(seed), dist, k=k, sigma=sigma),
        }
    return _REF_CACHE[key]


def resolve_geometry(run, dist=None, k=K_MODES):
    """返回 (centers, sigma, dist, source)。优先级：npz 元数据 > 数据集里估 > 全局默认。"""
    meta, train = run["meta"], run["train"]
    d = dist if dist is not None else meta.get("dist", None)
    if d is not None and not np.isfinite(d):
        d = None
    kk = int(meta.get("k", meta.get("K", k)))
    centers = meta.get("centers", None)
    if centers is None and d is not None:
        centers = gmm_centers(float(d), kk)
    sigma, src = None, "meta"
    if "sigma" in meta:
        sigma = float(meta["sigma"])
    elif centers is not None and train is not None:
        sigma, src = estimate_sigma(train, centers), "estimated from train"
    if sigma is None:
        sigma, src = SIGMA, "default"
    return centers, sigma, d, src


def _isnan(x):
    return x is None or (isinstance(x, float) and not np.isfinite(x)) or (
        np.isscalar(x) and not isinstance(x, str) and not np.isfinite(x))


def evaluate_run(npz_path, dist=None, seed=None, radius=None, with_traj=False,
                 k=K_MODES):
    """dist = 相邻中心间距；也可以给 radius（B 的环半径），内部转成 dist。"""
    if radius is not None:
        dist = chord(float(radius), k)
    run = load_run(npz_path)
    final, train = run["final"], run["train"]
    centers, sigma, d, src = resolve_geometry(run, dist, k)
    if _isnan(seed):
        seed = int(run["meta"].get("seed", 0))
    seed = int(seed)
    ref = reference_sets(d, sigma, seed, k)
    tau = nn_threshold(ref["calib"], train)
    res = {
        "mode_coverage": mode_coverage(final, centers, sigma),
        "MMD": mmd2(final, ref["target"]),
        "nn_ratio": nn_ratio(final, train, tau),
    }
    res["_sigma"] = sigma
    res["_sigma_source"] = src
    res["_overdispersion"] = overdispersion(final, centers, train)
    if with_traj and run["traj"] is not None:
        res["traj"] = traj_metrics(run["traj"], train, tau, centers, sigma)
    return res


def validate_npz(npz_path, min_snapshots=8):
    """验收逻辑的唯一实现在 check_npz.py（那份是单文件，B 直接能跑）。这里返回 errors。"""
    import check_npz
    errors, _ = check_npz.validate(npz_path)
    return errors


def _metric_col(df, canonical):
    """B 的 CSV 用 mmd（小写），A 自己建的表用 MMD，两边都能写。"""
    low = canonical.lower()
    if low in df.columns:
        return low
    if canonical in df.columns:
        return canonical
    return low if canonical == "MMD" else canonical


def _resolve_path(p, relink):
    """B 的 npz_path 是他在 Colab 上的绝对路径；用 relink 目录按文件名重定位。"""
    if isinstance(p, str) and os.path.exists(p):
        return p
    if relink and isinstance(p, str) and p:
        q = os.path.join(relink, os.path.basename(p))
        if os.path.exists(q):
            return q
    return p


METRICS_COLS = ["tag", "dist_idx", "cap_idx", "seed",
                "mode_coverage", "mmd", "nn_ratio"]      # metrics.csv 的列（B 的约定）


def _find_npz(path, npz_root, tag):
    """B 的 npz_path 是他 Colab 上的绝对路径；按 <npz_root>/<tag>/<文件名> 找本地文件。"""
    if isinstance(path, str) and os.path.exists(path):
        return path
    base = os.path.basename(str(path))
    for cand in ([os.path.join(npz_root, tag, base), os.path.join(npz_root, base)]
                 if npz_root else []):
        if os.path.exists(cand):
            return cand
    return None


def build_metrics(csvs, out="metrics.csv", npz_root="diffusion_project",
                  tags=None, append=True):
    """按 B 的约定，从 results_*.csv 读 npz_path，算三项指标，另存 metrics.csv。
    不改 B 的结果文件；重复运行只补没算过的 (tag, dist_idx, cap_idx, seed)。"""
    frames = [pd.read_csv(c) for c in csvs]
    df = pd.concat(frames, ignore_index=True)
    if tags:
        df = df[df["tag"].isin(tags)]

    done = set()
    if append and os.path.exists(out):
        old = pd.read_csv(out)
        done = {(r.tag, int(r.dist_idx), int(r.cap_idx), int(r.seed))
                for r in old.itertuples()}

    rows, missing = [], []
    t0 = time.time()
    for n, r in enumerate(df.itertuples(), 1):
        key = (r.tag, int(r.dist_idx), int(r.cap_idx), int(r.seed))
        if key in done:
            continue
        p = _find_npz(r.npz_path, npz_root, r.tag)
        if p is None:
            missing.append(r.npz_path)
            continue
        m = evaluate_run(p, radius=float(r.radius), seed=int(r.seed),
                         k=int(getattr(r, "K", K_MODES)))
        rows.append({"tag": r.tag, "dist_idx": int(r.dist_idx),
                     "cap_idx": int(r.cap_idx), "seed": int(r.seed),
                     "mode_coverage": m["mode_coverage"], "mmd": m["MMD"],
                     "nn_ratio": m["nn_ratio"]})
        if n % 10 == 0:
            print(f"  {n}/{len(df)}  {time.time()-t0:6.1f}s", flush=True)

    if rows:
        new = pd.DataFrame(rows)
        if append and os.path.exists(out):
            new = pd.concat([pd.read_csv(out), new], ignore_index=True)
        new = new.drop_duplicates(["tag", "dist_idx", "cap_idx", "seed"], keep="last")
        new = new.sort_values(["tag", "dist_idx", "cap_idx", "seed"])
        new.to_csv(out, index=False)
        print(f"已写出 {out}: {len(new)} 行（本次新增 {len(rows)}）")
    else:
        print(f"没有新增行；{out} 保持 {len(done)} 行")
    if missing:
        print(f"[警告] {len(missing)} 个 npz 找不到，例: {missing[:3]}")
    print(f"耗时 {time.time()-t0:.1f}s")
    return pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()


# =====================================================================
# 7. Pilot 分析
# =====================================================================
def pilot_report(df, dist_col="dist", cap_col="width", sat_tol=0.10):
    cols = {m: _metric_col(df, m) for m in METRIC_COLS}
    g = df.groupby([dist_col, cap_col])[list(cols.values())].mean()
    for m in METRIC_COLS:
        print(f"\n=== {m}（行={dist_col}，列={cap_col}）===")
        print(g[cols[m]].unstack(cap_col).round(4).to_string())

    cov = df.groupby(dist_col)[cols["mode_coverage"]].mean().sort_index()
    if len(cov) > 1:
        diffs = cov.diff().dropna()
        j = diffs.abs().idxmax()
        prev = cov.index[list(cov.index).index(j) - 1]
        print(f"\n[崩塌区间] mode_coverage 变化最大的区间: {prev} → {j}"
              f"（{cov[prev]:.2f} → {cov[j]:.2f}）")

    mmd = g[cols["MMD"]].unstack(cap_col).sort_index(axis=1)
    print(f"\n[饱和容量点] 每个距离下，MMD 达到总改善 {int((1-sat_tol)*100)}% 的最小容量:")
    for d, row in mmd.iterrows():
        gain = row.iloc[0] - row.min()
        if gain <= 0:
            print(f"  {dist_col}={d}: MMD 随容量无改善（可能始终崩塌或早已饱和）")
            continue
        ok = row[(row.iloc[0] - row) >= (1 - sat_tol) * gain]
        print(f"  {dist_col}={d}: 容量 {ok.index[0]}")

    nnr = df.groupby(cap_col)[cols["nn_ratio"]].mean().sort_index()
    if len(nnr) > 1:
        dif = nnr.diff().dropna()
        j = dif.idxmax()
        prev = nnr.index[list(nnr.index).index(j) - 1]
        print(f"\n[记忆出现点] nn_ratio 升幅最大的容量区间: {prev} → {j}"
              f"（{nnr[prev]:.3f} → {nnr[j]:.3f}）")


# =====================================================================
# 8. 假数据自测
# =====================================================================
def selftest():
    import tempfile
    dist, seed, n_train, n = 6.0, 1, 1000, 1000
    sigma = SIGMA
    sets = make_sets(dist, seed, n_train, sigma=sigma)
    train, calib, target = sets["train"], sets["calib"], sets["target"]
    c = gmm_centers(dist)
    tau = nn_threshold(calib, train)
    print(f"tau = {tau:.4f}  (sigma={sigma})")

    rng = np.random.default_rng(123)
    perfect = gmm_sample(n, 777, dist, sigma=sigma)
    memo = train[rng.choice(n_train, n)] + 0.002 * rng.standard_normal((n, 2))
    # 崩塌 A：只留 3 个模式
    collapsed = gmm_sample(4 * n, 888, dist, sigma=sigma)
    lab = _sqdist(collapsed, c).argmin(1)
    collapsed = collapsed[lab < 3][:n]
    # 崩塌 B：环化成均匀分布 —— 三个指标里只有 MMD 看得出来
    ang = rng.uniform(0, 2 * np.pi, n)
    rr = np.linalg.norm(c[0]) + sigma * rng.standard_normal(n)
    ring = np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)

    print(f"{'case':<12}{'coverage':>10}{'MMD':>10}{'nn_ratio':>10}{'抹平度':>8}")
    for name, X in [("perfect", perfect), ("memorize", memo),
                    ("collapse3", collapsed), ("ring", ring)]:
        print(f"{name:<12}{mode_coverage(X, c, sigma):>10.3f}"
              f"{mmd2(X, target):>10.4f}{nn_ratio(X, train, tau):>10.3f}"
              f"{overdispersion(X, c, train):>8.2f}")
    print("预期: perfect≈(1.0, 0.000, 0.05, 1.0)；memorize 的 nn_ratio≈1；"
          "collapse3≈(0.375, 大)；ring 的 coverage 仍是 1.0，只有 MMD 崩")

    noise = rng.standard_normal((n, 2))          # traj[0] 必须是标准正态噪声
    traj = np.stack([(1 - a) * noise + a * perfect for a in np.linspace(0, 1, 10)])
    tm = traj_metrics(traj, train, tau, c, sigma)
    print("\n轨迹 mean_nn_dist:", np.round(tm["mean_nn_dist"], 2))
    print("轨迹 mode_hit    :", np.round(tm["mode_hit"], 3))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.npz")
        np.savez(p, final=traj[-1].copy(), traj=traj, train=train,
                 losses=np.zeros(10), dist=dist, seed=seed, sigma=sigma)
        print("\nvalidate_npz:", validate_npz(p) or "OK")
        r = evaluate_run(p, dist=dist, seed=seed, with_traj=True)
        print("evaluate_run:", {k: (round(v, 4) if isinstance(v, (int, float, np.floating))
                                    else "...") for k, v in r.items()})
        csvp = os.path.join(d, "results_A.csv")
        pd.DataFrame({"tag": ["test"], "dist_idx": [0], "cap_idx": [0],
                      "radius": [dist / (2 * np.sin(np.pi / 8))], "seed": [seed],
                      "npz_path": [p]}).to_csv(csvp, index=False)
        build_metrics([csvp], out=os.path.join(d, "metrics.csv"),
                      npz_root=d, append=False)
        print(pd.read_csv(os.path.join(d, "metrics.csv")).round(4).to_string())

    rows = []
    for dd in [2, 4, 6, 8]:
        for w in [16, 32, 64, 128]:
            cov = min(1, (1.0 if dd >= 6 else (0.5 if dd == 4 else 0.25)) + 0.002 * w)
            rows.append(dict(radius=dd, width=w, mode_coverage=cov,
                             MMD=0.05 * (1 + (8 - dd) / 4) / (1 + w / 20),
                             nn_ratio=0.05 + (0.6 if w >= 128 else 0.02 * w / 32)))
    print("\n--- pilot 分析（假数据）---")
    pilot_report(pd.DataFrame(rows), "radius", "width")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "score", "metrics", "pilot"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="metrics.csv")
    ap.add_argument("--npz-root", default="diffusion_project")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--cap-col", default="width")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "score":
        r = evaluate_run(a.target, a.dist, a.seed, with_traj=True)
        print(f"sigma={r['_sigma']:.4f} ({r['_sigma_source']})")
        for k, v in r.items():
            if not k.startswith("_") and np.isscalar(v):
                print(f"  {k:15s} {v:.5f}")
        for k, v in r["traj"].items():
            print(f"  traj.{k:12s}", np.round(v, 3))
    elif a.cmd == "metrics":
        build_metrics([a.target] if a.target else [], out=a.out,
                      npz_root=a.npz_root, tags=a.tags)
    else:
        pilot_report(pd.read_csv(a.target), "radius", a.cap_col)
