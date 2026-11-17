# -*- coding: utf-8 -*-
"""两因素 ANOVA（平衡设计）+ 效应量 + bootstrap CI + Freedman-Lane 置换检验。

  python anova.py --csv metrics_all.csv --tag mainN40

输出：三个效应的 SS/df/MS/F/p、偏 η²（含 bootstrap 95%CI）、置换 p、Levene 检验。
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

METRICS = ["mode_coverage", "mmd", "nn_ratio"]


def twoway(y, a, b):
    """平衡两因素方差分解，返回各效应的 SS 与 df。"""
    df = pd.DataFrame({"y": y, "a": a, "b": b})
    ybar = df.y.mean()
    cell = df.groupby(["a", "b"]).y.mean()
    am = df.groupby("a").y.mean()
    bm = df.groupby("b").y.mean()
    n = df.groupby(["a", "b"]).size().iloc[0]
    A, B = len(am), len(bm)
    ss_a = n * B * ((am - ybar) ** 2).sum()
    ss_b = n * A * ((bm - ybar) ** 2).sum()
    ss_ab = n * sum((cell[i, j] - am[i] - bm[j] + ybar) ** 2 for i, j in cell.index)
    fit = df.set_index(["a", "b"]).index.map(cell)
    ss_e = float(((y - np.asarray(fit)) ** 2).sum())
    return dict(ss_a=ss_a, ss_b=ss_b, ss_ab=ss_ab, ss_e=ss_e,
                df_a=A - 1, df_b=B - 1, df_ab=(A - 1) * (B - 1),
                df_e=A * B * (n - 1), A=A, B=B, n=n, ybar=ybar, am=am, bm=bm, cell=cell)


def f_values(y, a, b):
    d = twoway(y, a, b)
    ms_e = max(d["ss_e"] / d["df_e"], 1e-12)
    return {"a": (d["ss_a"] / d["df_a"]) / ms_e,
            "b": (d["ss_b"] / d["df_b"]) / ms_e,
            "ab": (d["ss_ab"] / d["df_ab"]) / ms_e}, d


def perm_p(y, a, b, effect, n_perm=2000, seed=0):
    """Freedman-Lane：在去掉该效应的模型残差上置换。"""
    rng = np.random.default_rng(seed)
    obs, d = f_values(y, a, b)
    ybar, am, bm, cell = d["ybar"], d["am"], d["bm"], d["cell"]
    idx = pd.MultiIndex.from_arrays([a, b])
    if effect == "a":                     # 去掉 a：拟合 = ybar + b_j
        fitted = np.asarray(bm.loc[b])
    elif effect == "b":                   # 去掉 b：拟合 = ybar + a_i
        fitted = np.asarray(am.loc[a])
    else:                                 # 去掉交互：拟合 = ybar + a_i + b_j
        fitted = np.asarray(am.loc[a]) + np.asarray(bm.loc[b]) - ybar
    resid = y - fitted
    hits = 0
    for _ in range(n_perm):
        ystar = fitted + rng.permutation(resid)
        if f_values(ystar, a, b)[0][effect] >= obs[effect]:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def eta2_ci(y, a, b, effect, n_boot=1000, seed=1):
    """按种子做整块 bootstrap：每次有放回抽 5 个种子（连同 5×5 结构一起），
    这样既扰动格均值也扰动格内噪声，比格内重抽更贴合设计。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "a": a, "b": b, "seed": seed_col})
    seeds = np.array(sorted(df.seed.unique()))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        sub = pd.concat([df[df.seed == s] for s in pick], ignore_index=True)
        _, d = f_values(sub.y.values, sub.a.values, sub.b.values)
        ss = {"a": d["ss_a"], "b": d["ss_b"], "ab": d["ss_ab"]}[effect]
        out.append(ss / (ss + d["ss_e"]) if ss > 0 else 0.0)
    obs, d = f_values(y, a, b)
    ss = {"a": d["ss_a"], "b": d["ss_b"], "ab": d["ss_ab"]}[effect]
    return ss / (ss + d["ss_e"]), np.percentile(out, [2.5, 97.5])


def partial_omega2(ss_eff, df_eff, ss_e, df_e, n_total):
    """偏 ω²：对小样本/少重复更诚实，点估计不会系统性贴区间边。"""
    ms_eff = ss_eff / df_eff
    ms_e = max(ss_e / df_e, 1e-15)
    num = df_eff * (ms_eff - ms_e)
    den = df_eff * ms_eff + (n_total - df_eff) * ms_e
    return float(num / den) if den > 0 else float("nan")


def run(csv, tag, dv, fa="radius", fb="cap_idx"):
    m = pd.read_csv(csv)
    m = m[(m.tag == tag) & m[dv].notna()]
    if len(m) == 0:
        print(f"{tag}: 没有 {dv} 数据"); return
    global seed_col
    seed_col = m["seed"].values
    y, a, b = m[dv].values, m[fa].values, m[fb].values
    f, d = f_values(y, a, b)
    ms_e = d["ss_e"] / d["df_e"]
    names = {"a": f"{fa}", "b": f"{fb}", "ab": f"{fa}×{fb}"}
    dfs = {"a": d["df_a"], "b": d["df_b"], "ab": d["df_ab"]}
    print(f"\n=== {tag} | DV = {dv} | n = {len(m)} ===")
    print(f"{'效应':<16}{'SS':>12}{'df':>6}{'F':>9}{'p':>10}{'偏η²':>8}{'偏ω²':>8}"
          f"{'95%CI(boot)':>18}{'置换p':>9}")
    ss_map = {"a": d["ss_a"], "b": d["ss_b"], "ab": d["ss_ab"]}
    for eff in ("a", "b", "ab"):
        p = stats.f.sf(f[eff], dfs[eff], d["df_e"])
        e, ci = eta2_ci(y, a, b, eff)
        w2 = partial_omega2(ss_map[eff], dfs[eff], d["ss_e"], d["df_e"], len(y))
        pp = perm_p(y, a, b, eff)
        ptxt = "< 0.001" if pp < 0.001 else f"{pp:.3f}"
        print(f"{names[eff]:<16}{ss_map[eff]:>12.4g}"
              f"{dfs[eff]:>6}{f[eff]:>9.2f}{p:>10.3g}{e:>8.3f}{w2:>8.3f}"
              f"   [{ci[0]:.3f}, {ci[1]:.3f}]{ptxt:>10}")
    # 残差诊断
    fitted = np.asarray(d["cell"].loc[list(zip(a, b))])
    resid = y - fitted
    groups = [resid[(a == i) & (b == j)] for i in sorted(set(a)) for j in sorted(set(b))]
    lev = stats.levene(*groups)
    print(f"Levene: W={lev.statistic:.2f}, p={lev.pvalue:.3g}  "
          f"| 残差 sd={resid.std():.4g}")
    if dv == "nn_ratio":
        print("按容量的均值:", dict(zip(*[list(x) for x in
              zip(*m.groupby(fb)[dv].mean().items())])))
    if dv == "mmd":
        print("按容量的均值:", {k: round(v, 5) for k, v in m.groupby(fb)[dv].mean().items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metrics_all.csv")
    ap.add_argument("--tag", default="mainN40")
    ap.add_argument("--dvs", nargs="*", default=["mmd", "nn_ratio"])
    a = ap.parse_args()
    for dv in a.dvs:
        run(a.csv, a.tag, dv)
