# -*- coding: utf-8 -*-
"""由 results/metrics.csv 生成论文要用的统计表（CSV）。

  results/tables/table_main_5x5.csv       main 的 5×5（nn_ratio / mmd 的均值与 sd）
  results/tables/table_mainN40_5x5.csv    mainN40 的 5×5
  results/tables/table_anova.csv          两因素 ANOVA：SS/df/F/p/偏 η²/偏 ω²/置换 p
  results/tables/table_widthsweep.csv     宽度扫描（depth=3）
  results/tables/table_ablN.csv           密度消融
  results/tables/table_ablT.csv           扩散步数消融
  results/tables/table_h3_delta.csv       H3 方向：每个距离下 cap0 → 峰值的增量

用法： python make_tables.py --csv ../results/metrics.csv --out ../results/tables
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import anova as av

CAP_LABEL = {0: "1x8", 1: "2x16", 2: "3x64", 3: "4x256", 4: "5x512"}


def anova_rows(df, dv):
    y, a, b = df[dv].values, df["radius"].values, df["cap_idx"].values
    f, d = av.f_values(y, a, b)
    out = []
    for eff, name, dfe, ss in (("a", "radius", d["df_a"], d["ss_a"]),
                               ("b", "cap_idx", d["df_b"], d["ss_b"]),
                               ("ab", "radius x cap_idx", d["df_ab"], d["ss_ab"])):
        p = stats.f.sf(f[eff], dfe, d["df_e"])
        eta = ss / (ss + d["ss_e"])
        w2 = av.partial_omega2(ss, dfe, d["ss_e"], d["df_e"], len(y))
        pp = av.perm_p(y, a, b, eff)
        out.append(dict(group=df["group"].iloc[0], dv=dv, effect=name, SS=ss,
                        df=dfe, F=f[eff], p=p, partial_eta2=eta, partial_omega2=w2,
                        perm_p=pp, n=len(y)))
    lev = stats.levene(*[g[dv].values for _, g in df.groupby(["radius", "cap_idx"])])
    for r in out:
        r["levene_p"] = lev.pvalue
        r["resid_sd"] = float((y - np.asarray(d["cell"].loc[list(zip(a, b))])).std())
    return out


def pivot5(df, dv):
    t = df.pivot_table(index="radius", columns="cap_idx", values=dv, aggfunc="mean")
    s = df.pivot_table(index="radius", columns="cap_idx", values=dv, aggfunc="std")
    t.columns = [f"{dv}_{CAP_LABEL[c]}_mean" for c in t.columns]
    s.columns = [f"{dv}_{CAP_LABEL[c]}_sd" for c in s.columns]
    return t.round(6).join(s.round(6)).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results/metrics.csv")
    ap.add_argument("--out", default="../results/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = pd.read_csv(a.csv)

    for tag, group in (("main", "main"), ("mainN40", "mainN40")):
        sub = m[m.tag == tag]
        rows = []
        for dv in ("nn_ratio", "mmd", "mode_coverage"):
            if sub[dv].notna().any() and sub[dv].std() > 0:
                rows += anova_rows(sub, dv)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "table_anova.csv")
                                  if tag == "main" else os.path.join(a.out, "table_anova_n40.csv"),
                                  index=False, float_format="%.6g")
        out = pivot5(sub, "nn_ratio").merge(pivot5(sub, "mmd"), on="radius")
        out.to_csv(os.path.join(a.out, f"table_{group}_5x5.csv"), index=False,
                   float_format="%.6g")
        print(f"{tag}: 5x5 与 ANOVA 已写出")

    anova_all = pd.concat([pd.read_csv(os.path.join(a.out, "table_anova.csv")),
                           pd.read_csv(os.path.join(a.out, "table_anova_n40.csv"))],
                          ignore_index=True)
    anova_all.to_csv(os.path.join(a.out, "table_anova.csv"), index=False, float_format="%.6g")
    os.remove(os.path.join(a.out, "table_anova_n40.csv"))

    # H3：cap0 → 峰值的增量
    rows = []
    for tag in ("main", "mainN40"):
        sub = m[m.tag == tag]
        t = sub.pivot_table(index="radius", columns="cap_idx", values="nn_ratio")
        base = 0.050 if tag == "main" else 0.048
        for r, row in t.iterrows():
            rows.append(dict(group=tag, radius=r, baseline=base, cap0=row.iloc[0],
                             peak_cap=CAP_LABEL[int(row.idxmax())], peak=row.max(),
                             delta_cap0_to_peak=row.max() - row.iloc[0],
                             delta_cap0_to_cap4=row.iloc[-1] - row.iloc[0]))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "table_h3_delta.csv"),
                              index=False, float_format="%.4g")

    w = m[m.tag == "widthsweep"]
    w.groupby("width").agg(n=("seed", "size"), nn_ratio_mean=("nn_ratio", "mean"),
                           nn_ratio_sd=("nn_ratio", "std"),
                           mmd_mean=("mmd", "mean"),
                           coverage_mean=("mode_coverage", "mean")
                           ).reset_index().to_csv(os.path.join(a.out, "table_widthsweep.csv"),
                                                  index=False, float_format="%.4g")

    ab = m[m.group == "ablN"]
    ab.groupby(["n_train", "radius"]).agg(
        n=("seed", "size"), nn_ratio_mean=("nn_ratio", "mean"),
        nn_ratio_sd=("nn_ratio", "std"), mmd_mean=("mmd", "mean"),
        coverage_mean=("mode_coverage", "mean")
    ).reset_index().to_csv(os.path.join(a.out, "table_ablN.csv"),
                           index=False, float_format="%.4g")

    t = m[m.group == "ablT"]
    t[["tag", "radius", "cap_idx", "seed", "T", "mode_coverage", "mmd", "nn_ratio"]] \
        .to_csv(os.path.join(a.out, "table_ablT.csv"), index=False, float_format="%.4g")

    print("写出完毕:", os.listdir(a.out))


if __name__ == "__main__":
    main()
