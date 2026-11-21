# -*- coding: utf-8 -*-
"""核验 nn_ratio 的 0.05 基线在各 (距离, 种子) 格子是否一致。

对每个格子：train = make_gmm(1000, 8, r, .3, seed)
            calib = make_gmm(10000, 8, r, .3, 10000+seed)  -> tau = 5% 分位
            fresh = make_gmm(2000, 8, r, .3, 20000+seed)   -> baseline = P(nn<tau)
baseline 若在 0.05 附近无系统偏移，则 nn_ratio 不需要再除以基线。
"""
import numpy as np
import pandas as pd

import eval_toolkit as ev

RADII = [1.0, 1.5, 2.0, 3.0, 4.5]
SEEDS = [0, 1, 2, 3, 4]
N_FRESH = 2000          # 和一次运行生成的样本量一致

rows = []
for r in RADII:
    for s in SEEDS:
        train = ev.gmm_sample(1000, ev.seed_train(s), radius=r, sigma=0.3)
        calib = ev.gmm_sample(ev.N_CALIB, ev.seed_calib(s), radius=r, sigma=0.3)
        tau = ev.nn_threshold(calib, train)
        fresh = ev.gmm_sample(N_FRESH, ev.seed_target(s), radius=r, sigma=0.3)
        d1, _ = ev.nn_two(fresh, train)
        rows.append(dict(radius=r, seed=s, tau=tau, baseline=float(np.mean(d1 < tau)),
                         null_med=float(np.median(d1))))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
print("\n按种子汇总（每个种子的 5 个距离）:")
print(df.pivot_table(index="seed", columns="radius", values="baseline").round(4).to_string())
print("\n按距离汇总:")
g = df.groupby("radius")["baseline"].agg(["mean", "std", "min", "max"])
print(g.round(4).to_string())
print(f"\n全部 25 格: mean={df.baseline.mean():.4f} sd={df.baseline.std():.4f} "
      f"range=[{df.baseline.min():.4f}, {df.baseline.max():.4f}]")
print(f"二项标准误 (n={N_FRESH}) = {np.sqrt(0.05*0.95/N_FRESH):.4f}")
print("tau 范围:", f"{df.tau.min():.4f} ~ {df.tau.max():.4f}")
