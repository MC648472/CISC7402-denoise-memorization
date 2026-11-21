# -*- coding: utf-8 -*-
"""Diagnostic figure for one run: samples, deviation from the modes, and the
reverse-diffusion trajectory.

    python make_selfcheck_fig.py RUN.npz out.png [radius]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_toolkit as ev

npz = sys.argv[1] if len(sys.argv) > 1 else "selfcheck_samples.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "selfcheck_diagnostic.png"
dist = float(sys.argv[3]) if len(sys.argv) > 3 else 3.438

z = np.load(npz)
final, traj, train = z["final"], z["traj"], z["train"]
centers = ev.gmm_centers(dist)
sigma = ev.estimate_sigma(train, centers)
ref = ev.reference_sets(dist, sigma, 0)
tau = ev.nn_threshold(ref["calib"], train)
tm = ev.traj_metrics(traj, train, tau, centers, sigma)

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

ax[0].scatter(*train.T, s=3, c="0.6", alpha=.5, label="train (n=%d)" % len(train))
ax[0].scatter(*final.T, s=3, c="tab:blue", alpha=.35,
              label="generated (n=%d)" % len(final))
ax[0].scatter(*centers.T, marker="x", c="k", s=60, label="modes")
ax[0].set_aspect("equal"); ax[0].legend(fontsize=7); ax[0].set_title("samples")

lab_t = ev.assign_modes(train, centers)
lab_f = ev.assign_modes(final, centers)
r_t = np.linalg.norm(train - centers[lab_t], axis=1)
r_f = np.linalg.norm(final - centers[lab_f], axis=1)
bins = np.linspace(0, 1.6, 60)
ax[1].hist(r_t, bins=bins, density=True, alpha=.55, color="0.5",
           label=r"train  $\sigma$=%.2f" % sigma)
ax[1].hist(r_f, bins=bins, density=True, alpha=.55, color="tab:blue",
           label=r"generated  $\sigma$=%.2f" % ev.estimate_sigma(final, centers))
for x, c in [(sigma, "0.3"), (2 * sigma, "0.3")]:
    ax[1].axvline(x, ls="--", c=c, lw=1)
ax[1].set_xlabel(r"|x - nearest mode|"); ax[1].set_title("over-dispersion")
ax[1].legend(fontsize=7)

ax[2].plot(tm["radius"], "o-", color="tab:green")
ax[2].set_xlabel("snapshot (0 = pure noise)"); ax[2].set_ylabel("mean |x|")
ax[2].grid(alpha=.3); ax[2].set_title("ring radius along the reverse process")

ax[3].plot(tm["core_hit"] if "core_hit" in tm else tm["mode_hit"], "o-",
           color="tab:red", label="within 3$\\sigma$ of a mode")
ax[3].plot(tm["nn_ratio"], "s-", color="tab:purple", label="nn_ratio")
ax[3].set_xlabel("snapshot"); ax[3].set_title("mode hit / memorisation")
ax[3].legend(fontsize=7); ax[3].grid(alpha=.3)

fig.suptitle("B selfcheck run: 8-mode ring, dist=%.2f, data sigma=%.2f, "
             "traj=%d snapshots" % (dist, sigma, traj.shape[0]), fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("saved", out)
print("落在 1σ 内的样本比例: %.3f（完美模型 0.393）" % float(np.mean(r_f <= sigma)))
print("生成/训练的 sigma 比  : %.2f（1.0 = 没被抹平）"
      % (ev.estimate_sigma(final, centers) / sigma))
