# CISC7402 Project — 数据几何结构与模型容量对去噪模型记忆和泛化的影响

Effects of Data Geometry and Model Capacity on Memorization and Generalization in Denoising Models

## 团队分工

| | 姓名 | 邮箱 | GitHub | 负责 |
|---|---|---|---|---|
| **A 同学** | Deng Yunfan | mc64847@um.edu.mo | [MC648472](https://github.com/MC648472) | **评估与统计**：三项指标（mode coverage / MMD / 邻近比）、轨迹追踪、pilot 判读、ANOVA 与效应量、图表，以及论文的“评估指标 / 统计方法 / 结果”三节 |
| **B 同学** | Zhang Lu | mc65458@um.edu.mo | — | **模型与执行**：高斯混合生成器、MLP 去噪网络、DDPM 训练与采样、调度脚本，跑主实验与消融，写论文的“模型与实验设置 / 引言 / 局限” |

代码归属：

- **训练侧（B）**：`notebooks/B_experiments.ipynb`（`make_gmm` / `Denoiser` / `DDPM` / `train` / `sample` / `run_one` / `run_grid`）与 `src/extra_experiments_B.py`（宽度/深度扫描、样本量与步数消融、损失差诊断）。详见 `README_B.md`。
- **评估侧（A）**：`code/` 下的指标、验收、批量评估、统计与绘图脚本。

## 仓库结构

```
code/                 A 的评估与统计代码
  eval_toolkit.py       三项指标 + 采样参考集（tau 逐格校准）
  check_npz.py          单个 npz 的验收
  eval_all.py           按文件名批量算指标 → results/metrics.csv
  anova.py              两因素 ANOVA、偏 eta^2 / 偏 omega^2、Freedman-Lane 置换
  make_tables.py        生成 results/tables/ 下的统计表
  make_depth_fig.py     深度 x 宽度折线图
  memtest_eval.py       小样本记忆测试（逐 n 校准 tau）
  calib_baseline.py     0.05 基线的逐格核验
  make_grid.py          75 格待跑清单与进度检查
  make_selfcheck_fig.py 自检数据诊断图
  pilot_repro.py        本地 CPU 复现训练循环
  A_eval_notebook.ipynb Colab 评估驱动
  math_A_N40.ipynb      记忆区 mainN40 的训练 notebook
notebooks/            B 的实验 notebook（Colab，T4）
src/                  B 的额外实验脚本
results/
  metrics.csv           全量指标（413 行）：main / mainN40 / widthsweep / depthsweep /
                        ablN / ablT / memtest / pilot2
  tables/               统计表（ANOVA、5x5、H3 增量、消融、深度x宽度）
  gap_*.csv             B 的去噪损失差数据
figures/              深度x宽度折线、自检诊断图、B 的损失/散点图
requirements.txt
```

## 实验设计

5 档距离（环半径 1.0 / 1.5 / 2.0 / 3.0 / 4.5）× 5 档容量（(1,8) … (5,512)）× 5 个种子 = 125 次；
另有记忆区 `mainN40`（n=40、30000 步，125 次）、宽度扫描（depth=3，宽度 4–512）、
深度扫描（宽度 256 深度 1–6；宽度 512 深度 3/4/6）、密度消融 `ablN`（n=20–1000）、
扩散步数消融 `abl_T`（T=50/100/200）、小样本记忆测试 `memtest`、60000 步对照 `iter60k`。

## 复现

```bash
# 训练（B）：在 Colab 打开 notebooks/B_experiments.ipynb，按单元顺序跑；
#   或本地 CPU 复现同一训练循环
python code/pilot_repro.py --iters 30000 --n-train 40 --radius 1.5 --cap 2 --out ./runs

# 评估（A）：把 npz 目录传进来
python code/eval_all.py --roots ./runs --out results/metrics.csv

# 统计与图表
python code/make_tables.py --csv results/metrics.csv --out results/tables
python code/anova.py --csv results/metrics.csv --tag main
python code/make_depth_fig.py --csv results/metrics.csv --out figures/depth_width.png
```

## 指标定义

- **mode_coverage**：把点分到最近的中心，落在 3σ 内的才算；某模式拿到“期望份额”的 ≥10% 记为覆盖，返回被覆盖模式的比例。
- **mmd**：RBF 核的无偏 MMD²，带宽取目标集中位距离的 0.5/1/2 倍再平均，每边最多 3000 点。
- **nn_ratio**：生成样本中，到训练集最近邻距离小于 τ 的比例。**τ 逐格校准**——由该格训练集与同一分布的新样本（种子 10000+seed）算出近邻距离的 5% 分位，因此“完全不记忆”的基线恒为 0.05（n=1000 实测 0.050，n=40 实测 0.048）。

## 数据

原始 `.npz` / `.pt` / `.ckpt` 不入库（见 `.gitignore`），存在共享网盘。仓库里只保留算好的 `results/*.csv` 与图。
