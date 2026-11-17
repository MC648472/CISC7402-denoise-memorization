# CISC7402 Project — 数据几何结构与模型容量对去噪模型记忆和泛化的影响

Effects of Data Geometry and Model Capacity on Memorization and Generalization in Denoising Models

## 团队分工

| | 姓名 | 邮箱 | GitHub | 负责 |
|---|---|---|---|---|
| **A 同学** | Deng Yunfan | mc64847@um.edu.mo | [MC648472](https://github.com/MC648472) | **评估与统计**：三项指标（mode coverage / MMD / 邻近比）、轨迹追踪、pilot 判读、ANOVA 与效应量、图表，以及论文的“评估指标 / 统计方法 / 结果”三节 |
| **B 同学** | Zhang Lu | mc65458@um.edu.mo | — | **模型与执行**：高斯混合生成器、MLP 去噪网络、DDPM 训练与采样、调度脚本，跑主实验与消融，写论文的“模型与实验设置 / 引言 / 局限” |

代码归属很清晰：

- **B 写训练侧**：`make_gmm` / `Denoiser` / `DDPM` / `train` / `sample` / `run_one` / `run_grid`（在 Colab notebook 里，本仓库不重复存放）。
- **A 写评估侧**：`eval_toolkit.py`（三项指标）、`check_npz.py`（数据验收）、`eval_all.py`（批量评估）、`anova.py`（两因素 ANOVA + 效应量 + 置换检验）、`memtest_eval.py`、`calib_baseline.py`。

实验设计为 5 档距离 × 5 档容量 × 5 个种子 = 125 次训练；另有记忆区（n=40）、密度消融（ablN）、扩散步数消融（abl_T）、宽度扫描（widthsweep）与 memtest。

## 目录

```
code/     评估与统计分析代码
  eval_toolkit.py      三项指标 + 采样参考集（τ 逐格校准）
  check_npz.py         单个 npz 的验收（形状 / 轨迹 / losses / 元数据）
  eval_all.py          按文件名批量算指标 → results/metrics.csv
  anova.py             两因素 ANOVA、偏 η² / 偏 ω²、Freedman–Lane 置换检验
  memtest_eval.py      小样本记忆测试（逐 n 校准 τ）
  calib_baseline.py    0.05 基线的逐格核验
  make_grid.py         75 格待跑清单与进度检查
  make_selfcheck_fig.py  自检数据的诊断图
  pilot_repro.py       在本地 CPU 复现 B 的训练循环（需 torch）
  A_eval_notebook.ipynb  Colab 评估驱动
  math_A_N40.ipynb      记忆区 mainN40 的训练 notebook（A 负责的一半）
results/  算好的指标表（metrics.csv 为全量 359 行）
figures/  诊断图
```

## 复现

```bash
# 1) 训练：用 B 的 Colab notebook（T4 GPU），或本地 CPU 复现
python code/pilot_repro.py --iters 30000 --n-train 40 --radius 1.5 --cap 2 --out ./runs

# 2) 评估：把 npz 目录传进来，统一算三项指标
python code/eval_all.py --roots ./runs --out results/metrics.csv

# 3) 统计
python code/anova.py --csv results/metrics.csv --tag main
```

## 指标定义

- **mode_coverage**：把所有点分到最近的中心，落在 3σ 内的才算；某模式拿到“期望份额”的 ≥10% 记为覆盖，返回被覆盖模式的比例。
- **mmd**：RBF 核的无偏 MMD²，带宽取目标集中位距离的 0.5/1/2 倍再平均，每边最多 3000 点。
- **nn_ratio**：生成样本中，到训练集最近邻距离小于 τ 的比例。**τ 逐格校准**——由该格训练集与同一分布的新样本（种子 10000+seed）算出近邻距离的 5% 分位，所以“完全不记忆”的基线恒为 0.05（n=1000 实测 0.050，n=40 实测 0.048）。

## 数据

原始 `.npz` / `.pt` / `.ckpt` 不入库（见 `.gitignore`），存在共享网盘。仓库里只保留算好的 `results/*.csv`。
