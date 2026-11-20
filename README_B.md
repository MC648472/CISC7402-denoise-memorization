# Data geometry and model capacity in denoising diffusion models

Two-person course project (CISC7402). We study how cluster distance and MLP capacity affect
memorization and generalization of a DDPM on a 2D Gaussian mixture (8 modes equally spaced on a circle).

## Code
- `notebooks/B_experiments.ipynb`: data generator, MLP denoiser, DDPM training/sampling (with trajectory
  snapshots), batch runner with checkpoint resume. Open in Google Colab (GPU) and run the cells in order.
- `src/extra_experiments_B.py`: extra runners (width/depth sweeps, sample-size and step ablations)
  and the denoising-loss gap diagnostic. Run after the cells above.
- Sampling interface: `sample(model, n, return_trajectory=True)` returns `final` (n,2) and `traj` (10,n,2).
- Frozen settings: distances (circle radius) 1.0/1.5/2.0/3.0/4.5; capacities (depth,width)
  (1,8) (2,16) (3,64) (4,256) (5,512); 8 modes, sigma 0.3, T=100, seeds 0-4.
  Re-running cell 1 resets the values, so set them again before launching runs.

## Experiments
| tag | n_train | steps | design |
|---|---|---|---|
| main | 1000 | 10000 | 5 distances x 5 capacities x 5 seeds |
| mainN40 | 40 | 30000 | same |
| widthsweep | 40 | 30000 | depth 3, width 4-512, r=1.5/3.0, seeds 0-2 |
| depthsweep, depthsweep512 | 40 | 30000 | width 256 (depth 1-6), width 512 (depth 3/4/6) |
| ablN* | 20-1000 | 30000 | capacity (4,256), r=1.5/3.0 |
| abl_T* | 1000 | 10000 | T = 50, 100, 200 |
| iter60k | 40 | 60000 | (4,256) and (5,512), r=1.5/3.0 |

## Results and data
- `results/`: summary tables for the denoising-loss gap. `figures/`: figures used in the report.
- Sample files (.npz) and weights (.pt) are not stored here (size). Contact the authors.
- Metrics (mode coverage, MMD, nearest-neighbour ratio) and ANOVA: partner's code in `code/` (see `README.md`).
