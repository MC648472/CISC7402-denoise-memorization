# Run AFTER the cells of notebooks/B_experiments.ipynb (same notebook session).
# Always rewrite the frozen values first, because re-running cell 1 resets them:
#   DISTANCES  = [1.0, 1.5, 2.0, 3.0, 4.5]
#   CAPACITIES = [(1, 8), (2, 16), (3, 64), (4, 256), (5, 512)]

# %% any (depth, width) run, saves samples (.npz) and weights (.pt)
def run_custom_save(radius, depth, width, seed, cfg, tag):
    name = f"{tag}_r{radius}_d{depth}_w{width}_s{seed}"
    folder = os.path.join(OUT_DIR, tag); os.makedirs(folder, exist_ok=True)
    npz = os.path.join(folder, name + ".npz"); pt = os.path.join(folder, name + ".pt")
    ckpt = os.path.join(folder, name + ".ckpt")
    if os.path.exists(npz):
        print("skip (done):", name); return
    t0 = time.time()
    model, losses, x_train = train(depth, width, radius, seed, cfg, ckpt)
    final, traj = sample(model, cfg["n_sample"], True, seed=seed, n_traj=cfg["n_traj"])
    torch.save(model.state_dict(), pt)
    np.savez_compressed(npz, final=final, traj=traj, train=x_train,
                        losses=np.array(losses), means_radius=radius)
    append_row(dict(tag=tag, dist_idx=DISTANCES.index(radius), cap_idx=-1, radius=radius,
                    depth=depth, width=width, params=n_params(model), seed=seed,
                    n_train=cfg["n_train"], T=cfg["T"], K=cfg["K"],
                    final_loss=round(float(np.mean(losses[-200:])), 5),
                    seconds=round(time.time() - t0, 1), npz_path=npz))
    if os.path.exists(ckpt): os.remove(ckpt)

# %% denoising-loss gap: loss on fresh samples minus loss on the training set
def denoise_loss(model, x_np, n_rep=100, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    d = model.ddpm
    x = torch.tensor(x_np, device=DEVICE)
    model.eval(); tot = 0.0
    with torch.no_grad():
        for _ in range(n_rep):
            t = torch.randint(0, d.T, (len(x),), device=DEVICE, generator=g)
            eps = torch.randn(x.shape, device=DEVICE, generator=g)
            ab = d.abar[t][:, None]
            xt = ab.sqrt() * x + (1 - ab).sqrt() * eps
            tot += ((model(xt, t) - eps) ** 2).mean().item()
    return tot / n_rep
# fresh samples use seed 30000 + s; training set uses seed s (make_gmm(n, 8, r, 0.3, seed))

# %% experiments (tag -> call)
# main      : run_grid(range(5), tag="main")                                   # n=1000, 10000 steps
# mainN40   : cfg_mem = dict(CFG, n_train=40, iters=30000)
#             run_grid(range(5), seeds=SEEDS, cfg=cfg_mem, tag="mainN40")
# widthsweep: for s in [0,1,2]: for r in [1.5,3.0]: for w in [4,8,16,32,64,128,256,512]:
#                 run_custom_save(r, 3, w, s, cfg_mem, "widthsweep")            # depth 3
# depthsweep: depth 1-6, width 256, tag "depthsweep"; depthsweep512: depth 3/4/6, width 512
# ablN*     : for n in [20,100,300,1000]: cfg = dict(CFG, n_train=n, iters=30000)
#                 run_custom_save(r, 4, 256, s, cfg, f"ablN{n}")                # n=40 taken from mainN40
# abl_T*    : run_ablation("T", [50, 200], di=2, ci=2, seeds=(0, 1))            # T=100 taken from main
# iter60k   : cfg60 = dict(CFG, n_train=40, iters=60000); run_one(di, ci, s, cfg=cfg60, tag="iter60k")
