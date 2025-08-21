# sweep_asha.py
import argparse, os, time, json, numpy as np, torch, optuna
from torch.utils.data import DataLoader, random_split

from trainer import ReinFormerTrainer
from data.dataset import ReinformerFurnitureDataset
from eval.eval_offline import evaluate_offline

# --------- 工具 ----------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def infer_dims_from_sample(ds, vision_mode: str, T: int):
    # 从前几条样本里嗅探非零视觉维度，避免第一条碰巧是 (T,0)
    v1_dim = v2_dim = 0
    s0 = ds[0]
    Ds = int(s0[1].shape[-1]); Da = int(s0[2].shape[-1])
    if vision_mode == "cache":
        sniff_n = min(256, len(ds))
        for i in range(sniff_n):
            v1 = ds[i][-2]; v2 = ds[i][-1]
            if v1 is not None and v1.numel()>0 and v1.shape[-1]>0: v1_dim = int(v1.shape[-1])
            if v2 is not None and v2.numel()>0 and v2.shape[-1]>0: v2_dim = int(v2.shape[-1])
            if v1_dim or v2_dim: break
    return Ds, Da, v1_dim, v2_dim

def build_loaders(args):
    ds = ReinformerFurnitureDataset(
        root_dir=args.data_path,
        context_len=args.context_len,
        vision_mode=args.vision_mode,
        feature_root=(args.vision_cache_dir or None),
        gamma=args.gamma,
        sample_stride=args.sample_stride,
        max_episodes=args.limit_episodes,
        episode_range="",
        task_filter=args.tasks,
        strict_feature_len=False,
        verbose_mismatch=True,
    )
    if args.val_ratio <= 0.0:
        raise ValueError("ASHA 需要 val 集进行离线评估，请设置 --val_ratio>0")
    n_total = len(ds); n_val = max(1, int(args.val_ratio * n_total)); n_train = max(1, n_total - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(1, args.num_workers // 2), pin_memory=True, drop_last=False)
    return ds, train_loader, val_loader

# --------- 采样空间 ----------
def sample_hparams(trial, fixed):
    hp = {}
    # 固定项直接抄
    hp.update(fixed)
    # 可调项
    hp["tau"] = trial.suggest_float("tau", 0.7, 0.95)
    hp["lr"]  = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    hp["wd"]  = trial.suggest_float("wd", 1e-4, 2e-1, log=True)
    hp["dropout_p"] = trial.suggest_categorical("dropout_p", [0.0, 0.05, 0.1, 0.2])
    hp["grad_norm"] = trial.suggest_categorical("grad_norm", [0.5, 1.0, 2.0])
    hp["init_temperature"] = trial.suggest_float("init_temperature", 0.02, 0.5, log=True)
    hp["dt_mask"] = trial.suggest_categorical("dt_mask", [False, True])
    # 视觉开关（若只想腕部，改为 ["v1"]）
    hp["vision_choice"] = trial.suggest_categorical("vision_choice", ["both", "v1"])  # 不选 v2/none
    return hp

# --------- 目标函数（ASHA） ----------
def make_objective(args):
    device = torch.device(args.device)
    ds, train_loader, val_loader = build_loaders(args)
    Ds, Da, v1_dim, v2_dim = infer_dims_from_sample(ds, args.vision_mode, args.context_len)
    print(f"[dims] Ds={Ds}, Da={Da}, v1={v1_dim}, v2={v2_dim}")

    # 固定的模型大小（先不扫）
    fixed = dict(
        n_blocks=args.n_blocks, embed_dim=args.embed_dim, context_len=args.context_len,
        n_heads=args.n_heads, max_timestep=args.max_timestep,
    )

    def objective(trial: optuna.trial.Trial):
        # 设种子（不同 trial 随机不同）
        seed = args.seed + trial.number
        set_seed(seed)

        # 超参
        hp = sample_hparams(trial, fixed)
        use_v1 = (v1_dim > 0) and (hp["vision_choice"] in ["both","v1"])
        use_v2 = (v2_dim > 0) and (hp["vision_choice"] in ["both"])  # 我们不单独开 v2

        variant = dict(
            n_blocks=hp["n_blocks"], embed_dim=hp["embed_dim"], context_len=hp["context_len"],
            n_heads=hp["n_heads"], dropout_p=hp["dropout_p"], grad_norm=hp["grad_norm"],
            tau=hp["tau"], lr=hp["lr"], wd=hp["wd"], warmup_steps=args.warmup_steps,
            init_temperature=hp["init_temperature"], max_timestep=hp["max_timestep"],
            dt_mask=hp["dt_mask"], vision1_dim=(v1_dim if use_v1 else 0), vision2_dim=(v2_dim if use_v2 else 0),
        )

        trainer = ReinFormerTrainer(state_dim=Ds, act_dim=Da, device=device, variant=variant)

        # rung 预算（步数）
        rung_steps = args.rung_steps
        assert len(rung_steps) >= 1
        last_step = 0

        best_score = -1e9
        trial_dir = os.path.join(args.ckpt_dir, f"trial_{trial.number:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        train_iter = iter(train_loader)
        for rung_idx, target_step in enumerate(rung_steps):
            while last_step < target_step:
                try:
                    (ts, s, a, rtg, r, msk, v1, v2) = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    (ts, s, a, rtg, r, msk, v1, v2) = next(train_iter)

                log = trainer.train_step(
                    timesteps=ts, states=s, actions=a,
                    returns_to_go=rtg, rewards=r, traj_mask=msk,
                    vision1=(v1 if (use_v1 and v1_dim>0) else None),
                    vision2=(v2 if (use_v2 and v2_dim>0) else None),
                )
                last_step += 1

                # 定期离线评估 + 上报给 ASHA
                if (last_step % args.eval_interval == 0) or (last_step == target_step):
                    metrics = evaluate_offline(trainer.model, device, val_loader, tau=hp["tau"])
                    score = metrics["score"]
                    trial.report(score, last_step)
                    # 保存 trial 当前最好 ckpt
                    if score > best_score:
                        best_score = score
                        ckpt = {
                            "step": last_step, "model": trainer.model.state_dict(),
                            "opt": trainer.optimizer.state_dict(),
                            "state_mean": ds._state_mean.numpy(), "state_std": ds._state_std.numpy(),
                            "variant": variant,
                            "dims": {"state_dim": Ds, "act_dim": Da, "vision1_dim": (v1_dim if use_v1 else 0), "vision2_dim": (v2_dim if use_v2 else 0)},
                            "val_metrics": metrics,
                            "seed": seed,
                            "hyperparams": hp,
                        }
                        torch.save(ckpt, os.path.join(trial_dir, "best.pt"))
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        # 返回该 trial 的最好分数
        return float(best_score)

    return objective

# --------- 主入口 ----------
def main():
    pa = argparse.ArgumentParser()
    # 数据
    pa.add_argument("--data_path", type=str, required=True)
    pa.add_argument("--vision_mode", type=str, default="cache", choices=["cache","none","raw"])
    pa.add_argument("--vision_cache_dir", type=str, default="")
    pa.add_argument("--tasks", type=str, nargs="*", default=None)
    pa.add_argument("--context_len", type=int, default=10)
    pa.add_argument("--sample_stride", type=int, default=1)
    pa.add_argument("--gamma", type=float, default=1.0)
    pa.add_argument("--limit_episodes", type=int, default=0)
    pa.add_argument("--val_ratio", type=float, default=0.1)
    # 训练
    pa.add_argument("--device", type=str, default="cuda")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--batch_size", type=int, default=32)
    pa.add_argument("--num_workers", type=int, default=8)
    pa.add_argument("--warmup_steps", type=int, default=500)
    # 模型大小（先固定）
    pa.add_argument("--n_blocks", type=int, default=6)
    pa.add_argument("--embed_dim", type=int, default=512)
    pa.add_argument("--n_heads", type=int, default=8)
    pa.add_argument("--max_timestep", type=int, default=4096)
    # ASHA
    pa.add_argument("--rung_steps", type=int, nargs="+", default=[1000, 4000, 10000])
    pa.add_argument("--eval_interval", type=int, default=200)
    pa.add_argument("--n_trials", type=int, default=40)
    pa.add_argument("--ckpt_dir", type=str, default="checkpoints_sweep")
    # 可选：结束后做在线评测
    pa.add_argument("--env_id", type=str, default="")
    pa.add_argument("--sim_topk", type=int, default=2)
    pa.add_argument("--sim_episodes", type=int, default=20)
    pa.add_argument("--use_vision", type=str, default="both", choices=["both","v1","v2","none"])
    args = pa.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=args.eval_interval,   # 至少跑到第一次评估
        reduction_factor=3,                # η=3
        min_early_stopping_rate=0
    )
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    objective = make_objective(args)

    print("[ASHA] start optimize...")
    study.optimize(objective, n_trials=args.n_trials)

    print("[ASHA] best_value =", study.best_value)
    print("[ASHA] best_params =", study.best_params)

    # 保存 study 结果
    with open(os.path.join(args.ckpt_dir, "study_best.json"), "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)

    # ====== 可选：对 top-K 进行在线评测 ======
    if args.env_id:
        # 找到各 trial 的 best.pt，按 ckpt 里 val score 排序
        import glob
        cks = glob.glob(os.path.join(args.ckpt_dir, "trial_*", "best.pt"))
        scored = []
        for p in cks:
            try:
                ck = torch.load(p, map_location="cpu")
                scored.append((ck.get("val_metrics", {}).get("score", -1e9), p))
            except Exception:
                pass
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[:max(1, args.sim_topk)]
        print("[SIM] top candidates:", topk)

        # 调用 eval_sim 逐个跑仿真
        from eval_sim import rollout
        for rank, (sc, path) in enumerate(topk, 1):
            print(f"\n[SIM] Candidate #{rank} score={sc:.4f} path={path}")
            rollout(args.env_id, path, episodes=args.sim_episodes, device=args.device, use_vision=args.use_vision)

if __name__ == "__main__":
    main()
