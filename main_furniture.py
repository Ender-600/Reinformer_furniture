# main_furniture.py
# -*- coding: utf-8 -*-

import argparse, os, random, time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from trainer import ReinFormerTrainer
from data.dataset import ReinformerFurnitureDataset

# 离线评估（我们就用 eval_offline.py 里的函数）
try:
    from eval_offline import evaluate_offline
    HAS_OFFLINE_EVAL = True
except Exception:
    HAS_OFFLINE_EVAL = False

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def infer_dims_from_batch(batch, vision_mode: str):
    _, s, a, _, _, _, v1, v2 = batch
    Ds = int(s.shape[-1]); Da = int(a.shape[-1])
    if vision_mode == "cache":
        v1_dim = 0 if (v1.numel() == 0 or v1.shape[-1] == 0) else int(v1.shape[-1])
        v2_dim = 0 if (v2.numel() == 0 or v2.shape[-1] == 0) else int(v2.shape[-1])
    else:
        v1_dim = 0; v2_dim = 0
    return Ds, Da, v1_dim, v2_dim

def build_dataloaders(args):
    ds = ReinformerFurnitureDataset(
        root_dir=args.data_path,
        context_len=args.context_len,
        vision_mode=args.vision_mode,
        feature_root=(args.vision_cache_dir or None),
        gamma=args.gamma,
        sample_stride=args.sample_stride,
        max_episodes=args.limit_episodes,
        episode_range="",
        task_filter=args.tasks,   # <--- 现在使用 CLI 的 --tasks
        strict_feature_len=False,
        verbose_mismatch=True,
    )
    if args.val_ratio <= 0.0:
        train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = None
    else:
        n_total = len(ds); n_val = max(1, int(args.val_ratio * n_total)); n_train = max(1, n_total - n_val)
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=max(1, args.num_workers // 2), pin_memory=True, drop_last=False)
    return ds, train_loader, val_loader

def main():
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vision_mode", type=str, default="cache", choices=["cache", "none", "raw"])
    parser.add_argument("--vision_cache_dir", type=str, default="")
    parser.add_argument("--context_len", type=int, default=10)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tasks", type=str, nargs="*", default=None)

    # 训练
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    # 评估（离线）
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    # 模型/优化器
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--max_timestep", type=int, default=4096)
    parser.add_argument("--dt_mask", action="store_true")
    parser.add_argument("--limit_episodes", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Data
    ds, train_loader, val_loader = build_dataloaders(args)
    first_batch = next(iter(train_loader))
    Ds, Da, v1_dim, v2_dim = infer_dims_from_batch(first_batch, args.vision_mode)
    print(f"[dims] state_dim={Ds}, act_dim={Da}, vision1_dim={v1_dim}, vision2_dim={v2_dim}")
    print(f"[data] train iters/epoch ~= {len(train_loader)}; val_loader = {val_loader is not None}")

    variant = dict(
        n_blocks=args.n_blocks, embed_dim=args.embed_dim, context_len=args.context_len,
        n_heads=args.n_heads, dropout_p=args.dropout_p, grad_norm=args.grad_norm,
        tau=args.tau, lr=args.lr, wd=args.wd, warmup_steps=args.warmup_steps,
        init_temperature=args.init_temperature, max_timestep=args.max_timestep,
        dt_mask=args.dt_mask, vision1_dim=v1_dim, vision2_dim=v2_dim,
    )

    trainer = ReinFormerTrainer(state_dim=Ds, act_dim=Da, device=device, variant=variant)

    # train loop
    os.makedirs(args.ckpt_dir, exist_ok=True)
    data_iter = iter(train_loader)
    t0 = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60); print(f"Start training @ {start_time}"); print("=" * 60)

    best_score = -1e9  # 用于离线评估选最优
    for step in range(1, args.max_train_steps + 1):
        try:
            (timesteps, states, actions, returns_to_go, rewards, traj_mask, v1, v2) = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            (timesteps, states, actions, returns_to_go, rewards, traj_mask, v1, v2) = next(data_iter)

        log = trainer.train_step(
            timesteps=timesteps, states=states, actions=actions,
            returns_to_go=returns_to_go, rewards=rewards, traj_mask=traj_mask,
            vision1=(v1 if v1_dim > 0 else None), vision2=(v2 if v2_dim > 0 else None),
        )

        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(f"[{step}/{args.max_train_steps}] "
                  f"loss={log['loss']:.4f} rtg={log['rtg_loss']:.4f} act={log['action_loss']:.4f} "
                  f"H={log['entropy']:.3f} T={log['temperature']:.3f} ({dt:.1f}s)")
            t0 = time.time()

        # offline eval + 保存 best
        if args.eval_interval > 0 and (step % args.eval_interval == 0) and (val_loader is not None):
            if not HAS_OFFLINE_EVAL:
                print("[warn] 未检测到 eval_offline.evaluate_offline，跳过。")
            else:
                metrics = evaluate_offline(trainer.model, device, val_loader, tau=args.tau)
                msg = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                print(f"[EVAL @ step {step}] {msg}")

                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    ckpt = {
                        "step": step, "model": trainer.model.state_dict(),
                        "opt": trainer.optimizer.state_dict(),
                        # 保存统计与变体/维度
                        "state_mean": ds._state_mean.numpy(),
                        "state_std": ds._state_std.numpy(),
                        "variant": variant,
                        "dims": {"state_dim": Ds, "act_dim": Da, "vision1_dim": v1_dim, "vision2_dim": v2_dim},
                        "val_metrics": metrics,
                    }
                    best_path = os.path.join(args.ckpt_dir, f"reinformer_best.pt")
                    torch.save(ckpt, best_path)
                    print(f"[ckpt] best updated -> {best_path}")

        # save ckpt (periodic)
        if args.save_interval > 0 and (step % args.save_interval == 0):
            ckpt = {
                "step": step, "model": trainer.model.state_dict(), "opt": trainer.optimizer.state_dict(),
                "state_mean": ds._state_mean.numpy(), "state_std": ds._state_std.numpy(),
                "variant": variant,
                "dims": {"state_dim": Ds, "act_dim": Da, "vision1_dim": v1_dim, "vision2_dim": v2_dim},
            }
            path = os.path.join(args.ckpt_dir, f"reinformer_step{step}.pt")
            torch.save(ckpt, path)
            print(f"[ckpt] saved -> {path}")

    # final ckpt
    final_ckpt = {
        "step": args.max_train_steps,
        "model": trainer.model.state_dict(),
        "opt": trainer.optimizer.state_dict(),
        "state_mean": ds._state_mean.numpy(),
        "state_std": ds._state_std.numpy(),
        "variant": variant,
        "dims": {"state_dim": Ds, "act_dim": Da, "vision1_dim": v1_dim, "vision2_dim": v2_dim},
    }
    final_path = os.path.join(args.ckpt_dir, f"reinformer_final_step{args.max_train_steps}.pt")
    torch.save(final_ckpt, final_path)
    print(f"[ckpt] saved -> {final_path}")
    print("Training finished.")

if __name__ == "__main__":
    main()
