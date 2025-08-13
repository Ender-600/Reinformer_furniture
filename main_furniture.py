# main_furniture.py
# -*- coding: utf-8 -*-

import argparse, os, random, time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from trainer import ReinFormerTrainer
from data.dataset import ReinformerFurnitureDataset  # 按你的实际路径
# 离线评估：请将我们之前给你的 evaluate_offline(...) 放到 eval.py 里
try:
    from eval import evaluate_offline
    HAS_OFFLINE_EVAL = True
except Exception:
    HAS_OFFLINE_EVAL = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_dims_from_batch(batch, vision_mode: str):
    """
    batch: (timesteps, states, actions, returns_to_go, rewards, mask, v1, v2),
           每个形状为 (B,T,...) 或 (B,T)
    """
    _, s, a, _, _, _, v1, v2 = batch
    Ds = int(s.shape[-1])
    Da = int(a.shape[-1])
    if vision_mode == "cache":
        v1_dim = 0 if (v1.numel() == 0 or v1.shape[-1] == 0) else int(v1.shape[-1])
        v2_dim = 0 if (v2.numel() == 0 or v2.shape[-1] == 0) else int(v2.shape[-1])
    else:
        # raw/none：本实现不在模型里直接用 raw 图像；raw 时请先在 dataset 端做 CNN 编码为向量
        v1_dim = 0
        v2_dim = 0
    return Ds, Da, v1_dim, v2_dim


def build_dataloaders(args):
    """
    构建完整数据集，并切分 train/val（可选）。
    支持：
      - data_path:  lerobot 根目录 / 单个 parquet / pkl 文件 / pkl 目录
      - reward_fallback_root: 缺失 reward 时从旧库回填（与 episode 对齐）
      - vision_cache_dir: R3M/VIP 预特征根目录（vision_mode=cache 时需要）
    """
    ds = ReinformerFurnitureDataset(
        data_path=args.data_path,
        context_len=args.context_len,
        vision_mode=args.vision_mode,
        vision_cache_dir=(args.vision_cache_dir or None),
        sample_stride=args.sample_stride,
        gamma=args.gamma,
        reward_agg=args.reward_agg,
        reward_index=args.reward_index,
        reward_weights=(None if not args.reward_weights else [float(x) for x in args.reward_weights.split(",")]),
        # 若你的 dataset 已实现 reward_fallback_root 参数，则放开：
        reward_fallback_root=(args.reward_fallback_root or None),
    )

    if args.val_ratio <= 0.0:
        train_loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = None
    else:
        n_total = len(ds)
        n_val = max(1, int(args.val_ratio * n_total))
        n_train = max(1, n_total - n_val)
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )
    return ds, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()

    # ========== 数据相关 ==========
    parser.add_argument("--data_path", type=str, required=True,
                        help="lerobot 根目录 / 单个 parquet / pkl 文件 / pkl 目录")
    parser.add_argument("--reward_fallback_root", type=str, default="",
                        help="缺失 reward 的回填根目录（与 episode 对齐）")
    parser.add_argument("--vision_mode", type=str, default="cache", choices=["cache", "none", "raw"])
    parser.add_argument("--vision_cache_dir", type=str, default="",
                        help="R3M/VIP 预特征根目录（vision_mode=cache 时需要）")
    parser.add_argument("--context_len", type=int, default=10)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--reward_agg", type=str, default="sum", choices=["sum", "index", "weighted"])
    parser.add_argument("--reward_index", type=int, default=0)
    parser.add_argument("--reward_weights", type=str, default="", help="加权模式下的权重, 逗号分隔，如 '1,0,0.5'")

    # ========== 训练相关 ==========
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=0, help=">0 则周期性保存 ckpt")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    # ========== 评估相关（离线） ==========
    parser.add_argument("--eval_interval", type=int, default=0, help=">0 则每 N 步做一次离线评估")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集占比（0~1），eval_interval>0 时生效")

    # ========== 模型/优化器超参 ==========
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.8)   # expectile
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--max_timestep", type=int, default=4096)
    parser.add_argument("--dt_mask", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ========== Data ==========
    ds, train_loader, val_loader = build_dataloaders(args)
    first_batch = next(iter(train_loader))  # 触发一次读取，推断维度
    Ds, Da, v1_dim, v2_dim = infer_dims_from_batch(first_batch, args.vision_mode)
    print(f"[dims] state_dim={Ds}, act_dim={Da}, vision1_dim={v1_dim}, vision2_dim={v2_dim}")
    print(f"[data] train iters/epoch ~= {len(train_loader)}; val_loader = {val_loader is not None}")

    # ========== 组装 variant ==========
    variant = dict(
        n_blocks=args.n_blocks,
        embed_dim=args.embed_dim,
        context_len=args.context_len,
        n_heads=args.n_heads,
        dropout_p=args.dropout_p,
        grad_norm=args.grad_norm,
        tau=args.tau,
        lr=args.lr,
        wd=args.wd,
        warmup_steps=args.warmup_steps,
        init_temperature=args.init_temperature,
        max_timestep=args.max_timestep,
        dt_mask=args.dt_mask,
        vision1_dim=v1_dim,
        vision2_dim=v2_dim,
    )

    # ========== Trainer ==========
    trainer = ReinFormerTrainer(
        state_dim=Ds,
        act_dim=Da,
        device=device,
        variant=variant
    )

    # ========== 训练循环 ==========
    os.makedirs(args.ckpt_dir, exist_ok=True)
    data_iter = iter(train_loader)
    t0 = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print(f"Start training @ {start_time}")
    print("=" * 60)

    for step in range(1, args.max_train_steps + 1):
        try:
            (timesteps, states, actions, returns_to_go, rewards, traj_mask, v1, v2) = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            (timesteps, states, actions, returns_to_go, rewards, traj_mask, v1, v2) = next(data_iter)

        loss = trainer.train_step(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            rewards=rewards,
            traj_mask=traj_mask,
            vision1=(v1 if v1_dim > 0 else None),
            vision2=(v2 if v2_dim > 0 else None),
        )

        # log
        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(f"[{step}/{args.max_train_steps}] loss={loss:.4f}  ({dt:.1f}s)")
            t0 = time.time()

        # offline eval
        if args.eval_interval > 0 and (step % args.eval_interval == 0) and val_loader is not None:
            if not HAS_OFFLINE_EVAL:
                print("[warn] 未检测到 eval.evaluate_offline，已跳过离线评估。")
            else:
                metrics = evaluate_offline(
                    model=trainer.model,
                    device=device,
                    val_loader=val_loader,
                    tau=args.tau,
                )
                msg = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                print(f"[EVAL @ step {step}] {msg}")

        # save ckpt
        if args.save_interval > 0 and (step % args.save_interval == 0):
            ckpt = {
                "step": step,
                "model": trainer.model.state_dict(),
                "opt": trainer.optimizer.state_dict(),
            }
            path = os.path.join(args.ckpt_dir, f"reinformer_step{step}.pt")
            torch.save(ckpt, path)
            print(f"[ckpt] saved -> {path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
