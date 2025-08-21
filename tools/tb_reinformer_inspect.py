# tools/tb_reinformer_inspect.py
# -*- coding: utf-8 -*-
"""
将 ReinFormer 的数据流可视化到 TensorBoard：
- Graph：模型前向图（若 add_graph 支持）
- Hist/Scalars：各 token 嵌入、每个 block 输出等中间张量
- Figure：一张“Token 网格→Flatten→Transformer→还原→三个 head→取最后一步动作→env.step”的流程图
- Text：维度/num_inputs 小抄

用法示例：
python tools/tb_reinformer_inspect.py \
  --ckpt /liujinxin/isaacgym/.../Reinformer_furniture/checkpoints/reinformer_final_step1000.pt \
  --logdir runs/reinformer_tb \
  --context_len 10

然后：
tensorboard --logdir runs/reinformer_tb --bind_all
"""

import argparse, os, io
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.tensorboard import SummaryWriter

# === 你项目里的模型 ===
from model.reinformer import ReinFormer

# ---------- 读 ckpt 里的维度 ----------
def read_dims_from_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    dims = ck.get("dims", {}) or {}
    variant = ck.get("variant", {}) or {}
    info = dict(
        state_dim = dims.get("state_dim"),
        act_dim   = dims.get("act_dim"),
        v1_dim    = dims.get("vision1_dim", 0) or 0,
        v2_dim    = dims.get("vision2_dim", 0) or 0,
        context_len = variant.get("context_len"),
    )
    return ck, info

# ---------- 画流程图（返回 matplotlib Figure） ----------
def draw_flow_figure(T, Ds, Da, Dv1, Dv2, h_dim, n_blocks):
    n_vision = int(Dv1 > 0) + int(Dv2 > 0)
    token_types = ["state"]
    if Dv1 > 0: token_types.append("vision1")
    if Dv2 > 0: token_types.append("vision2")
    token_types += ["rtg", "action"]
    num_inputs = 3 + n_vision

    fig = plt.figure(figsize=(14, 8), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.5], width_ratios=[1.7, 2.4], hspace=0.25, wspace=0.18)
    ax_grid = fig.add_subplot(gs[0,0]); ax_flow = fig.add_subplot(gs[0,1]); ax_slide = fig.add_subplot(gs[1,:])

    # token 网格
    ax_grid.axis("off")
    colors = {"state":"#6baed6","vision1":"#9ecae1","vision2":"#c6dbef","rtg":"#74c476","action":"#fd8d3c"}
    left, bottom, cw, ch = 0.8, 0.8, 1.0, 0.6
    for r, name in enumerate(token_types):
        for t in range(T):
            x = left + t*cw; y = bottom + (len(token_types)-1-r)*ch
            rect = Rectangle((x, y), cw*0.95, ch*0.9, ec="0.2", fc=colors.get(name,"lightgray"), lw=1.0)
            ax_grid.add_patch(rect)
            if r == len(token_types)-1:
                ax_grid.text(x+cw*0.5, y+ch+0.08, f"t={t}", ha="center", va="bottom", fontsize=9, color="0.35")
            if t == T-1:
                rect.set_linewidth(2.0); rect.set_edgecolor("black")
    for r, name in enumerate(token_types):
        y = bottom + (len(token_types)-1-r)*ch + ch*0.45
        ax_grid.text(left-0.2, y, name, ha="right", va="center", fontsize=10)
    ax_grid.set_xlim(0, left+T*cw+1.5); ax_grid.set_ylim(0, bottom+len(token_types)*ch+1.2)
    ax_grid.set_title(f"Token 网格：时间 T×种类 num_inputs={num_inputs}", fontsize=11)

    # 右侧流程（简化版）
    ax_flow.axis("off")
    x0 = 0.2; y = 0.75
    ax_flow.add_patch(Rectangle((x0, y-0.08), 1.8, 0.16, fc="#ddd", ec="k"))
    ax_flow.text(x0+0.9, y, f"Flatten: (B, T×{num_inputs}, h_dim)", ha="center", va="center", fontsize=10)
    ax_flow.annotate("", xy=(x0+2.2, y), xytext=(x0+2.0, y), arrowprops=dict(arrowstyle="->", lw=1.5))
    Tx, Tw, Th = x0+2.2, 1.8, 0.5
    for i in range(n_blocks):
        yy = y - Th/2 + (i - n_blocks/2 + 0.5)*(Th/n_blocks)
        ax_flow.add_patch(Rectangle((Tx, yy), Tw, Th/n_blocks, fc="#f2f2f2", ec="k", lw=0.8))
    ax_flow.text(Tx+Tw/2, y, f"Transformer × {n_blocks}\n输出同长: (B, T×{num_inputs}, h_dim)",
                 ha="center", va="center", fontsize=10)
    ax_flow.annotate("", xy=(Tx+Tw+0.4, y), xytext=(Tx+Tw, y), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax_flow.add_patch(Rectangle((Tx+Tw+0.4, y-0.08), 2.1, 0.16, fc="#ddd", ec="k"))
    ax_flow.text(Tx+Tw+1.45, y, "Reshape: (B, num_inputs, T, h_dim)\n按索引取各“轨道”", ha="center", va="center", fontsize=10)
    heads = [("predict_rtg","(B,T,1)","从 state 轨道取"),
             ("predict_action",f"(B,T,Da={Da})","从 rtg 轨道取"),
             ("predict_state",f"(B,T,Ds={Ds})","从 action 轨道取")]
    y_list = [0.48, 0.32, 0.16]; xh = Tx+Tw+0.6
    for (name, shp, note), yy in zip(heads, y_list):
        ax_flow.annotate("", xy=(xh-0.3, yy), xytext=(xh-0.6, y), arrowprops=dict(arrowstyle="->", lw=1.2))
        ax_flow.add_patch(Rectangle((xh-0.3, yy-0.06), 2.0, 0.12, fc="#fff", ec="k"))
        ax_flow.text(xh+0.7, yy, f"{name}: {shp}\n{note}", ha="center", va="center", fontsize=10)
    xlast = xh+1.9; yy = y_list[1]
    ax_flow.annotate("", xy=(xlast, yy), xytext=(xh+1.7, yy), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax_flow.add_patch(Rectangle((xlast, yy-0.06), 2.1, 0.12, fc="#ffe6cc", ec="k"))
    ax_flow.text(xlast+1.05, yy, "只取最后一步： action[:, -1, :] → (B, Da)", ha="center", va="center", fontsize=10)

    # 滑动窗口
    ax_slide.axis("off")
    ax_slide.set_title("滑动窗口（context_len = T）随时间右移", fontsize=11)
    def draw_bar(x0, y0, t0):
        for i in range(T):
            rect = Rectangle((x0+i*0.5, y0), 0.48, 0.28, fc="#cce5ff", ec="k")
            ax_slide.add_patch(rect)
            ax_slide.text(x0+i*0.5+0.24, y0+0.14, f"{t0+i}", ha="center", va="center", fontsize=9)
    draw_bar(2.0, 0.55, 0)
    draw_bar(2.0, 0.20, 1)
    ax_slide.annotate("", xy=(4.9, 0.47), xytext=(4.9, 0.34), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax_slide.text(5.05, 0.405, "新观测进入，窗口整体右移\n长度始终保持为 T", fontsize=10, va="center")

    fig.suptitle(f"ReinFormer × FurnitureSim — 数据流示意 | Ds={Ds}, Da={Da}, Dv1={Dv1}, Dv2={Dv2}, h={h_dim}, T={T}", y=0.98)
    return fig

# ---------- 钩子：抓取中间张量 ----------
def register_hooks(model, bucket: dict):
    names = []

    def save(name):
        names.append(name)
        def _hook(_m, _inp, out):
            try:
                bucket[name] = out.detach()
            except Exception:
                try:
                    bucket[name] = out[0].detach()
                except Exception:
                    pass
        return _hook

    # 基础嵌入
    model.embed_state.register_forward_hook(save("embed_state"))
    model.embed_rtg.register_forward_hook(save("embed_rtg"))
    model.embed_action.register_forward_hook(save("embed_action"))
    if model.embed_vision1 is not None:
        model.embed_vision1.register_forward_hook(save("embed_vision1"))
    if model.embed_vision2 is not None:
        model.embed_vision2.register_forward_hook(save("embed_vision2"))

    # LN 后的 tokens（flatten 后）
    model.embed_ln.register_forward_hook(save("embed_ln"))

    # 每个 block 输出
    if hasattr(model, "transformer"):
        for i, blk in enumerate(model.transformer):
            blk.register_forward_hook(save(f"block_{i:02d}_out"))

    return names

# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--logdir", type=str, default="runs/reinformer_tb")
    ap.add_argument("--context_len", type=int, default=None, help="不填则用 ckpt.variant.context_len")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    # 可选覆盖（一般不需要）
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout_p", type=float, default=0.1)
    ap.add_argument("--init_temperature", type=float, default=0.1)
    ap.add_argument("--target_entropy", type=float, default= -1.0)  # 仅占位
    ap.add_argument("--max_timestep", type=int, default=4096)
    ap.add_argument("--dt_mask", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # 读 ckpt 维度
    ck, info = read_dims_from_ckpt(args.ckpt)
    Ds = int(info["state_dim"]); Da = int(info["act_dim"])
    v1_dim = int(info["v1_dim"]); v2_dim = int(info["v2_dim"])
    T = int(args.context_len or info["context_len"] or 10)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 构建模型并加载权重
    model = ReinFormer(
        state_dim=Ds, act_dim=Da,
        n_blocks=args.n_blocks, h_dim=args.embed_dim,
        context_len=T, n_heads=args.n_heads, drop_p=args.dropout_p,
        init_temperature=args.init_temperature, target_entropy=args.target_entropy,
        max_timestep=args.max_timestep, dt_mask=args.dt_mask,
        vision1_dim=v1_dim, vision2_dim=v2_dim,
    ).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)

    # 准备一个 dummy batch（只为写 TensorBoard，不跑环境）
    B = args.batch_size
    timesteps = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).repeat(B,1)             # (B,T) int64
    states    = torch.randn(B, T, Ds, device=device) * 0.01                                           # (B,T,Ds)
    actions   = torch.zeros(B, T, Da, device=device)                                                  # (B,T,Da)
    rtg       = torch.ones(B, T, 1, device=device)                                                    # (B,T,1)
    v1 = torch.zeros(B, T, v1_dim, device=device) if v1_dim > 0 else None
    v2 = torch.zeros(B, T, v2_dim, device=device) if v2_dim > 0 else None

    # 注册钩子收集中间张量
    bucket = {}
    _ = register_hooks(model, bucket)

    # 1) Graph（可能会失败，故 try）
    try:
        with torch.no_grad():
            writer.add_graph(model,
                             (timesteps, states, actions, rtg, v1, v2),
                             verbose=False)
        print("[TB] Graph 已写入（Graph 标签）")
    except Exception as e:
        print(f"[TB] add_graph 失败（可忽略）：{e}")

    # 2) 跑一遍前向，拿到 head 输出，同时钩子会抓到中间层
    with torch.no_grad():
        rtg_pred, action_head, state_pred = model(
            timesteps=timesteps, states=states, actions=actions,
            returns_to_go=rtg, vision1=v1, vision2=v2
        )

    # 3) 将中间张量写入 TB（Hist + Scalars + Text）
    step = 0
    def log_tensor(tag, x):
        if not torch.is_tensor(x): return
        x_cpu = x.detach().float().flatten().cpu()
        if x_cpu.numel() == 0: return
        writer.add_histogram(f"hists/{tag}", x_cpu, step)
        writer.add_scalar(f"stats/{tag}_mean", x_cpu.mean().item(), step)
        writer.add_scalar(f"stats/{tag}_std",  x_cpu.std(unbiased=False).item(), step)
        writer.add_scalar(f"stats/{tag}_absmax", float(x_cpu.abs().max().item()), step)
        writer.add_text(f"shapes/{tag}", str(list(x.shape)), step)

    for name, ten in bucket.items():
        log_tensor(name, ten)

    # head 输出也记录下
    log_tensor("head/rtg_pred", rtg_pred)       # (B,T,1)
    # action_head 可能是分布或带属性的对象，尽量拿“均值”类的东西
    try:
        if hasattr(action_head, "mean"):
            log_tensor("head/action_mean", action_head.mean)
        elif hasattr(action_head, "loc"):
            log_tensor("head/action_loc", action_head.loc)
        elif hasattr(action_head, "mu"):
            log_tensor("head/action_mu", action_head.mu)
        elif hasattr(action_head, "logits"):
            log_tensor("head/action_logits", action_head.logits)
        else:
            # 退而求其次，尝试调用 .rsample()
            sample = action_head.rsample()
            log_tensor("head/action_sample", sample)
    except Exception:
        pass
    log_tensor("head/state_pred", state_pred)   # (B,T,Ds)

    # 4) 只取最后一步动作 (B,Da) 的演示
    try:
        # 这里演示“如果是正态分布头”，取其均值；不是的话按上面兜底
        if hasattr(action_head, "mean"):
            act_last = action_head.mean[:, -1, :]         # (B,Da)
        elif hasattr(action_head, "loc"):
            act_last = action_head.loc[:, -1, :]
        else:
            # 直接取 state_pred 的最后一步当 demo（仅演示维度，不用于环境）
            act_last = state_pred[:, -1, :Da]
        writer.add_histogram("rollout/last_action", act_last.flatten().cpu(), step)
        writer.add_text("rollout/last_action_shape", str(list(act_last.shape)), step)
    except Exception:
        pass

    # 5) 写一张“数据流示意图”到 TB 的 Images（方便你从 TB 直接看图）
    fig = draw_flow_figure(T=T, Ds=Ds, Da=Da, Dv1=v1_dim, Dv2=v2_dim, h_dim=args.embed_dim, n_blocks=args.n_blocks)
    writer.add_figure("flow/diagram", fig, global_step=step)
    plt.close(fig)

    # 6) 维度/配置小抄
    num_inputs = 3 + int(v1_dim>0) + int(v2_dim>0)
    cheat = (
        f"Ds={Ds}, Da={Da}, v1_dim={v1_dim}, v2_dim={v2_dim}, h_dim={args.embed_dim}, "
        f"T={T}, num_inputs={num_inputs}, n_blocks={args.n_blocks}, n_heads={args.n_heads}\n"
        f"动作含义（Da=8）：[dx, dy, dz, qx, qy, qz, qw, gripper]"
    )
    writer.add_text("cheatsheet/dims", cheat, step)

    writer.flush()
    writer.close()
    print(f"[TB] 完成。用 TensorBoard 打开：tensorboard --logdir {args.logdir} --bind_all")
    print("    标签里看：Graph / Histograms / Scalars / Images(flow/diagram) / Text(cheatsheet)")

if __name__ == "__main__":
    main()
