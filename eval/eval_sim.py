
import argparse, os, time
import numpy as np

import gym
import furniture_bench  
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv 

import torch
import torch.nn.functional as F

from trainer import ReinFormerTrainer
from data.dataset import ReinformerFurnitureDataset


# --------------------- 视觉编码（R3M/VIP） ---------------------
def build_encoder(vision: str, device: torch.device):
    if vision is None or vision.lower() == "none":
        return None, 0
    v = vision.lower()
    if v == "r3m":
        from r3m import load_r3m
        net = load_r3m("resnet50").to(device).eval()
        dim = 2048
    elif v == "vip":
        from vip import load_vip
        net = load_vip().to(device).eval()
        dim = 1024
    else:
        raise ValueError("--vision must be one of: none | r3m | vip")
    for p in net.parameters():
        p.requires_grad_(False)
    return net, dim


def _img_to_224_chw_float(img_hw3: torch.Tensor) -> torch.Tensor:
    """
    IsaacGym 相机图像默认是 (H,W,3) uint8 on CUDA。转为 (3,224,224) float32 in [0,1]。
    """
    if img_hw3.dtype != torch.float32:
        img = img_hw3.to(torch.float32) / 255.0
    else:
        img = img_hw3
    img = img.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
    return img.squeeze(0)  # 3,224,224


@torch.no_grad()
def encode_vision(obs: dict, encoder, device: torch.device):
    """
    从 obs 里取 color_image1/2，在线编码为 (Dv,) numpy。若 encoder=None，返回 None。
    """
    if encoder is None:
        return None, None

    # env 返回 shape: (num_envs, H, W, 3)，num_envs=1
    img1 = obs["color_image1"][0]  # H,W,3 (uint8/float)
    img2 = obs["color_image2"][0]

    x1 = _img_to_224_chw_float(img1).unsqueeze(0).to(device)  # 1,3,224,224
    x2 = _img_to_224_chw_float(img2).unsqueeze(0).to(device)

    v1 = encoder(x1).squeeze(0).detach().float().cpu().numpy()
    v2 = encoder(x2).squeeze(0).detach().float().cpu().numpy()
    return v1, v2


# --------------------- Policy 封装 ---------------------
class ReinFormerPolicy:
    """
    用 ReinFormerTrainer 的 model，暴露统一 act(...) 接口。
    修复点：只取最后一个时间步的动作 (B,Da)，避免把 (B,T,Da) 传给 env。
    """
    def __init__(self, state_dim, act_dim, device, variant, ckpt_path):
        self.device = device
        self.trainer = ReinFormerTrainer(state_dim=state_dim, act_dim=act_dim, device=device, variant=variant)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.trainer.model.load_state_dict(ckpt["model"], strict=True)
        self.trainer.model.eval()

    @torch.no_grad()
    def _to_last_step(self, x: torch.Tensor) -> torch.Tensor:
        # 张量形状若为 (B,T,...) 则取最后一步；若已是 (B,...) 则原样返回
        if x.dim() >= 3:
            return x[:, -1, ...]
        return x

    @torch.no_grad()
    def _dist_to_last_action(self, dist_or_tensor, temperature: float) -> torch.Tensor:
        """
        兼容不同 actor 实现，并且**统一抽取最后一步**：
        - 分布对象：取 mean（temperature≈0）或 sample/rsample -> (B,T,Da) -> (B,Da)
        - (mu, log_std) 元组：同上
        - 直接张量：可能是 (B,T,Da) 或 (B,Da) -> 取最后一步
        """
        d = dist_or_tensor

        # 直接张量
        if torch.is_tensor(d):
            a = self._to_last_step(d)

        # torch.distributions 风格
        elif hasattr(d, "mean") or hasattr(d, "sample") or hasattr(d, "rsample"):
            if temperature <= 1e-6 and hasattr(d, "mean"):
                a_full = d.mean
            else:
                if hasattr(d, "rsample"):
                    a_full = d.rsample()
                else:
                    a_full = d.sample()
            a = self._to_last_step(a_full)

        # 可能是 (mu, log_std) 或 (mu,) 之类
        elif isinstance(d, (list, tuple)) and len(d) >= 1 and torch.is_tensor(d[0]):
            mu = d[0]
            mu = self._to_last_step(mu)
            if temperature <= 1e-6 or len(d) < 2 or (not torch.is_tensor(d[1])):
                a = mu
            else:
                log_std = self._to_last_step(d[1])
                a = mu + torch.randn_like(mu) * (log_std.exp() * float(temperature))

        else:
            raise RuntimeError(f"Unrecognized action head type: {type(d)}")

        # 最终动作范围
        return torch.clamp(a, -1.0, 1.0)

    @torch.no_grad()
    def act(self, *, timesteps, states, actions, returns_to_go, vision1=None, vision2=None, temperature=0.0):
        """
        期望输入：
          - timesteps: (B,T) long
          - states:    (B,T,Ds)
          - actions:   (B,T,Da)
          - returns_to_go: (B,T,1)
          - vision1/2: (B,T,Dv) 或 None
        返回：
          - (B, Da) in [-1,1] —— 只取最后一个时间步
        """
        m = self.trainer.model
        rtg_preds, action_head, _ = m(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            vision1=vision1,
            vision2=vision2,
        )
        a = self._dist_to_last_action(action_head, temperature=float(temperature))
        return a


# --------------------- 评估 roll-out ---------------------
@torch.no_grad()
def rollout_episode(
    env: FurnitureSimEnv,
    policy: ReinFormerPolicy,
    device: torch.device,
    *,
    context_len: int = 10,
    target_return: float = 1.0,
    max_steps: int = 3000,
    state_mean: np.ndarray = None,
    state_std: np.ndarray = None,
    vision_encoder=None,
    v1_dim: int = 0,
    v2_dim: int = 0,
    temperature: float = 0.0,
):
    """
    在仿真里跑 1 条轨迹；按训练的 token 布局对齐：state,(v1,v2),rtg,action。
    - 滑动窗口：仅取最近 context_len 个时间步
    - 左侧 padding：历史不足时在左边补零
    - timesteps: long + clamp，避免 Embedding 报错
    """
    def obs_to_state(o: dict) -> np.ndarray:
        # 训练时的 35D: joint_positions(7)+ee_pos(3)+ee_quat(4)+parts_poses(num_parts*7)
        rs = o["robot_state"]
        jp = rs["joint_positions"][0].detach().cpu().numpy()  # (7,)
        ee_pos = rs["ee_pos"][0].detach().cpu().numpy()        # (3,)
        ee_quat = rs["ee_quat"][0].detach().cpu().numpy()      # (4,)
        pp = o["parts_poses"][0].detach().cpu().numpy()        # (num_parts*7,)
        s = np.concatenate([jp, ee_pos, ee_quat, pp], axis=0).astype(np.float32)
        return s

    def norm_state(x: np.ndarray) -> np.ndarray:
        if state_mean is None or state_std is None:
            return x
        return (x - state_mean) / (state_std + 1e-6)

    def _stack_and_pad(hist, T, device, dtype=torch.float32):
        """把 list[step, ...] 堆成 (1,T,...)，不足左侧 0-pad，多了取后 T 个"""
        if len(hist) == 0:
            arr = torch.zeros(0, device=device, dtype=dtype).view(0, -1)
        else:
            arr = torch.tensor(np.stack(hist, axis=0), device=device, dtype=dtype)
        if arr.dim() == 1:
            arr = arr.unsqueeze(-1)
        # 左 pad
        if arr.shape[0] < T:
            pad = torch.zeros((T - arr.shape[0], *arr.shape[1:]), device=device, dtype=dtype)
            arr = torch.cat([pad, arr], dim=0)
        else:
            arr = arr[-T:]
        return arr.unsqueeze(0)  # (1,T, ...)

    obs = env.reset()
    ep_ret, ep_len = 0.0, 0

    # 初始观测 -> state & vision
    s0 = obs_to_state(obs)
    v1_0 = v2_0 = None
    if vision_encoder is not None:
        v1_0, v2_0 = encode_vision(obs, vision_encoder, device)

    # 历史序列
    states_hist = [s0]                           # t=0 对应的 state
    actions_hist = []                            # a_{t-1}，t=0 时为空，稍后用 0 向量占位
    rtgs_hist = [float(target_return)]
    vision1_hist = [v1_0] if v1_0 is not None else []
    vision2_hist = [v2_0] if v2_0 is not None else []

    # 如果模型需要视觉，但没传 encoder，则用全 0 特征占位
    if vision_encoder is None and (v1_dim > 0 or v2_dim > 0):
        print("[warn] ckpt 需要视觉特征，但 --vision=none；采用全零向量占位（性能会下降）。")

    # 便捷：动作维度
    act_dim = policy.trainer.model.act_dim

    # 获取 embed_timestep 的 num_embeddings，供 clamp 上界
    max_ts = int(getattr(policy.trainer.model.embed_timestep, "num_embeddings", 4096))

    for t in range(max_steps):
        # 1) 构建 timesteps (long) + clamp
        t_lo = max(0, t - (context_len - 1))
        TS = torch.arange(t_lo, t + 1, device=device, dtype=torch.long)[None, :]  # (1, T')
        if TS.shape[1] < context_len:
            pad_ts = torch.zeros(1, context_len - TS.shape[1], device=device, dtype=torch.long)
            TS = torch.cat([pad_ts, TS], dim=1)
        TS = torch.clamp(TS, min=0, max=max_ts - 1)

        # 2) 组装 S/A/RTG/(V1,V2) 同步滑窗 + 左 pad
        S = _stack_and_pad([norm_state(x) for x in states_hist], context_len, device, dtype=torch.float32)  # (1,T,Ds)

        if len(actions_hist) == 0:
            actions_hist.append(np.zeros((act_dim,), dtype=np.float32))
        A = _stack_and_pad(actions_hist, context_len, device, dtype=torch.float32)  # (1,T,Da)

        RTG = _stack_and_pad([[x] for x in rtgs_hist], context_len, device, dtype=torch.float32)  # (1,T,1)

        V1 = V2 = None
        if v1_dim > 0:
            if len(vision1_hist) == 0:
                vision1_hist.append(np.zeros((v1_dim,), dtype=np.float32))
            V1 = _stack_and_pad(vision1_hist, context_len, device, dtype=torch.float32)
        if v2_dim > 0:
            if len(vision2_hist) == 0:
                vision2_hist.append(np.zeros((v2_dim,), dtype=np.float32))
            V2 = _stack_and_pad(vision2_hist, context_len, device, dtype=torch.float32)

        # 3) 模型出动作
        act_t = policy.act(
            timesteps=TS, states=S, actions=A, returns_to_go=RTG,
            vision1=V1, vision2=V2, temperature=temperature
        ).squeeze(0).detach().cpu().numpy()  # (Da,)

        # 4) 与环境交互
        obs, reward, done, info = env.step(act_t[None, ...])  # (1,Da)
        reward = float(reward.squeeze().detach().cpu().numpy()) if torch.is_tensor(reward) else float(reward)
        done_bool = bool(done.squeeze().detach().cpu().numpy()) if torch.is_tensor(done) else bool(done)

        ep_ret += reward
        ep_len += 1

        # 5) 更新历史（注意顺序：先记录动作，再记录新 state/vision）
        actions_hist.append(act_t.astype(np.float32))
        s_next = obs_to_state(obs)
        states_hist.append(s_next)

        if vision_encoder is not None:
            v1_t, v2_t = encode_vision(obs, vision_encoder, device)
            if v1_dim > 0:
                vision1_hist.append(v1_t.astype(np.float32))
            if v2_dim > 0:
                vision2_hist.append(v2_t.astype(np.float32))

        # 6) 更新 RTG（决策 Transformer 常见写法）
        rtgs_hist.append(max(0.0, rtgs_hist[-1] - reward))

        if done_bool:
            break

    return {"return": ep_ret, "length": ep_len}


# --------------------- 维度/统计 兜底推断（通常用不到） ---------------------
def infer_dims_from_dataset(args, device):
    ds = ReinformerFurnitureDataset(
        root_dir=args.data_path,
        context_len=args.context_len,
        vision_mode="none",
        feature_root=None,
        gamma=1.0,
        sample_stride=1,
        max_episodes=args.limit_episodes,
        episode_range="",
        task_filter=args.tasks if args.tasks else None,
        strict_feature_len=False,
        verbose_mismatch=True,
    )
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)))
    (_, states, actions, _, _, _, _, _) = batch
    Ds = int(states.shape[-1]); Da = int(actions.shape[-1])
    state_mean = states.mean(dim=(0,1)).to(device).cpu().numpy()
    state_std  = states.std(dim=(0,1)).clamp_min(1e-6).to(device).cpu().numpy()
    return Ds, Da, state_mean, state_std


# --------------------- 主函数 ---------------------
def main():
    ap = argparse.ArgumentParser()
    # 模型/数据
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--tasks", type=str, nargs="*", default=["lamp"])
    ap.add_argument("--context_len", type=int, default=10)
    ap.add_argument("--limit_episodes", type=int, default=100)

    # 环境
    ap.add_argument("--furniture", type=str, default="lamp")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max_env_steps", type=int, default=3000)
    ap.add_argument("--eval_episodes", type=int, default=10)

    # 视觉：none | r3m | vip
    ap.add_argument("--vision", type=str, default="r3m")

    # 模型结构（若 ckpt.variant 缺项时使用）
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout_p", type=float, default=0.1)
    ap.add_argument("--grad_norm", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--init_temperature", type=float, default=0.1)
    ap.add_argument("--max_timestep", type=int, default=4096)
    ap.add_argument("--dt_mask", action="store_true")

    # 其它
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    device = torch.device(args.device)

    # 1) 读 ckpt 里的 dims/variant/统计
    ckpt = torch.load(args.ckpt, map_location="cpu")
    dims = ckpt.get("dims", {}) or {}
    Ds = dims.get("state_dim")
    Da = dims.get("act_dim")
    v1_dim_ck = dims.get("vision1_dim", 0) or 0
    v2_dim_ck = dims.get("vision2_dim", 0) or 0

    state_mean = ckpt.get("state_mean", None)
    state_std  = ckpt.get("state_std", None)
    if isinstance(state_mean, torch.Tensor): state_mean = state_mean.numpy()
    if isinstance(state_std, torch.Tensor):  state_std  = state_std.numpy()

    # 若缺失维度或统计，兜底推断
    if Ds is None or Da is None:
        Ds, Da, state_mean, state_std = infer_dims_from_dataset(args, device)

    # 2) 组装 variant（优先用 ckpt.variant）
    var_ck = ckpt.get("variant", {}) or {}
    variant = dict(
        n_blocks=var_ck.get("n_blocks", args.n_blocks),
        embed_dim=var_ck.get("embed_dim", args.embed_dim),
        context_len=var_ck.get("context_len", args.context_len),
        n_heads=var_ck.get("n_heads", args.n_heads),
        dropout_p=var_ck.get("dropout_p", args.dropout_p),
        grad_norm=var_ck.get("grad_norm", args.grad_norm),
        tau=var_ck.get("tau", args.tau),
        lr=var_ck.get("lr", args.lr),
        wd=var_ck.get("wd", args.wd),
        warmup_steps=var_ck.get("warmup_steps", args.warmup_steps),
        init_temperature=var_ck.get("init_temperature", args.init_temperature),
        max_timestep=var_ck.get("max_timestep", args.max_timestep),
        dt_mask=var_ck.get("dt_mask", args.dt_mask),
        vision1_dim=v1_dim_ck,
        vision2_dim=v2_dim_ck,
    )

    # 3) 构建 policy
    policy = ReinFormerPolicy(state_dim=Ds, act_dim=Da, device=device, variant=variant, ckpt_path=args.ckpt)

    # 4) 视觉编码器（如需）
    vision_encoder, dim_enc = build_encoder(args.vision, device)
    # 一致性检查（如果 ckpt 需要视觉，但你传 none -> 用 0 占位。若维度不匹配，会抛异常。）
    if args.vision.lower() != "none":
        if (v1_dim_ck and dim_enc != v1_dim_ck) or (v2_dim_ck and dim_enc != v2_dim_ck):
            raise ValueError(
                f"Checkpoint expects vision dims (v1={v1_dim_ck}, v2={v2_dim_ck}) "
                f"but encoder '{args.vision}' outputs {dim_enc}. "
                f"请用与训练一致的编码器。"
            )
    obs_keys = [
        "robot_state/joint_positions",
        "robot_state/ee_pos",
        "robot_state/ee_quat",
        "parts_poses",
    ]
    if args.vision.lower() != "none":
        obs_keys += ["color_image1", "color_image2"]
    # 5) 建环境（启用我们需要的观测）
    env: FurnitureSimEnv = FurnitureSimEnv(
        furniture=args.furniture,
        num_envs=1,
        headless=args.headless,
        obs_keys=obs_keys, 
        concat_robot_state=False,   # 我们手动拼接 state
        act_rot_repr="quat",
        action_type="delta",
        ctrl_mode="osc",
        max_env_steps=args.max_env_steps,
        np_step_out=False,
        channel_first=False,
        randomness="low",
    )

    # 6) 多次评估
    returns, lengths = [], []
    t0 = time.time()
    for ep in range(args.eval_episodes):
        stats = rollout_episode(
            env, policy, device,
            context_len=variant["context_len"],
            target_return=1.0,
            max_steps=args.max_env_steps,
            state_mean=state_mean, state_std=state_std,
            vision_encoder=vision_encoder,
            v1_dim=variant["vision1_dim"], v2_dim=variant["vision2_dim"],
            temperature=args.temperature,
        )
        returns.append(stats["return"]); lengths.append(stats["length"])
        print(f"[eval] ep={ep+1}/{args.eval_episodes}  return={stats['return']:.2f}  len={stats['length']}")

    dt = time.time() - t0
    print(f"[summary] avg_return={np.mean(returns):.2f}  avg_len={np.mean(lengths):.1f}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()


"""
用法示例（你的路径）：
export LD_LIBRARY_PATH=/opt/nvidia525
export VK_ICD_FILENAMES=/opt/nvidia525/icd525.json
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_MESA_device_select:VK_LAYER_MESA_overlay:VK_LAYER_NV_device_diagnostics_tooling:VK_LAYER_NV_optimus
unset VK_INSTANCE_LAYERS VK_LAYER_PATH
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY=
export XDG_RUNTIME_DIR=/tmp/$USER-runtime; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"

VK_LOADER_DEBUG=info vulkaninfo | head -n 50  # 应该看到 /opt/nvidia525/icd525.json；不要出现 lavapipe

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="/liujinxin/isaacgym/robust-rearrangement/furniture-bench:$PYTHONPATH" \
python -u -m eval.eval_sim \
  --ckpt checkpoints/reinformer_final_step1000.pt \
  --data_path /liujinxin/isaacgym/robust-rearrangement/furniture-bench/furniture_dataset \
  --tasks lamp \
  --furniture lamp \
  --context_len 10 \
  --eval_episodes 5 \
  --vision r3m \
  --headless \
  --temperature 0.0 \
  --device cuda
"""