# data/dataset.py
import os, pickle
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ================== constants ==================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# ================== utils ==================
def _decode_image_any(obj) -> np.ndarray:
    """把 pkl 中的图像对象转成 float32 [0,1] HWC。自动识别 CHW->HWC。"""
    if isinstance(obj, np.ndarray):
        arr = obj.astype(np.float32)
    else:
        if isinstance(obj, (bytes, bytearray, memoryview)):
            im = Image.open(BytesIO(obj)).convert("RGB")
            return np.asarray(im, np.float32)/255.0
        if isinstance(obj, dict):
            if obj.get("bytes") is not None:
                im = Image.open(BytesIO(obj["bytes"])).convert("RGB")
                return np.asarray(im, np.float32)/255.0
            if obj.get("data") is not None:
                im = Image.open(BytesIO(obj["data"])).convert("RGB")
                return np.asarray(im, np.float32)/255.0
            if obj.get("path"):
                im = Image.open(obj["path"]).convert("RGB")
                return np.asarray(im, np.float32)/255.0
            if "array" in obj:
                arr = np.array(obj["array"], dtype=np.float32)
            else:
                raise TypeError(f"Unsupported image dict keys: {list(obj.keys())}")
        elif isinstance(obj, str) and os.path.exists(obj):
            im = Image.open(obj).convert("RGB")
            return np.asarray(im, np.float32)/255.0
        else:
            raise TypeError(f"Unsupported image type: {type(obj)}")
    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1,2,0))
    # 0..255 -> 0..1
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr

def _img_to_uint8_224(x: np.ndarray) -> np.ndarray:
    """确保 HWC uint8 的 224x224。"""
    if x.dtype != np.uint8:
        x = (x * 255.0).clip(0,255).astype(np.uint8)
    im = Image.fromarray(x).resize((224,224), Image.BILINEAR)
    return np.asarray(im, np.uint8)

def _flatten_robot_state_dict(rs: dict) -> np.ndarray:
    """官方 robot_state -> 1D（总 35 维）。"""
    parts = [
        rs['ee_pos'],           # (3,)
        rs['ee_quat'],          # (4,)
        rs['ee_pos_vel'],       # (3,)
        rs['ee_ori_vel'],       # (3,)
        rs['joint_positions'],  # (7,)
        rs['joint_velocities'], # (7,)
        rs['joint_torques'],    # (7,)
        np.array([rs['gripper_width']], dtype=np.float32),  # (1,)
    ]
    return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts], axis=0)  # (35,)

def _feature_path_for_pkl(feature_root: Optional[Path], root_dir: Path, pkl_path: Path) -> Optional[Path]:
    """把 <root_dir>/.../xxx.pkl 映射到 <feature_root>/.../xxx.npz"""
    if feature_root is None:
        return None
    rel = pkl_path.relative_to(root_dir).with_suffix(".npz")
    return feature_root / rel

# ================== PKL Dataset ==================
class ReinformerFurnitureDataset(Dataset):
    """
    只读取官方 PKL 数据（可给文件/任务目录/整个 root）。

    视觉：
      - raw   -> (T,3,224,224) 标准化像素
      - cache -> (T,D) 从 feature_root 镜像路径加载 npz（键: vision1/vision2）
      - none  -> (T,0)

    输出（八元组）：
      (timesteps, states, actions, returns_to_go, rewards, traj_mask, vision1_or_img1, vision2_or_img2)
    """
    def __init__(
        self,
        root_dir: str,                         # e.g. .../furniture_dataset 或 .../low/lamp
        context_len: int = 10,
        vision_mode: str = "raw",              # "raw" | "cache" | "none"
        feature_root: Optional[str] = None,    # 预特征根目录（cache 模式）
        gamma: float = 1.0,
        sample_stride: int = 1,
        episode_cache_size: int = 4,
        max_episodes: int = 0,
        episode_range: str = "",               # "start:end"
        task_filter: Optional[List[str]] = None,   # ['lamp','round_table'] 等（按路径包含判断）
        strict_feature_len: bool = False,      # cache 下特征长度与 L 不符时，是否报错（默认截齐）
        verbose_mismatch: bool = True,         # 打印长度不匹配的提示
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.context_len = int(context_len)
        self.vision_mode = vision_mode.lower().strip()
        self.feature_root = Path(feature_root) if feature_root else None
        self.gamma = float(gamma)
        self.sample_stride = max(1, int(sample_stride))
        self.strict_feature_len = bool(strict_feature_len)
        self.verbose_mismatch = bool(verbose_mismatch)

        # -------- 索引所有 pkl --------
        if self.root_dir.is_file() and self.root_dir.suffix == ".pkl":
            pkl_files = [self.root_dir]
        else:
            pkl_files = [Path(p) for p in sorted(self.root_dir.rglob("*.pkl"))]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl under {self.root_dir}")

        # 任务过滤（例如只取 lamp）
        if task_filter:
            toks = [t.lower() for t in task_filter]
            keep = []
            for p in pkl_files:
                lowpath = str(p).lower()
                if any(t in lowpath for t in toks):
                    keep.append(p)
            if keep:
                pkl_files = keep

        # 可选 episode 限制
        if episode_range:
            lo, hi = episode_range.split(":")
            lo = 0 if lo=="" else int(lo)
            hi = len(pkl_files) if hi=="" else int(hi)
            pkl_files = pkl_files[lo:hi]
        elif max_episodes and max_episodes > 0:
            pkl_files = pkl_files[:max_episodes]

        # 建 meta（读取长度 & 统一动作维度）
        self.ep_meta: List[Dict] = []
        self._act_dim: Optional[int] = None
        for fp in pkl_files:
            with open(fp, "rb") as f:
                d = pickle.load(f)
            O, A, R = d["observations"], np.asarray(d["actions"]), np.asarray(d["rewards"])
            # 对齐 (s_t, a_t, r_t, s_{t+1})
            L = min(len(O), len(A)+1, len(R)+1)
            if L < 2:
                continue
            act_dim = A.shape[-1]
            if self._act_dim is None:
                self._act_dim = int(act_dim)
            elif int(act_dim) != self._act_dim:
                raise ValueError(f"动作维度不一致：{fp} 有 {act_dim}，先前是 {self._act_dim}")
            self.ep_meta.append({"path": str(fp), "length": int(L)})

        if not self.ep_meta:
            raise RuntimeError("未发现有效 episode。")

        # 采样索引（每条 ep 从 0 到 L-2 可作为起点）
        self.sample_index: List[Tuple[int,int]] = []
        for epi, m in enumerate(self.ep_meta):
            L = m["length"]
            for st in range(0, max(1, L-1), self.sample_stride):
                self.sample_index.append((epi, st))

        # LRU episode 缓存
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_order: List[int] = []
        self._cache_max = int(episode_cache_size)

        # 预估 state 标准化统计
        self._state_mean, self._state_std = self._estimate_state_stats(max_ep=16)

    # ---------- 懒加载一条 episode ----------
    def _read_episode(self, epi: int) -> Dict[str, np.ndarray]:
        if epi in self._cache:
            return self._cache[epi]
        p = Path(self.ep_meta[epi]["path"])
        with open(p, "rb") as f:
            d = pickle.load(f)

        O = d["observations"]
        A = np.asarray(d["actions"], np.float32)
        R = np.asarray(d["rewards"], np.float32)

        # 对齐长度
        L = min(len(O), len(A)+1, len(R)+1)
        if len(O) == len(A):   # 少一帧观测则补最后一帧
            O = O + [O[-1]]

        # 状态：flatten robot_state -> (L, 35)
        state = np.stack([_flatten_robot_state_dict(O[t]["robot_state"]) for t in range(L)], axis=0).astype(np.float32)

        # 动作/奖励对齐到 L
        act_dim = A.shape[-1]
        action = np.zeros((L, act_dim), np.float32)
        reward = np.zeros((L,), np.float32)
        if L-1 > 0:
            action[:L-1] = A[:L-1]
            action[L-1]  = A[L-2]
            reward[:L-1] = R[:L-1]
            reward[L-1]  = R[L-2]

        # 两路像素
        img1 = None
        img2 = None
        if self.vision_mode == "raw":
            img1 = np.stack([_img_to_uint8_224(_decode_image_any(O[t]["color_image1"])) for t in range(L)], axis=0)
            img2 = np.stack([_img_to_uint8_224(_decode_image_any(O[t]["color_image2"])) for t in range(L)], axis=0)

        # 可选 cache 特征（.npz 里键名：vision1/vision2）
        vision1 = np.zeros((L,0), np.float32)
        vision2 = np.zeros((L,0), np.float32)
        if self.vision_mode == "cache":
            fp = _feature_path_for_pkl(self.feature_root, self.root_dir, p)
            if fp is not None and fp.exists():
                with np.load(fp, allow_pickle=False) as z:
                    v1 = z.get("vision1")
                    v2 = z.get("vision2")
                if v1 is not None: v1 = np.asarray(v1, np.float32)
                if v2 is not None: v2 = np.asarray(v2, np.float32)

                # 长度对齐（默认取最短；strict 时报错）
                lens = [L]
                if v1 is not None: lens.append(len(v1))
                if v2 is not None: lens.append(len(v2))
                M = min(lens)
                if (v1 is not None and len(v1)!=M) or (v2 is not None and len(v2)!=M) or (L!=M):
                    if self.strict_feature_len:
                        raise RuntimeError(f"length mismatch @ {fp}: L={L}, v1={None if v1 is None else len(v1)}, v2={None if v2 is None else len(v2)}")
                    if self.verbose_mismatch:
                        print(f"[warn] length mismatch -> clip to {M}: L={L}, v1={None if v1 is None else len(v1)}, v2={None if v2 is None else len(v2)} | {fp}")

                # 截齐
                state  = state[:M]; action = action[:M]; reward = reward[:M]
                if v1 is not None: vision1 = v1[:M]
                if v2 is not None: vision2 = v2[:M]
                L = M

        blob = dict(
            length=L,
            state=state, action=action, reward=reward,
            image1=(img1 if img1 is not None else np.zeros((0,), np.uint8)),
            image2=(img2 if img2 is not None else np.zeros((0,), np.uint8)),
            vision1=vision1, vision2=vision2
        )

        # LRU
        self._cache[epi] = blob
        self._cache_order.append(epi)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return blob

    # ---------- 统计 ----------
    def _estimate_state_stats(self, max_ep=16):
        xs = []
        for m in self.ep_meta[:max_ep]:
            with open(m["path"], "rb") as f:
                d = pickle.load(f)
            O, A = d["observations"], d["actions"]
            L = min(len(O), len(A)+1)
            if len(O) == len(A):
                O = O + [O[-1]]
            s = np.stack([_flatten_robot_state_dict(O[t]["robot_state"]) for t in range(L)], axis=0).astype(np.float32)
            xs.append(s)
        X = np.concatenate(xs, axis=0)
        m = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        s = X.std(axis=0, dtype=np.float64).astype(np.float32) + 1e-6
        return torch.from_numpy(m), torch.from_numpy(s)

    # ---------- Dataset 接口 ----------
    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        epi, start = self.sample_index[idx]
        ep = self._read_episode(epi)

        L = int(ep["length"])
        T = self.context_len
        if start >= L:
            start = max(0, L-1)
        end = min(L, start + T)
        eff = end - start

        state, action, reward = ep["state"], ep["action"], ep["reward"]
        img1, img2 = ep["image1"], ep["image2"]
        v1, v2     = ep["vision1"], ep["vision2"]

        Ds, Da = state.shape[1], action.shape[1]
        timesteps = torch.arange(start, start+T, dtype=torch.long)

        s   = torch.zeros((T, Ds), dtype=torch.float32)
        a   = torch.zeros((T, Da), dtype=torch.float32)
        r   = torch.zeros((T,), dtype=torch.float32)
        msk = torch.zeros((T,), dtype=torch.float32)

        s[:eff] = torch.from_numpy(state[start:end])
        a[:eff] = torch.from_numpy(action[start:end])
        r[:eff] = torch.from_numpy(reward[start:end])
        msk[:eff] = 1.0

        # 标准化 state
        s = (s - self._state_mean) / self._state_std

        # RTG
        ep_r = torch.from_numpy(reward)
        if self.gamma == 1.0:
            rtg_full = torch.flip(torch.cumsum(torch.flip(ep_r, dims=[0]), dim=0), dims=[0])
        else:
            rtg_full = torch.zeros_like(ep_r); run = 0.0
            for t in range(L-1, -1, -1):
                run = float(ep_r[t]) + self.gamma * run
                rtg_full[t] = run
        rtg = torch.zeros((T,), dtype=torch.float32)
        rtg[:eff] = rtg_full[start:end]

        # 视觉
        if self.vision_mode == "raw":
            def prep(np_imgs):
                if np_imgs.size == 0:
                    return torch.zeros((T, 3, 224, 224), dtype=torch.float32)
                x = torch.from_numpy(np_imgs[start:end]).permute(0,3,1,2).float().div(255.0)
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
                out = torch.zeros((T,)+tuple(x.shape[1:]), dtype=torch.float32)
                out[:eff] = x
                return out
            v1_t = prep(img1)
            v2_t = prep(img2)
            return timesteps, s, a, rtg, r, msk, v1_t, v2_t

        elif self.vision_mode == "cache":
            def prep_feat(F):
                if F is None or F.size == 0 or F.shape[1] == 0:
                    return torch.zeros((T,0), dtype=torch.float32)
                sub = torch.from_numpy(F[start:end]).float()
                out = torch.zeros((T, sub.shape[1]), dtype=torch.float32)
                out[:eff] = sub
                return out
            v1_t = prep_feat(v1)
            v2_t = prep_feat(v2)
            return timesteps, s, a, rtg, r, msk, v1_t, v2_t

        else:  # none
            return timesteps, s, a, rtg, r, msk, torch.zeros((T,0), dtype=torch.float32), torch.zeros((T,0), dtype=torch.float32)
