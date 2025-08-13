# dataset.py
# -*- coding: utf-8 -*-

import os, json, glob
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

# ================== constants ==================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

FRONT_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"


# ================== utils ==================
def _is_root_with_meta(p: Union[str, Path]) -> bool:
    p = Path(p)
    return p.is_dir() and (p/"meta"/"info.json").exists() and (p/"meta"/"episodes.jsonl").exists()

def _load_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def _decode_image_any(obj) -> np.ndarray:
    """把 parquet/pkl 中的图像对象转成 float32 [0,1] HWC。"""
    if isinstance(obj, np.ndarray):
        arr = obj.astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
            arr = np.transpose(arr, (1,2,0))
        if arr.max() > 1.5:  # uint8 -> [0,1]
            arr = arr / 255.0
        return arr
    if isinstance(obj, (bytes, bytearray, memoryview)):
        im = Image.open(BytesIO(obj)).convert("RGB")
        return np.asarray(im, np.float32) / 255.0
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
            return _decode_image_any(np.array(obj["array"]))
    if isinstance(obj, str) and os.path.exists(obj):
        im = Image.open(obj).convert("RGB")
        return np.asarray(im, np.float32)/255.0
    raise TypeError(f"Unsupported image type: {type(obj)}")

def _img_to_uint8_224(x: np.ndarray) -> np.ndarray:
    """确保 HWC uint8 的 224x224（便于统一标准化）。"""
    if x.dtype != np.uint8:
        x = (x * 255.0).clip(0,255).astype(np.uint8)
    im = Image.fromarray(x).resize((224,224), Image.BILINEAR)
    return np.asarray(im, np.uint8)

def _flatten_robot_state_dict(rs: dict) -> np.ndarray:
    """pkl 的 robot_state dict -> 1D（按官方字段顺序），总 35 维。"""
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
    return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts], axis=0)


# -------- 视频解码：优先 decord，其次 torchvision -> imageio -> cv2 --------
def _read_video_all(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    # 1) decord
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        frames = vr.get_batch(range(len(vr))).asnumpy()  # (L,H,W,3) uint8
        return frames
    except Exception:
        pass
    # 2) torchvision
    try:
        import torchvision
        v, _, _ = torchvision.io.read_video(path, pts_unit="sec")  # (L,H,W,3) uint8
        return v.numpy()
    except Exception:
        pass
    # 3) imageio
    try:
        import imageio
        rdr = imageio.get_reader(path)
        frames = [frame for frame in rdr]
        rdr.close()
        arr = np.stack(frames, axis=0)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    except Exception:
        pass
    # 4) cv2
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        frames=[]
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError("Empty video.")
        arr = np.stack(frames, axis=0).astype(np.uint8)
        return arr
    except Exception as e:
        raise RuntimeError(f"Cannot decode video: {path}. Error={e}")


# ================== 统一数据集 ==================
class MultiSourceFurnitureDataset(Dataset):
    """
    两种来源：
      (A) LeRobot：root_main（主数据+视频路径） + root_reward（补 reward，可选） + root_features（cache 特征，可选）
      (B) PKL：--pkl 或 --pkl_dir

    视觉：
      - raw   -> (T,3,224,224) ImageNet 标准化
      - cache -> (T,D) 读取 npz 的 vision1/vision2
      - none  -> (T,0)

    输出（八元组）：
      (timesteps, states, actions, returns_to_go, rewards, traj_mask, vision1, vision2)
    """

    def __init__(
        self,
        # 选择一种：A(LeRobot) 或 B(PKL)
        root_main: Optional[str] = None,
        root_reward: Optional[str] = None,
        root_features: Optional[str] = None,
        pkl: Optional[str] = None,
        pkl_dir: Optional[str] = None,

        context_len: int = 10,
        gamma: float = 1.0,
        vision_mode: str = "raw",            # raw | cache | none
        reward_agg: str = "sum",
        reward_index: int = 0,
        reward_weights: Optional[List[float]] = None,
        episode_cache_size: int = 6,
        sample_stride: int = 1,
        task_filter_substr: str = "",
    ):
        super().__init__()
        self.context_len = int(context_len)
        self.gamma = float(gamma)
        self.vision_mode = vision_mode.lower().strip()
        self.reward_agg = reward_agg
        self.reward_index = int(reward_index)
        self.reward_weights = None if reward_weights is None else np.array(reward_weights, np.float32)
        self.sample_stride = max(1, int(sample_stride))
        self.task_filter = (task_filter_substr or "").lower().strip()

        # 判定模式
        self.mode = None
        if pkl or pkl_dir:
            self.mode = "pkl"
        elif root_main:
            self.mode = "lerobot"
        else:
            raise ValueError("必须提供 root_main（lerobot）或 pkl/pkl_dir（PKL）。")

        # ---------------- LeRobot 索引 ----------------
        if self.mode == "lerobot":
            self.root_main = Path(root_main)
            assert _is_root_with_meta(self.root_main), f"root_main 不合法: {root_main}"
            self.root_reward = Path(root_reward) if root_reward else None
            if self.root_reward:
                assert _is_root_with_meta(self.root_reward), f"root_reward 不合法: {root_reward}"
            self.root_features = Path(root_features) if root_features else None

            info = json.load(open(self.root_main/"meta"/"info.json", "r"))
            self.main_data_tpl  = info["data_path"]
            self.video_tpl      = info.get("video_path", None)
            self.chunk_size     = int(info.get("chunks_size", 1000))
            self.fps            = info.get("fps", 10)
            if self.video_tpl is None and self.vision_mode == "raw":
                raise RuntimeError("info.json 缺少 video_path，raw 模式无法从视频取像素。")

            if self.root_reward:
                info_r = json.load(open(self.root_reward/"meta"/"info.json", "r"))
                self.reward_data_tpl = info_r["data_path"]
                self.reward_chunk_size = int(info_r.get("chunks_size", 1000))
            else:
                self.reward_data_tpl = None
                self.reward_chunk_size = None

            self.epi_meta: List[Dict] = []
            for o in _load_jsonl(self.root_main/"meta"/"episodes.jsonl"):
                ep_idx = int(o["episode_index"])
                L = int(o["length"])
                if L < 2: continue
                tasks = o.get("tasks", [])
                if self.task_filter:
                    joined = " ".join(tasks) if isinstance(tasks, list) else str(tasks)
                    if self.task_filter not in joined.lower(): 
                        continue
                pq = self._main_parquet_path(ep_idx)
                if pq.exists():
                    self.epi_meta.append({"mode":"lerobot","ep_idx":ep_idx,"chunk":ep_idx//self.chunk_size,"length":L,"pq":pq})

        # ---------------- PKL 索引 ----------------
        else:
            self.pkl_files: List[Path] = []
            if pkl and Path(pkl).is_file():
                self.pkl_files = [Path(pkl)]
            elif pkl_dir and Path(pkl_dir).is_dir():
                self.pkl_files = [Path(x) for x in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))]
            if not self.pkl_files:
                raise FileNotFoundError("未找到 pkl 文件。")
            # 每个 pkl 视为一个 episode
            self.epi_meta = []
            for fp in self.pkl_files:
                try:
                    import pickle
                    with open(fp, "rb") as f:
                        d = pickle.load(f)
                    O, A = d.get("observations", []), d.get("actions", [])
                    L = min(len(O), len(A) + 1)
                    if L >= 2:
                        self.epi_meta.append({"mode":"pkl","path":str(fp),"length":L})
                except Exception:
                    continue
            if not self.epi_meta:
                raise RuntimeError("pkl 中未发现有效 episode。")

        # 采样索引
        self.sample_index: List[Tuple[int,int]] = []
        for i, m in enumerate(self.epi_meta):
            L = m["length"]
            for st in range(0, max(1, L-1), self.sample_stride):
                self.sample_index.append((i, st))

        # LRU
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_order: List[int] = []
        self._cache_max = int(episode_cache_size)

        # 状态统计
        self._state_mean, self._state_std = self._estimate_state_stats(max_ep=16)

    # ---------- LeRobot 路径拼接 ----------
    def _main_parquet_path(self, ep_idx:int) -> Path:
        ch = ep_idx // self.chunk_size
        rel = self.main_data_tpl.format(episode_chunk=ch, episode_index=ep_idx)
        return self.root_main / rel

    def _video_path(self, ep_idx:int, video_key:str) -> Path:
        ch = ep_idx // self.chunk_size
        rel = self.video_tpl.format(episode_chunk=ch, episode_index=ep_idx, video_key=video_key)
        return self.root_main / rel

    def _reward_parquet_path(self, ep_idx:int) -> Optional[Path]:
        if self.root_reward is None or self.reward_data_tpl is None:
            return None
        ch = ep_idx // (self.reward_chunk_size or self.chunk_size)
        rel = self.reward_data_tpl.format(episode_chunk=ch, episode_index=ep_idx)
        return self.root_reward / rel

    def _feature_path(self, ep_idx:int) -> Optional[Path]:
        if self.root_features is None:
            return None
        ch = ep_idx // self.chunk_size
        return self.root_features / f"chunk-{ch:03d}" / f"episode_{ep_idx:06d}.npz"

    # ---------- 懒加载 ----------
    def _read_episode(self, epi: int) -> Dict[str, np.ndarray]:
        if epi in self._cache:
            return self._cache[epi]
        m = self.epi_meta[epi]

        if m["mode"] == "lerobot":
            ep_idx = m["ep_idx"]; pq = m["pq"]
            # state/action
            df = pd.read_parquet(pq, columns=["observation.state", "action"])
            state  = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            action = np.stack(df["action"].to_numpy()).astype(np.float32)
            L = state.shape[0]
            # reward（来自 reward 根目录）
            reward = np.zeros((L,), dtype=np.float32)
            rpq = self._reward_parquet_path(ep_idx)
            if rpq is not None and rpq.exists():
                df_r = pd.read_parquet(rpq, columns=["reward"])
                rew = np.stack(df_r["reward"].to_numpy())
                if rew.ndim == 2:
                    if self.reward_agg == "sum":
                        rew = rew.sum(axis=1)
                    elif self.reward_agg == "index":
                        rew = rew[:, self.reward_index]
                    elif self.reward_agg == "weighted":
                        if (self.reward_weights is None) or (self.reward_weights.shape[0] != rew.shape[1]):
                            raise ValueError("reward_weights 尺寸不匹配")
                        rew = (rew * self.reward_weights).sum(axis=1)
                    else:
                        raise ValueError(f"Unknown reward_agg={self.reward_agg}")
                rew = rew.astype(np.float32)
                M = min(L, len(rew))
                reward[:M] = rew[:M]
                if M < L and M > 0:
                    reward[M:] = rew[M-1]

            # vision
            image1, image2 = None, None
            vision1 = np.zeros((L,0), np.float32)
            vision2 = np.zeros((L,0), np.float32)

            if self.vision_mode == "cache":
                fp = self._feature_path(ep_idx)
                if fp and fp.exists():
                    with np.load(fp, allow_pickle=False) as z:
                        v1 = np.asarray(z.get("vision1", np.zeros((L,0),np.float32)), np.float32)
                        v2 = np.asarray(z.get("vision2", np.zeros((L,0),np.float32)), np.float32)
                    M = min(L, len(v1), len(v2))
                    vision1, vision2 = v1[:M], v2[:M]
                    state, action, reward = state[:M], action[:M], reward[:M]
                    L = M

            elif self.vision_mode == "raw":
                v1p = self._video_path(ep_idx, FRONT_KEY)
                v2p = self._video_path(ep_idx, WRIST_KEY)
                frames1 = _read_video_all(v1p)
                frames2 = _read_video_all(v2p)
                M = min(L, len(frames1), len(frames2))
                image1, image2 = frames1[:M], frames2[:M]
                state, action, reward = state[:M], action[:M], reward[:M]
                L = M

            blob = dict(
                mode="lerobot", length=L, ep_idx=ep_idx,
                state=state, action=action, reward=reward,
                image1=image1, image2=image2, vision1=vision1, vision2=vision2
            )

        else:  # PKL
            import pickle
            with open(m["path"], "rb") as f:
                d = pickle.load(f)
            O = d["observations"]; A = np.asarray(d["actions"], np.float32); R = np.asarray(d["rewards"], np.float32)
            Lp = m["length"]
            # state（35维）
            state = np.stack([_flatten_robot_state_dict(O[t]["robot_state"]) for t in range(Lp)], axis=0).astype(np.float32)
            # action/reward 对齐到 Lp（最后一帧复制上一帧）
            action = np.zeros((Lp, A.shape[-1]), np.float32)
            reward = np.zeros((Lp,), np.float32)
            if Lp-1 > 0:
                action[:Lp-1] = A[:Lp-1]
                action[Lp-1] = A[Lp-2]
                reward[:Lp-1] = R[:Lp-1]
                reward[Lp-1] = R[Lp-2]
            # 两路像素（已 224x224x3 的也统一一下）
            img1 = np.stack([_img_to_uint8_224(_decode_image_any(O[t]["color_image1"])) for t in range(Lp)], axis=0)
            img2 = np.stack([_img_to_uint8_224(_decode_image_any(O[t]["color_image2"])) for t in range(Lp)], axis=0)

            blob = dict(
                mode="pkl", length=Lp,
                state=state, action=action, reward=reward,
                image1=img1, image2=img2,
                vision1=np.zeros((Lp,0), np.float32), vision2=np.zeros((Lp,0), np.float32)
            )

        # 缓存
        self._cache[epi] = blob
        self._cache_order.append(epi)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return blob

    # ---------- 状态统计 ----------
    def _estimate_state_stats(self, max_ep=16):
        xs=[]
        for m in self.epi_meta[:max_ep]:
            if m["mode"] == "lerobot":
                df = pd.read_parquet(m["pq"], columns=["observation.state"])
                s = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            else:
                import pickle
                with open(m["path"], "rb") as f:
                    d = pickle.load(f)
                L = m["length"]
                s = np.stack([_flatten_robot_state_dict(d["observations"][t]["robot_state"]) for t in range(L)], axis=0).astype(np.float32)
            xs.append(s)
        X = np.concatenate(xs, axis=0)
        m = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        s = X.std(axis=0, dtype=np.float64).astype(np.float32) + 1e-6
        return torch.from_numpy(m), torch.from_numpy(s)

    # ---------- PyTorch Dataset 接口 ----------
    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        epi, start = self.sample_index[idx]
        ep = self._read_episode(epi)

        L   = ep["length"]; T = self.context_len
        end = min(L, start + T); eff = end - start

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
                x = torch.from_numpy(np_imgs[start:end]).permute(0,3,1,2).float().div(255.0)
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
                out = torch.zeros((T,)+tuple(x.shape[1:]), dtype=torch.float32)
                out[:eff] = x
                return out
            v1_t = prep(img1) if img1 is not None else torch.zeros((T,3,224,224), dtype=torch.float32)
            v2_t = prep(img2) if img2 is not None else torch.zeros((T,3,224,224), dtype=torch.float32)
            return timesteps, s, a, rtg, r, msk, v1_t, v2_t

        elif self.vision_mode == "cache":
            def prep_feat(F):
                if F is None or F.shape[1] == 0: return torch.zeros((T,0), dtype=torch.float32)
                sub = torch.from_numpy(F[start:end]).float()
                out = torch.zeros((T, sub.shape[1]), dtype=torch.float32)
                out[:eff] = sub
                return out
            v1_t, v2_t = prep_feat(v1), prep_feat(v2)
            return timesteps, s, a, rtg, r, msk, v1_t, v2_t

        else:  # none
            return timesteps, s, a, rtg, r, msk, torch.zeros((T,0), dtype=torch.float32), torch.zeros((T,0), dtype=torch.float32)


# ================== 简单测试 ==================
if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    # 选一类输入：
    pa.add_argument("--root_main", type=str, default="", help="LeRobot 主根目录")
    pa.add_argument("--root_reward", type=str, default="")
    pa.add_argument("--root_features", type=str, default="")
    pa.add_argument("--pkl", type=str, default="")
    pa.add_argument("--pkl_dir", type=str, default="")

    pa.add_argument("--vision_mode", type=str, default="raw", choices=["raw","cache","none"])
    pa.add_argument("--context_len", type=int, default=10)
    pa.add_argument("--gamma", type=float, default=1.0)
    pa.add_argument("--sample_stride", type=int, default=1)
    pa.add_argument("--only_episode", type=int, default=-1)
    args = pa.parse_args()

    ds = MultiSourceFurnitureDataset(
        root_main=(args.root_main or None),
        root_reward=(args.root_reward or None),
        root_features=(args.root_features or None),
        pkl=(args.pkl or None),
        pkl_dir=(args.pkl_dir or None),
        vision_mode=args.vision_mode,
        context_len=args.context_len,
        gamma=args.gamma,
        sample_stride=args.sample_stride,
    )

    if args.only_episode >= 0:
        try:
            # LeRobot 模式：按 episode_index 过滤
            idx = next(i for i,m in enumerate(ds.epi_meta) if m.get("ep_idx", -1)==args.only_episode)
            ds.sample_index = [(i,st) for (i,st) in ds.sample_index if i==idx]
            print(f"[only_episode={args.only_episode}] samples={len(ds)}")
        except StopIteration:
            print("only_episode 未命中；全量测试。")

    print("Dataset size:", len(ds))
    x = ds[0]
    names = ["timesteps","states","actions","returns_to_go","rewards","mask","vision1/IMG1","vision2/IMG2"]
    for n,t in zip(names,x):
        print(f"{n:16s} shape={tuple(t.shape)} dtype={getattr(t,'dtype',type(t))}")
