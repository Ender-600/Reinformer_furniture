import os
import argparse
import pickle
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import torch


# ------------------------ Image helpers ------------------------
def _decode_image_any(obj) -> np.ndarray:
    """把 pkl 中的图像对象转为 float32 HWC[0..1]。自动识别 CHW->HWC。"""
    if isinstance(obj, np.ndarray):
        arr = obj.astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.max() > 1.5:  # 0..255 -> 0..1
            arr /= 255.0
        return arr

    if isinstance(obj, (bytes, bytearray, memoryview)):
        im = Image.open(BytesIO(obj)).convert("RGB")
        return np.asarray(im, np.float32) / 255.0

    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            im = Image.open(BytesIO(obj["bytes"])).convert("RGB")
            return np.asarray(im, np.float32) / 255.0
        if obj.get("data") is not None:
            im = Image.open(BytesIO(obj["data"])).convert("RGB")
            return np.asarray(im, np.float32) / 255.0
        if obj.get("path"):
            im = Image.open(obj["path"]).convert("RGB")
            return np.asarray(im, np.float32) / 255.0
        if "array" in obj:
            return _decode_image_any(np.asarray(obj["array"]))
        raise TypeError(f"Unsupported image dict keys: {list(obj.keys())}")

    if isinstance(obj, str) and os.path.exists(obj):
        im = Image.open(obj).convert("RGB")
        return np.asarray(im, np.float32) / 255.0

    raise TypeError(f"Unsupported image type: {type(obj)}")


def _to_224_chw_float(x_hwc01: np.ndarray) -> torch.Tensor:
    """HWC[0..1] -> (3,224,224) float32，保持 0..1，不再做 ImageNet 归一化。"""
    img = Image.fromarray((x_hwc01 * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)  # (3,224,224)
    return t


# ------------------------ Encoder loader ------------------------
def build_encoder(encoder: str, device: torch.device):
    enc = encoder.lower()
    if enc == "r3m":
        from r3m import load_r3m
        net = load_r3m("resnet50").to(device).eval()
        out_dim = 2048
    elif enc == "vip":
        from vip import load_vip
        net = load_vip().to(device).eval()
        out_dim = 1024
    else:
        raise ValueError("encoder must be 'r3m' or 'vip'")
    for p in net.parameters():
        p.requires_grad_(False)
    return net, out_dim


# ------------------------ File listing ------------------------
def list_pkl_files(root_dir: Path) -> List[Path]:
    if root_dir.is_file() and root_dir.suffix == ".pkl":
        return [root_dir]
    return sorted(root_dir.rglob("*.pkl"))


def filter_by_furniture(paths: List[Path], furn_list: Optional[List[str]]) -> List[Path]:
    if not furn_list:
        return paths
    toks = [s.strip().lower() for s in furn_list if s.strip()]
    keep = []
    for p in paths:
        low = "/".join(p.parts).lower()
        if any(tok in low for tok in toks):
            keep.append(p)
    return keep


def filter_by_episode_range(paths: List[Path], episode_range: str) -> List[Path]:
    """基于文件名 stem 的整数过滤（如 00000.pkl -> 0）"""
    if not episode_range:
        return paths
    lo_s, hi_s = episode_range.split(":")
    lo = 0 if lo_s == "" else int(lo_s)
    hi = 10**12 if hi_s == "" else int(hi_s)

    def ep_id(p: Path) -> int:
        try:
            return int(p.stem)
        except Exception:
            return -1

    return [p for p in paths if lo <= ep_id(p) < hi]


def filter_by_episode_ids(paths: List[Path], episode_ids: List[int]) -> List[Path]:
    if not episode_ids:
        return paths
    idset = set(int(x) for x in episode_ids)
    out = []
    for p in paths:
        try:
            sid = int(p.stem)
        except Exception:
            continue
        if sid in idset:
            out.append(p)
    return out


def feature_path_for_pkl(feature_root: Path, root_dir: Path, pkl_path: Path) -> Path:
    """把 <root_dir>/.../00000.pkl -> <feature_root>/.../00000.npz"""
    rel = pkl_path.relative_to(root_dir).with_suffix(".npz")
    return feature_root / rel


# ------------------------ Main ------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract visual features from furniture_dataset PKL episodes.")
    ap.add_argument("--root_dir", type=str, required=True,
                    help="furniture_dataset 的根目录或任意子目录（会递归找 *.pkl）")
    ap.add_argument("--feature_root", type=str, required=True,
                    help="输出根目录，特征将镜像原目录结构保存为 *.npz（vision1/vision2）")
    ap.add_argument("--encoder", type=str, choices=["r3m", "vip"], required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--skip_exists", action="store_true", help="若目标 npz 已存在则跳过")

    # 选择子集
    ap.add_argument("--furniture", type=str, default="",
                    help="逗号分隔的家具名子串（如 'lamp,round_table'），仅处理匹配路径的文件")
    ap.add_argument("--episode_range", type=str, default="",
                    help="基于文件名 stem 的范围过滤（例如 '0:100'、':50'、'100:'）")
    ap.add_argument("--episode_ids", type=str, default="",
                    help="逗号分隔的 episode id 列表（基于文件名 stem，如 '0,5,12'）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个 episode（0 表示不限）")
    args = ap.parse_args()

    # device
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but no CUDA device is available.")
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    root_dir = Path(args.root_dir).resolve()
    feature_root = Path(args.feature_root).resolve()
    feature_root.mkdir(parents=True, exist_ok=True)

    enc, D = build_encoder(args.encoder, device)
    print(f"[init] encoder={args.encoder}  D={D}  device={device}")

    # 构建待处理列表
    pkls = list_pkl_files(root_dir)
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found under: {root_dir}")

    furn_list = [s for s in args.furniture.split(",") if s.strip()] if args.furniture else None
    pkls = filter_by_furniture(pkls, furn_list)
    pkls = filter_by_episode_range(pkls, args.episode_range)
    if args.episode_ids:
        ids = [int(x) for x in args.episode_ids.split(",") if x.strip()]
        pkls = filter_by_episode_ids(pkls, ids)
    if args.limit and args.limit > 0:
        pkls = pkls[: args.limit]

    print(f"[scan] total episodes to process: {len(pkls)}")

    # 逐条处理
    n_total = len(pkls)
    for i, pkl_path in enumerate(pkls, 1):
        out_npz = feature_path_for_pkl(feature_root, root_dir, pkl_path)
        out_npz.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_exists and out_npz.exists():
            print(f"[{i}/{n_total}] skip exists: {out_npz}")
            continue

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[{i}/{n_total}] load fail: {pkl_path} -> {e}")
            continue

        obs = data.get("observations", [])
        L = len(obs)
        if L < 1:
            print(f"[{i}/{n_total}] empty observations: {pkl_path}")
            continue

        # color_image1(wrist) / color_image2(front)
        imgs1 = [obs[t]["color_image1"] for t in range(L)]
        imgs2 = [obs[t]["color_image2"] for t in range(L)]

        # to (N,3,224,224)
        X1 = torch.stack([_to_224_chw_float(_decode_image_any(x)) for x in imgs1], dim=0)
        X2 = torch.stack([_to_224_chw_float(_decode_image_any(x)) for x in imgs2], dim=0)

        # encode in batches
        V1 = torch.zeros((L, D), dtype=torch.float32)
        V2 = torch.zeros((L, D), dtype=torch.float32)

        with torch.no_grad():
            for s in range(0, L, args.batch_size):
                e = min(L, s + args.batch_size)
                v1 = enc(X1[s:e].to(device))
                V1[s:e] = v1.detach().float().cpu()
                v2 = enc(X2[s:e].to(device))
                V2[s:e] = v2.detach().float().cpu()

        np.savez_compressed(out_npz, vision1=V1.numpy(), vision2=V2.numpy())
        print(f"[{i}/{n_total}] saved: {out_npz}  (L={L}, D={D})")


if __name__ == "__main__":
    main()
"""
python tools/extract_features.py \
   --root_dir /liujinxin/isaacgym/robust-rearrangement/furniture-bench/furniture_dataset \
   --feature_root /liujinxin/isaacgym/robust-rearrangement/furniture-bench/features_r3m \
   --encoder r3m \
   --furniture one_leg,drawer,cabinet,chair,desk,lamp,round_table,square_table,stool \
   --skip_exists
"""