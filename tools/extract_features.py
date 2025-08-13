# tools/extract_features_lerobot.py
import os, json, argparse
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
import torch

def _load_jsonl(p):
    with open(p, "r") as f:
        for ln in f:
            ln=ln.strip()
            if ln: yield json.loads(ln)

def _decode_image_any(obj):
    # -> float32 HWC in [0,1]
    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.ndim==3 and arr.shape[0]==3:  # CHW -> HWC
            arr = np.transpose(arr, (1,2,0))
        arr = arr.astype(np.float32)
        if arr.max() > 1.5: arr /= 255.0
        return arr
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
            return _decode_image_any(np.array(obj["array"]))
    if isinstance(obj, str) and os.path.exists(obj):
        im = Image.open(obj).convert("RGB")
        return np.asarray(im, np.float32)/255.0
    raise TypeError(f"Unsupported image type: {type(obj)}")

def _to_224_chw_float(x_hwc01: np.ndarray) -> torch.Tensor:
    # HWC[0,1] -> 224x224 CHW float32 (no further normalization,与很多R3M示例一致)
    img = Image.fromarray((x_hwc01*255).astype(np.uint8)).resize((224,224), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, np.float32)/255.0).permute(2,0,1)  # (3,224,224)
    return t

def build_encoder(encoder: str, device: torch.device):
    encoder = encoder.lower()
    if encoder=="r3m":
        from r3m import load_r3m
        net = load_r3m('resnet50').to(device).eval()
        out_dim = 2048
    elif encoder=="vip":
        from vip import load_vip
        net = load_vip().to(device).eval()
        out_dim = 1024
    else:
        raise ValueError("encoder must be r3m or vip")
    for p in net.parameters(): p.requires_grad_(False)
    return net, out_dim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_root", required=True, type=str,
                    help="根目录，包含 meta/info.json 与 data/chunk-*/episode_*.parquet")
    ap.add_argument("--out_dir", required=True, type=str,
                    help="输出根目录，按 chunk-XXX/episode_YYYYYY.npz 存放")
    ap.add_argument("--encoder", choices=["r3m","vip"], required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--episodes", type=str, default="",
                    help="可选: 逗号分隔的 episode_index 列表，仅处理这些ID")
    ap.add_argument("--skip_exists", action="store_true", help="已存在 npz 则跳过")
    args = ap.parse_args()

        # ---- device selection & checks ----
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but PyTorch reports no CUDA device. Check drivers / CUDA_VISIBLE_DEVICES.")
        device = torch.device(args.device)  # supports 'cuda' or 'cuda:N'
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    root = Path(args.lerobot_root)
    info = json.load(open(root/"meta"/"info.json","r"))
    tpl = info["data_path"]
    chunk_size = info.get("chunks_size", 1000)

    if args.episodes:
        filt = set(int(x) for x in args.episodes.split(","))
    else:
        filt = None

    enc, D = build_encoder(args.encoder, device)
    print(f"[init] device={device}, cuda_available={torch.cuda.is_available()}")
    print(f"[init] encoder device = {next(enc.parameters()).device}")
    amp_enabled = (device.type == "cuda")

    out_root = Path(args.out_dir)

    # 遍历 episodes
    ep_list = list(_load_jsonl(root/"meta"/"episodes.jsonl"))
    print(f"Total episodes: {len(ep_list)}")
    for k, o in enumerate(ep_list):
        ep_idx = int(o["episode_index"]); L = int(o["length"])
        if L < 2: continue
        if filt is not None and ep_idx not in filt: continue

        ch = ep_idx // chunk_size
        pq = root / tpl.format(episode_chunk=ch, episode_index=ep_idx)
        if not pq.exists():
            print(f"[{k+1}/{len(ep_list)}] Missing parquet: {pq}")
            continue

        out_dir = out_root / f"chunk-{ch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"episode_{ep_idx:06d}.npz"
        if args.skip_exists and out_npz.exists():
            print(f"[{k+1}/{len(ep_list)}] Skip exists: {out_npz}")
            continue

        df = pd.read_parquet(pq)
        imgs1 = df["image"].tolist()
        imgs2 = df["wrist_image"].tolist() if "wrist_image" in df.columns else [None]*len(imgs1)

        # 转 tensor batch (N,3,224,224)
        X1 = torch.stack([_to_224_chw_float(_decode_image_any(x)) for x in imgs1], dim=0)
        if imgs2[0] is not None:
            X2 = torch.stack([_to_224_chw_float(_decode_image_any(x)) for x in imgs2], dim=0)
        else:
            X2 = torch.zeros_like(X1)

        # encode in batches
        V1 = torch.zeros((len(X1), D), dtype=torch.float32)
        V2 = torch.zeros((len(X2), D), dtype=torch.float32)
        enc = enc.to(device)
        with torch.no_grad():
            for s in range(0, len(X1), args.batch_size):
                e = min(len(X1), s + args.batch_size)
                v1 = enc(X1[s:e].to(device))
                V1[s:e] = v1.detach().float().cpu()
                if imgs2[0] is not None:
                    v2 = enc(X2[s:e].to(device))
                    V2[s:e] = v2.detach().float().cpu()

        np.savez_compressed(out_npz, vision1=V1.numpy(), vision2=V2.numpy())
        print(f"[{k+1}/{len(ep_list)}] Saved: {out_npz}  (L={len(X1)}, D={D})")

if __name__ == "__main__":
    main()
