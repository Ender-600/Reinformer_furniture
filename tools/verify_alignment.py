#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd

FRONT_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"

def _is_root_with_meta(p: Path) -> bool:
    return p.is_dir() and (p/"meta"/"info.json").exists() and (p/"meta"/"episodes.jsonl").exists()

def _load_jsonl(p: Path):
    with open(p, "r") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def _read_info(root: Path):
    info = json.load(open(root/"meta"/"info.json","r"))
    data_tpl  = info["data_path"]
    video_tpl = info.get("video_path", None)
    chunks    = int(info.get("chunks_size", 1000))
    return data_tpl, video_tpl, chunks

def _parquet_path(root: Path, data_tpl: str, chunks: int, ep_idx: int) -> Path:
    return root / data_tpl.format(episode_chunk=ep_idx//chunks, episode_index=ep_idx)

def _feature_path(feat_root: Path, chunks: int, ep_idx: int) -> Path:
    return feat_root / f"chunk-{ep_idx//chunks:03d}" / f"episode_{ep_idx:06d}.npz"

def _video_path(root: Path, video_tpl: str, chunks: int, ep_idx: int, video_key: str) -> Path:
    return root / video_tpl.format(episode_chunk=ep_idx//chunks, episode_index=ep_idx, video_key=video_key)

def _len_parquet_actions(pq: Path) -> int:
    try:
        df = pd.read_parquet(pq, columns=["action"])
        return len(df)
    except Exception:
        # 退化：读表头/全表
        df = pd.read_parquet(pq)
        return len(df)

def _len_parquet_rewards(pq: Path) -> int:
    # 旧库 reward 可能是标量或向量（多头），统一成标量长度
    try:
        df = pd.read_parquet(pq, columns=["reward"])
    except Exception:
        df = pd.read_parquet(pq)
        if "reward" not in df.columns:
            return -1
    try:
        arrs = np.stack(df["reward"].to_numpy())
        return len(arrs)
    except Exception:
        return len(df)

def _len_npz_feats(npz: Path):
    if not npz.exists():
        return -1, -1
    try:
        with np.load(npz, allow_pickle=False) as z:
            v1 = z.get("vision1", None)
            v2 = z.get("vision2", None)
            L1 = int(v1.shape[0]) if v1 is not None else -1
            L2 = int(v2.shape[0]) if v2 is not None else -1
            return L1, L2
    except Exception:
        return -1, -1

def _len_video_fast(vpath: Path):
    # 可选：尝试 decord/torchvision/imageio/cv2 任意一个拿帧数
    # 返回 -1 表示无法读取
    try:
        import decord
        vr = decord.VideoReader(str(vpath), num_threads=1)
        return len(vr)
    except Exception:
        pass
    try:
        import torchvision
        v, _, _ = torchvision.io.read_video(str(vpath), pts_unit="sec")
        return int(v.shape[0])
    except Exception:
        pass
    try:
        import imageio
        rdr = imageio.get_reader(str(vpath))
        cnt = 0
        for _ in rdr:
            cnt += 1
        rdr.close()
        return cnt
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return -1
        # 有的编解码器帧计数不准，兜底逐帧数（很慢），这里只读属性：
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else -1
    except Exception:
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_root", required=True, help="新 LeRobot 根（含视频）")
    ap.add_argument("--old_root", required=True, help="旧 LeRobot 根（含 reward）")
    ap.add_argument("--feat_root", default="", help="特征根（可选，npz 的 vision1/vision2）")
    ap.add_argument("--limit", type=int, default=200, help="最多检查多少条（按交集排序）")
    ap.add_argument("--check_video", type=int, default=0, help="是否检查视频帧数（慢）")
    ap.add_argument("--out", default="report_alignment.csv", help="CSV 报告输出路径")
    args = ap.parse_args()

    new_root = Path(args.new_root)
    old_root = Path(args.old_root)
    feat_root = Path(args.feat_root) if args.feat_root else None

    assert _is_root_with_meta(new_root), f"new_root 非法: {new_root}"
    assert _is_root_with_meta(old_root), f"old_root 非法: {old_root}"
    if feat_root:
        assert feat_root.exists(), f"feat_root 不存在: {feat_root}"

    new_data_tpl, new_video_tpl, new_chunks = _read_info(new_root)
    old_data_tpl, _, old_chunks = _read_info(old_root)
    if new_video_tpl is None and args.check_video:
        print("[warn] info.json 无 video_path；忽略 --check_video")
        args.check_video = 0

    # 建立 ep 索引集合
    new_eps = [int(o["episode_index"]) for o in _load_jsonl(new_root/"meta"/"episodes.jsonl")]
    old_eps = [int(o["episode_index"]) for o in _load_jsonl(old_root/"meta"/"episodes.jsonl")]
    inter = sorted(set(new_eps).intersection(old_eps))
    if args.limit > 0:
        inter = inter[:args.limit]

    print(f"[info] 交集 episodes 数: {len(inter)} (limit={args.limit})")

    rows = []
    mism = 0
    for k, ep in enumerate(inter, 1):
        new_pq = _parquet_path(new_root, new_data_tpl, new_chunks, ep)
        old_pq = _parquet_path(old_root, old_data_tpl, old_chunks, ep)
        if not new_pq.exists():
            rows.append([ep, "MISS_NEW_PARQUET", "", "", "", "", "", ""])
            mism += 1
            continue
        if not old_pq.exists():
            rows.append([ep, "MISS_OLD_PARQUET", "", "", "", "", "", ""])
            mism += 1
            continue

        len_new = _len_parquet_actions(new_pq)
        len_old = _len_parquet_rewards(old_pq)

        v1_len = v2_len = ""
        if feat_root:
            fpath = _feature_path(feat_root, new_chunks, ep)
            l1, l2 = _len_npz_feats(fpath)
            v1_len = l1
            v2_len = l2

        vid1 = vid2 = ""
        if args.check_video:
            v1p = _video_path(new_root, new_video_tpl, new_chunks, ep, FRONT_KEY)
            v2p = _video_path(new_root, new_video_tpl, new_chunks, ep, WRIST_KEY)
            vid1 = _len_video_fast(v1p) if v1p.exists() else -1
            vid2 = _len_video_fast(v2p) if v2p.exists() else -1

        # 判断是否一致（允许 off-by-one? 可按需放宽）
        ok = True
        base = len_new
        comps = [x for x in [len_old, v1_len if v1_len != "" else None, v2_len if v2_len != "" else None,
                             vid1 if vid1 != "" else None, vid2 if vid2 != "" else None] if x is not None]
        for c in comps:
            if c == -1:
                ok = False
                break
            if c != base:
                ok = False
        status = "OK" if ok else "MISMATCH"

        if status != "OK":
            mism += 1

        rows.append([ep, status, len_new, len_old, v1_len, v2_len, vid1, vid2])

        if k % 20 == 0:
            print(f"[{k}/{len(inter)}] ep={ep} -> {status}")

    # 写 CSV
    outp = Path(args.out)
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_index", "status",
                    "len_new_parquet", "len_old_reward",
                    "len_feat_v1", "len_feat_v2",
                    "len_video_front", "len_video_wrist"])
        for r in rows:
            w.writerow(r)

    total = len(inter)
    print("\n===== Summary =====")
    print(f"Checked episodes: {total}")
    print(f"Mismatched or missing: {mism} ({mism/total*100:.1f}%)")
    print(f"CSV saved to: {outp}")

if __name__ == "__main__":
    main()


