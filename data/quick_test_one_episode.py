# quick_test_one_episode.py
from dataset import ReinformerFurnitureDataset

ROOT = "/defaultShare/furniture_bench_dataset_lerobot"  # 根目录（含 meta/info.json）
EP_ID = 0                                               # 想看的 episode_id
CTX = 10

ds = ReinformerFurnitureDataset(
    data_path=ROOT,
    context_len=CTX,
    vision_mode="raw",     # raw=解码视频帧；cache=读预特征；none=不带视觉
)

# 只保留这一条 episode 的采样索引
epi = next(i for i,m in enumerate(ds.ep_meta) if m.get("ep_idx")==EP_ID)
ds.sample_index = [(i,st) for (i,st) in ds.sample_index if i==epi]

print("episode length:", ds.ep_meta[epi]["length"], "num samples:", len(ds))
x = ds[0]
names = ["timesteps","states","actions","returns_to_go","rewards","mask","vision1/IMG1","vision2/IMG2"]
for n,t in zip(names,x):
    print(f"{n:14s}", tuple(t.shape), getattr(t,"dtype",type(t)))


print(ds[0]["rewards"])


