# tools/isaac_smoke.py
from isaacgym import gymapi
import os

def try_create(use_gpu_pipeline: bool, gpu_id: int = 0):
    print(f"\n[Smoke] use_gpu_pipeline={use_gpu_pipeline}, gpu_id={gpu_id}")
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.use_gpu_pipeline = use_gpu_pipeline
    sim_params.substeps = 1
    sim_params.dt = 1.0 / 60.0
    # 只用 PhysX，图形设备=GPU
    sim = gym.create_sim(0, gpu_id, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "create_sim returned None"
    print("[Smoke] sim created OK")
    gym.destroy_sim(sim)
    print("[Smoke] destroy_sim OK")

if __name__ == "__main__":
    # 先试 GPU pipeline，再试 CPU pipeline（禁用 GPU pipeline）
    try_create(True, 0)
    try_create(False, 0)
