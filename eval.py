import torch
import torch.nn.functional as F
from collections import defaultdict

@torch.no_grad()
def evaluate_offline(
    model,
    device,
    val_loader,          # 验证集 DataLoader（与你训练集同构的 batch：8元组）
    tau: float = 0.99,   # expectile 超参，跟训练一致
):
    model.eval()
    meters = defaultdict(float)
    n_batches = 0

    for batch in val_loader:
        (
            timesteps,   # (B,T)
            states,      # (B,T,Ds)
            actions,     # (B,T,Da)
            returns_to_go, # (B,T)
            rewards,     # (B,T)
            traj_mask,   # (B,T)
            vision1,     # (B,T,Dv1) 或 (B,T,0)
            vision2,     # (B,T,Dv2) 或 (B,T,0)
        ) = batch

        # to device
        timesteps    = timesteps.to(device)
        states       = states.to(device)
        actions      = actions.to(device)
        returns_to_go= returns_to_go.to(device).unsqueeze(-1)  # (B,T,1)
        traj_mask    = traj_mask.to(device)

        v1_in = vision1.to(device) if (hasattr(vision1, "shape") and vision1.shape[-1] > 0) else None
        v2_in = vision2.to(device) if (hasattr(vision2, "shape") and vision2.shape[-1] > 0) else None

        # forward（注意把视觉特征传进去；如果没用就是 None）
        rtg_pred, act_dist, state_pred = model(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            vision1=v1_in,
            vision2=v2_in,
        )

        # --- RTG expectile loss（与训练同式） ---
        norm = returns_to_go.abs().mean().clamp_min(1e-6)
        u = (returns_to_go - rtg_pred) / norm
        rtg_loss = torch.abs(tau - (u < 0).float()) * (u ** 2)          # (B,T,1)
        # masked mean 到标量
        while traj_mask.dim() < rtg_loss.dim():
            traj_mask = traj_mask.unsqueeze(-1)
        rtg_loss = (rtg_loss * traj_mask).sum() / traj_mask.sum().clamp_min(1e-8)

        # --- Action NLL ---
        nll = -act_dist.log_prob(actions).sum(-1)                        # (B,T)
        nll = (nll * traj_mask.squeeze(-1)).sum() / traj_mask.squeeze(-1).sum().clamp_min(1e-8)

        # --- State MSE ---
        smse = F.mse_loss(state_pred, states, reduction='none').sum(-1)  # (B,T)
        smse = (smse * traj_mask.squeeze(-1)).sum() / traj_mask.squeeze(-1).sum().clamp_min(1e-8)

        # --- Entropy（可选监控） ---
        ent = act_dist.entropy().sum(-1)                                 # (B,T)
        ent = (ent * traj_mask.squeeze(-1)).sum() / traj_mask.squeeze(-1).sum().clamp_min(1e-8)

        meters["rtg_loss"]   += rtg_loss.item()
        meters["action_nll"] += nll.item()
        meters["state_mse"]  += smse.item()
        meters["entropy"]    += ent.item()
        n_batches += 1

    for k in meters:
        meters[k] /= max(n_batches, 1)
    return dict(meters)
