# eval_offline.py
import torch

def _masked_mean(x, mask, eps: float = 1e-8):
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

@torch.no_grad()
def evaluate_offline(
    model: torch.nn.Module,
    device: torch.device,
    val_loader,
    tau: float = 0.8,
    lambda_H: float = 0.05,  # 熵权重
    lambda_R: float = 0.0,   # RTG MAE 权重（先设 0，简单稳）
):
    """
    返回：{
      'nll_per_t', 'entropy_per_t', 'rtg_mae_per_t', 'score'
    }
    score 越大越好：-NLL + λ_H * H - λ_R * MAE
    """
    model.eval()
    nll_sum = 0.0
    ent_sum = 0.0
    rtg_mae_sum = 0.0
    tok_cnt = 0.0

    for batch in val_loader:
        (ts, s, a, rtg, r, msk, v1, v2) = batch
        ts = ts.to(device)                      # (B,T)
        s  = s.to(device)                       # (B,T,Ds)
        a  = a.to(device)                       # (B,T,Da)
        rtg = rtg.to(device).unsqueeze(-1)      # (B,T,1)
        msk = msk.to(device).float()            # (B,T)
        v1 = (v1.to(device) if (v1 is not None and v1.numel()>0) else None)
        v2 = (v2.to(device) if (v2 is not None and v2.numel()>0) else None)

        rtg_pred, act_dist, _ = model(ts, s, a, rtg, v1, v2)

        logp = act_dist.log_prob(a).sum(dim=-1)   # (B,T)
        ent  = act_dist.entropy().sum(dim=-1)     # (B,T)

        nll_sum     += float(-(logp * msk).sum().cpu())
        ent_sum     += float((ent * msk).sum().cpu())
        rtg_mae_sum += float(((rtg_pred - rtg).abs() * msk.unsqueeze(-1)).sum().cpu())
        tok_cnt     += float(msk.sum().cpu())

    nll_pt  = nll_sum / max(1.0, tok_cnt)
    ent_pt  = ent_sum / max(1.0, tok_cnt)
    rtg_mae = rtg_mae_sum / max(1.0, tok_cnt)

    score = -nll_pt + lambda_H * ent_pt - lambda_R * rtg_mae
    return {
        "nll_per_t": nll_pt,
        "entropy_per_t": ent_pt,
        "rtg_mae_per_t": rtg_mae,
        "score": score,
    }
