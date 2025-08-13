import torch
import torch.nn.functional as F
from model import ReinFormer
from lamb import Lamb

def masked_mean(x, mask, eps: float = 1e-8):
    """
    x:   (B,T,...) 或 (B,T)
    mask:(B,T) -> 1 有效, 0 无效
    返回按 mask 的平均
    """
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den



class ReinFormerTrainer:
    def __init__(
        self, 
        state_dim,
        act_dim,
        device,
        variant
    ):
        super().__init__()
                
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device
        self.grad_norm = variant["grad_norm"]
        self.tau = variant["tau"]                       # expectile
        self.context_len = variant["context_len"]

        v1_dim = variant.get("vision1_dim", 0)
        v2_dim = variant.get("vision2_dim", 0)

        self.model = ReinFormer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=variant["n_blocks"],
            h_dim=variant["embed_dim"],
            context_len=variant["context_len"],
            n_heads=variant["n_heads"],
            drop_p=variant["dropout_p"],
            init_temperature=variant["init_temperature"],
            target_entropy=-self.act_dim,
            max_timestep=variant.get("max_timestep", 4096),
            dt_mask=variant.get("dt_mask", False),
            vision1_dim=v1_dim,
            vision2_dim=v2_dim,
        ).to(self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["lr"],
            weight_decay=variant["wd"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/variant["warmup_steps"], 1)
        )

        self.tau = variant["tau"]
        self.context_len=variant["context_len"]


        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    
    def train_step(
        self,
        timesteps,
        states,
        actions,
        returns_to_go,
        rewards,
        traj_mask,
        vision1=None,   # (B,T,Dv1) or None
        vision2=None,   # (B,T,Dv2) or None
    ):
        self.model.train()
        # data to gpu ------------------------------------------------
        timesteps = timesteps.to(self.device)      # B x T
        states = states.to(self.device)            # B x T x state_dim
        actions = actions.to(self.device)          # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).unsqueeze(-1)  # (B,T,1)
        rewards      = rewards.to(self.device).unsqueeze(-1)         # (B,T,1)
        traj_mask    = traj_mask.to(self.device).float()    # (B,T)

        # vision --------------------------------------------------
        if vision1 is not None:
            vision1 = vision1.to(self.device)
            if vision1.ndim == 5:  # (B,T,3,224,224)
                raise ValueError(
                    "trainer 收到 raw 像素 (B,T,3,224,224)。请用 cache 特征（R3M/VIP），"
                    "或在进入 trainer 前自行加 CNN encoder。"
                )
            if vision1.shape[-1] == 0:
                vision1 = None
        if vision2 is not None:
            vision2 = vision2.to(self.device)
            if vision2.ndim == 5:
                raise ValueError(
                    "trainer 收到 raw 像素 (B,T,3,224,224)。请用 cache 特征（R3M/VIP），"
                    "或在进入 trainer 前自行加 CNN encoder。"
                )
            if vision2.shape[-1] == 0:
                vision2 = None

        # model forward ----------------------------------------------
        rtg_preds, action_dist, _ = self.model(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            vision1=vision1,
            vision2=vision2,
        )

        # returns_to_go_loss -----------------------------------------
        # u = (target - pred)/norm, 其中 target=returns_to_go
        target_rtg = returns_to_go                              # (B,T,1)
        pred_rtg   = rtg_preds                                   # (B,T,1)
        norm = target_rtg.abs().mean().clamp_min(1e-6)
        u = (target_rtg - pred_rtg) / norm                       # (B,T,1)
        rtg_loss = masked_mean(torch.abs(self.tau - (u < 0).float()) * (u ** 2), traj_mask)

        # action_loss ------------------------------------------------
        # log_prob/entropy: (B,T,Da) -> sum over action dims -> (B,T)
        log_prob = action_dist.log_prob(actions).sum(dim=-1)     # (B,T)
        entropy  = action_dist.entropy().sum(dim=-1)             # (B,T)

        action_loss = -masked_mean(log_prob + self.model.temperature().detach() * entropy, traj_mask) # 取负号是因为我们最小化 loss

        loss = rtg_loss + action_loss

        # optimization -----------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.grad_norm
        )
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        # temperature_loss = (
        #     self.model.temperature() * (entropy - self.model.target_entropy).detach()
        # )
        temperature_loss = self.model.temperature() * (
            masked_mean(entropy, traj_mask) - self.model.target_entropy
        ).detach()
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        self.scheduler.step()

        # return loss.detach().cpu().item()
        return {
            "loss": float(loss.detach().cpu()),
            "rtg_loss": float(rtg_loss.detach().cpu()),
            "action_loss": float(action_loss.detach().cpu()),
            "entropy": float(masked_mean(entropy, traj_mask).detach().cpu()),
            "temperature": float(self.model.temperature().detach().cpu()),
        }