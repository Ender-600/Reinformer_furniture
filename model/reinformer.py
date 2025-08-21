import torch
import torch.nn as nn
import numpy as np
from model.net import Block, BaseActor


class ReinFormer(nn.Module):
    def __init__(
        self, 
        state_dim, 
        act_dim, 
        n_blocks, 
        h_dim, 
        context_len,
        n_heads, 
        drop_p, 
        init_temperature,
        target_entropy,
        max_timestep=4096,
        dt_mask=False,
        vision1_dim: int = 0,
        vision2_dim: int = 0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### vision token
        self.n_vision = int(vision1_dim > 0) + int(vision2_dim > 0)
        ### transformer blocks
        self.num_inputs = 3+ self.n_vision  # state, rtg, action, vision1, vision2
        input_seq_len = self.num_inputs * context_len
        blocks = [
            Block(
                h_dim, 
                input_seq_len, 
                n_heads, 
                drop_p,
                self.num_inputs,
                mgdt=False,
                dt_mask=dt_mask
            ) 
            for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_action = nn.Linear(act_dim, h_dim)
        self.embed_vision1 = nn.Linear(vision1_dim, h_dim) if vision1_dim > 0 else None
        self.embed_vision2 = nn.Linear(vision2_dim, h_dim) if vision2_dim > 0 else None
        
        ### prediction heads
        self.predict_rtg = nn.Linear(h_dim, 1)
        # stochastic action
        self.predict_action = BaseActor(h_dim, self.act_dim)
        self.predict_state = nn.Linear(h_dim, state_dim)

        # For entropy
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32)
        )
        self.target_entropy = target_entropy

    def temperature(self):
        return self.log_temperature.exp()
    
    @torch.no_grad()
    def _infer_nvision(self):
        return self.n_vision

    def forward(
        self, 
        timesteps, 
        states, 
        actions, 
        returns_to_go,
        vision1 = None,
        vision2 = None
    ):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)  # (B, T, h)
        state_embeddings = self.embed_state(states) + time_embeddings
        rtg_embeddings   = self.embed_rtg(returns_to_go) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        tokens = [state_embeddings]

        # 视觉 token（按顺序插在 state 之后、rtg 之前）
        if self.embed_vision1 is not None and vision1 is not None and vision1.shape[-1] > 0:
            v1_emb = self.embed_vision1(vision1) + time_embeddings
            tokens.append(v1_emb)
        if self.embed_vision2 is not None and vision2 is not None and vision2.shape[-1] > 0:
            v2_emb = self.embed_vision2(vision2) + time_embeddings
            tokens.append(v2_emb)

        tokens += [rtg_embeddings, action_embeddings]  # 末尾依次是 rtg, action

        # 堆叠成 (B, num_inputs*T, h_dim)
        h = torch.stack(tokens, dim=1)              # (B, num_inputs, T, h)
        h = h.permute(0, 2, 1, 3).reshape(B, self.num_inputs * T, self.h_dim)
        h = self.embed_ln(h)

        # transformer
        h = self.transformer(h)                     # (B, num_inputs*T, h)
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)  # (B, num_inputs, T, h)

        # 依据 token 布局做预测的索引
        idx_state  = 0
        idx_rtg    = 1 + self.n_vision
        idx_action = idx_rtg + 1

        # predictions
        rtg_preds         = self.predict_rtg(h[:, idx_state])   # given s
        action_dist_preds = self.predict_action(h[:, idx_rtg])  # given s,(v1,v2),R
        state_preds       = self.predict_state(h[:, idx_action]) # given s,(v1,v2),R,a

        return rtg_preds, action_dist_preds, state_preds
