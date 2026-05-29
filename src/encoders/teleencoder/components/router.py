"""
Router modules for TeleEncoder MoE layers.

NaiveRouter  (ablation): per-sample routing bias via Pool(context_tokens) → Linear
DynamicRouter (default): per-token  routing bias via CrossAttn(tokens, context_tokens)

Both return ctx_bias [B, L, M] to be added to token_logits with a learnable weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveRouter(nn.Module):
    """Per-sample context routing bias (same bias for all tokens in a sample).

    Pool(context_tokens) → Linear → [B, M] → broadcast to [B, L, M]

    Ablation baseline — no per-token differentiation.
    """

    def __init__(self, d_model: int, num_experts: int, init_std: float = 0.01):
        super().__init__()
        self.W = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.W.weight, std=init_std)

    def forward(self, tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens:         [B, L, d_model]   (used for shape/device only)
        context_tokens: [B, 2K, d_model]
        Returns:        ctx_bias [B, L, M]
        """
        L = tokens.shape[1]
        ctx_pooled = context_tokens.mean(dim=1)                     # [B, d_model]
        bias = self.W(ctx_pooled)                                   # [B, M]
        return bias.unsqueeze(1).expand(-1, L, -1)                  # [B, L, M]


class DynamicRouter(nn.Module):
    """Per-token context routing bias via cross-attention.

    q = W_q(tokens)   [B, L, d_r]
    k = W_k(ctx)      [B, 2K, d_r]
    v = W_v(ctx)      [B, 2K, M]   ← value projects to expert space

    ctx_bias = softmax(q k^T / √d_r) @ v   [B, L, M]

    Each token attends to context tokens and aggregates their expert-space
    contributions, yielding a unique routing bias per token.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_r: int = 32,
        init_std: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_r = d_r
        self.W_q = nn.Linear(d_model, d_r, bias=False)
        self.W_k = nn.Linear(d_model, d_r, bias=False)
        self.W_v = nn.Linear(d_model, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.W_q.weight, std=init_std)
        nn.init.normal_(self.W_k.weight, std=init_std)
        nn.init.normal_(self.W_v.weight, std=init_std)

    def forward(self, tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens:         [B, L, d_model]
        context_tokens: [B, 2K, d_model]
        Returns:        ctx_bias [B, L, M]
        """
        q = self.W_q(tokens)           # [B, L, d_r]
        k = self.W_k(context_tokens)   # [B, 2K, d_r]
        v = self.W_v(context_tokens)   # [B, 2K, M]

        scale = self.d_r ** -0.5
        attn  = torch.bmm(q, k.transpose(1, 2)) * scale   # [B, L, 2K]
        attn  = self.dropout(F.softmax(attn, dim=-1))

        return torch.bmm(attn, v)                          # [B, L, M]
