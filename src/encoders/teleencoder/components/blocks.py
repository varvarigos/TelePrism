"""
TeleEncoder transformer block: MHSA + MoE FFN.

Attention mode:
  "hybrid_channel_patch": channel-level cross-channel attention, broadcast to patches.
                          Efficient O(C²) attention rather than O((C·S)²) full-sequence.
  "standard":             standard full-sequence self-attention over all C·S tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .experts import MoELayer
from .router import NaiveRouter, DynamicRouter


class TeleEncoderBlock(nn.Module):
    """Single TeleEncoder block: MHSA + MoELayer.

    Args:
        router_type: "dynamic" (per-token cross-attn bias) or "naive" (pooled bias, ablation).
        attn_mode:   "hybrid_channel_patch" or "standard".
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        router_type: str = "dynamic",
        ctx_weight: float = 0.4,
        router_temperature: float = 1.0,
        sparse_inference: bool = True,
        router_init_std: float = 0.01,
        expert_init_gain: float = 0.3,
        router_jitter_std: float = 0.0,
        attn_mode: str = "standard",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_mode = attn_mode
        self.d_model = d_model

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # MoE FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            ctx_weight=ctx_weight,
            router_temperature=router_temperature,
            sparse_inference=sparse_inference,
            router_init_std=router_init_std,
            expert_init_gain=expert_init_gain,
            router_jitter_std=router_jitter_std,
            dropout=dropout,
        )

        # Context router — wired into MoE
        if router_type == "dynamic":
            router = DynamicRouter(
                d_model=d_model, num_experts=num_experts,
                d_r=max(32, d_model // 4), init_std=router_init_std,
            )
        elif router_type == "naive":
            router = NaiveRouter(
                d_model=d_model, num_experts=num_experts, init_std=router_init_std,
            )
        else:
            raise ValueError(f"Unknown router_type: {router_type!r}. Use 'dynamic' or 'naive'.")
        self.moe.router = router

    def _channel_attn(
        self,
        x: torch.Tensor,
        num_channels: int,
        num_patches: int,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cross-channel attention at channel level, broadcast back to patches."""
        B, L, d = x.shape
        x_r = x.reshape(B, num_channels, num_patches, d)
        ch_tokens = x_r.mean(dim=2)                             # [B, C, d]
        ch_norm = self.norm1(ch_tokens)
        attn_out, _ = self.self_attn(ch_norm, ch_norm, ch_norm, attn_mask=attn_mask)
        ch_updated = ch_tokens + self.dropout(attn_out)         # [B, C, d]
        ch_delta = (ch_updated - ch_tokens).unsqueeze(2)        # [B, C, 1, d]
        x = x + ch_delta.expand(-1, -1, num_patches, -1).reshape(B, L, d)
        return x

    def _standard_attn(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        return x + self.dropout(attn_out)

    def forward(
        self,
        x: torch.Tensor,                        # [B, L, d_model]  L = C * S
        context_tokens: Optional[torch.Tensor], # [B, 2K, d_model]
        num_channels: int,
        num_patches: int,
        attn_mask: Optional[torch.Tensor] = None,
        return_gate_stats: bool = False,
        force_dense: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Returns:
            x:          [B, L, d_model]
            gate_stats: (full_gates, router_logits) or None
        """
        # MHSA
        if self.attn_mode == "hybrid_channel_patch":
            x = self._channel_attn(x, num_channels, num_patches, attn_mask)
        else:
            x = self._standard_attn(x, attn_mask)

        # MoE FFN
        x_norm = self.norm2(x)
        moe_out, gate_stats = self.moe(
            x_norm, context_tokens,
            num_channels=num_channels,
            num_patches=num_patches,
            return_gate_stats=return_gate_stats,
            force_dense=force_dense,
        )
        x = x + self.dropout(moe_out)

        return x, gate_stats
