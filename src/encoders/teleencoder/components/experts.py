"""
Expert networks and MoE layer for TeleEncoder.

ExpertNetwork: FFN with EiLM (Expert-informed Linear Modulation) via cross-attention
               to context tokens. Each expert is independently conditioned.
MoELayer:      Top-K sparse routing + expert dispatch + auxiliary loss bookkeeping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ExpertNetwork(nn.Module):
    """FFN expert with MoME-style EiLM (Expert-independent Linear Modulation).

    Matches MoME Eq. 4 exactly, with identity initialization following DiT/ControlNet
    best practice:

        z     = Pool(context_tokens)    [B, d]  — same vector for all patches
        γ_i   = w_i^T · z              [B, 1]  — scalar per expert per sample
        β_i   = W_i · z                [B, d]  — vector per expert per sample
        output = (1 + γ_i) · FFN(x) + β_i

    The (1 + γ) formulation with zero-initialized w_i, W_i gives perfect identity
    at init (γ=0, β=0 → output=FFN(x)) regardless of what context_tokens contain.
    This is critical here because our context tokens are learned from scratch
    (unlike MoME's pretrained LLM context), so z ≈ random at init.

    Contrast with our previous implementation which used per-token cross-attention
    (4× more parameters, wrong sigmoid init giving γ=0.5 not 1.0).

    Parameters per expert: d (w_gamma) + d² (W_beta) ≈ 16.5K vs ~65K previously.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # FFN
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # EiLM: per-expert linear maps from pooled context z [d]
        # w_gamma: dot product → scalar γ_i  (MoME: w_i ∈ R^d)
        # W_beta:  linear map  → vector β_i  (MoME: W_i ∈ R^(d×d))
        self.w_gamma = nn.Linear(d_model, 1, bias=False)
        self.W_beta  = nn.Linear(d_model, d_model, bias=False)

        # Zero init → identity at initialization (DiT / ControlNet practice)
        nn.init.zeros_(self.w_gamma.weight)
        nn.init.zeros_(self.W_beta.weight)

    def forward(
        self,
        x: torch.Tensor,                     # [B, L, d_model]
        z_tokens: Optional[torch.Tensor],    # [B, L, d_model] per-token context, or None
    ) -> torch.Tensor:
        # FFN path
        h = self.dropout(F.gelu(self.fc1(x)))
        h = self.fc2(h)                                        # [B, L, d_model]

        if z_tokens is None:
            return h

        # MoME EiLM: per-token modulation — each patch gets its channel's scale token
        gamma = self.w_gamma(z_tokens)          # [B, L, 1]
        beta  = self.W_beta(z_tokens)           # [B, L, d]

        # (1 + γ) · h + β  — identity when γ=0, β=0
        return (1.0 + gamma) * h + beta


class MoELayer(nn.Module):
    """Mixture of Experts with top-K sparse routing.

    routing logits = token_logits + ctx_weight · ctx_bias
    where ctx_bias comes from NaiveRouter or DynamicRouter (set via self.router).

    Expert outputs are weighted-summed using sparse top-K gate values.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        ctx_weight: float = 0.4,
        router_temperature: float = 1.0,
        sparse_inference: bool = True,
        router_init_std: float = 0.01,
        expert_init_gain: float = 0.3,
        router_jitter_std: float = 0.0,
        dropout: float = 0.1,
        eilm_d_r: int = 64,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.ctx_weight = ctx_weight
        self.router_temperature = router_temperature
        self.sparse_inference = sparse_inference
        self.router_jitter_std = router_jitter_std

        # Token routing gate (W_gate · x → logits per expert)
        self.W_gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.W_gate.weight, std=router_init_std)

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        for expert in self.experts:
            nn.init.xavier_uniform_(expert.fc2.weight, gain=expert_init_gain)

        # EiLM context attention: scale token attends over all C+K context tokens
        # to learn which tokens (scale + pattern) are relevant for each channel.
        self.eilm_q = nn.Linear(d_model, eilm_d_r, bias=False)
        self.eilm_k = nn.Linear(d_model, eilm_d_r, bias=False)
        self.eilm_v = nn.Linear(d_model, d_model,  bias=False)
        self._eilm_d_r = eilm_d_r
        # Zero-init v projection so attention adds nothing at init (identity preserved)
        nn.init.zeros_(self.eilm_v.weight)

        # Context router — assigned externally by TeleEncoderBlock
        self.router: Optional[nn.Module] = None

    def forward(
        self,
        x: torch.Tensor,                        # [B, L, d_model]
        context_tokens: Optional[torch.Tensor], # [B, C+K, d_model]
        num_channels: int = 1,
        num_patches: int = 1,
        return_gate_stats: bool = False,
        force_dense: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            num_channels: C — number of KPI channels (first C tokens of context are
                          channel-aligned scale tokens Z_s^(c)).
            num_patches:  S — patches per channel. L must equal C * S.
            force_dense:  if True, route uniformly to all experts (warmup phase).
                          EiLM modulation remains active.
        Returns:
            output:     [B, L, d_model]
            gate_stats: (full_gates [B,L,M], router_logits [B,L,M]) or None
        """
        B, L, d = x.shape
        M = self.num_experts

        # Per-token EiLM context: each channel's scale token attends over all C+K
        # context tokens (scale + pattern) to learn which to use for modulation.
        if context_tokens is not None:
            scale_tokens = context_tokens[:, :num_channels, :]              # [B, C, d]
            Q = self.eilm_q(scale_tokens)                                   # [B, C, d_r]
            K = self.eilm_k(context_tokens)                                 # [B, C+K, d_r]
            V = self.eilm_v(context_tokens)                                 # [B, C+K, d]
            attn = torch.softmax(
                Q @ K.transpose(1, 2) / self._eilm_d_r ** 0.5, dim=-1
            )                                                               # [B, C, C+K]
            z_channels = scale_tokens + attn @ V                           # [B, C, d]
            z_tokens = (z_channels
                        .unsqueeze(2)
                        .expand(-1, -1, num_patches, -1)
                        .reshape(B, L, d))                                  # [B, L, d]
        else:
            z_tokens = None

        # Token routing logits
        token_logits = self.W_gate(x)                          # [B, L, M]

        # Context routing bias (skip during dense warmup)
        if not force_dense and self.router is not None and context_tokens is not None:
            ctx_bias      = self.router(x, context_tokens)     # [B, L, M]
            router_logits = token_logits + self.ctx_weight * ctx_bias
        else:
            router_logits = token_logits

        # Training exploration jitter
        if self.training and self.router_jitter_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_std

        if force_dense:
            # Uniform equal weight to all experts — no top-k selection
            gate_weight_all = torch.full((B, L, M), 1.0 / M, device=x.device, dtype=x.dtype)
            output = torch.zeros_like(x)
            for e in range(M):
                expert_out = self.experts[e](x, z_tokens)
                output += gate_weight_all[..., e:e+1] * expert_out

            gate_stats = None
            if return_gate_stats:
                gate_stats = (gate_weight_all, router_logits)
            return output, gate_stats

        # Temperature scaling + top-K gating
        scaled_logits = router_logits / self.router_temperature
        topk_vals, topk_idx = torch.topk(scaled_logits, self.top_k, dim=-1)  # [B, L, top_k]
        topk_gates = F.softmax(topk_vals, dim=-1)                             # [B, L, top_k]

        # Expert computation and weighted aggregation
        output = torch.zeros_like(x)

        for e in range(M):
            gate_weight = torch.zeros(B, L, device=x.device, dtype=x.dtype)
            for k_slot in range(self.top_k):
                gate_weight += (topk_idx[..., k_slot] == e).float() * topk_gates[..., k_slot]

            if self.sparse_inference and not gate_weight.any():
                continue

            expert_out = self.experts[e](x, z_tokens)
            output += gate_weight.unsqueeze(-1) * expert_out

        gate_stats = None
        if return_gate_stats:
            full_gates = torch.zeros(B, L, M, device=x.device, dtype=x.dtype)
            full_gates.scatter_(-1, topk_idx, topk_gates)
            gate_stats = (full_gates, router_logits)

        return output, gate_stats
