"""
TeleEncoder: Context-Aware MoE Transformer for 5G KPI Time-Series.

Architecture:
  1. Input normalization: zscore per channel (preserves x_raw for scale context)
  2. Patch embedding: Linear(patch_size, d_model) applied to z-scored patches
  3. Context generation (two tracks, from x_raw + normalized embeddings):
       Scale track:   CrossChannelScaleGraph(x_raw) → EGNN → QueryPool → scale_tokens [B,K,d]
       Pattern track: h_channels → CCM → pattern_tokens [B,K,d]
       context_tokens = cat([scale_tokens, pattern_tokens])  [B, 2K, d]
  4. Transformer blocks: MHSA(hybrid_channel_patch) + MoELayer(DynamicRouter/NaiveRouter)
  5. Output: [B, C*S, d] → reshape → mean over S → [B, C, d]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .components.context import ContextGenerator
from .components.blocks import TeleEncoderBlock


class TeleEncoder(nn.Module):
    """Main TeleEncoder model.

    Args:
        num_channels:            Number of KPI channels (C = 18).
        seq_len:                 Time series length (L = 128).
        d_model:                 Embedding dimension.
        num_heads:               Attention heads.
        num_layers:              Number of TeleEncoder blocks.
        d_ff:                    FFN hidden dimension in experts.
        num_experts:             Number of MoE experts (M).
        top_k:                   Experts selected per token.
        K:                       Context prototypes per track (total 2K context tokens).
        patch_size:              Patch size (must equal patch_stride for non-overlapping).
        patch_stride:            Patch stride.
        dropout:                 Dropout rate.
        router_type:             "dynamic" (default) or "naive" (ablation).
        ctx_weight:              Context routing bias weight.
        router_temperature:      Routing temperature.
        attn_mode:               "hybrid_channel_patch" or "standard".
        scale_encoder:           "graph" (default) or "nme" (ablation for scale track).
        d_node:                  GNN node embedding dimension.
        n_gnn_layers:            Number of EGNN layers.
        gnn_alpha_init:          Initial Pearson weight in edge scoring.
        gnn_tau_init:            Initial bandwidth for RBF edge term.
        sparse_inference:        Skip zero-weight experts during forward.
        router_init_std:         Router weight init std.
        expert_init_gain:        Expert fc2 xavier gain (stability).
        router_jitter_std:       Training jitter on router logits.
        dense_routing_warmup_epochs: Epochs of dense routing before sparse kicks in.
    """

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        K: int = 4,
        patch_size: int = 32,
        patch_stride: int = 32,
        dropout: float = 0.1,
        router_type: str = "dynamic",
        ctx_weight: float = 0.4,
        router_temperature: float = 1.0,
        attn_mode: str = "standard",
        scale_encoder: str = "graph",
        d_node: int = 32,
        n_gnn_layers: int = 2,
        gnn_alpha_init: float = 0.5,
        gnn_tau_init: float = 0.3,
        sparse_inference: bool = True,
        router_init_std: float = 0.01,
        expert_init_gain: float = 0.3,
        router_jitter_std: float = 0.0,
        dense_routing_warmup_epochs: int = 3,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dense_routing_warmup_epochs = dense_routing_warmup_epochs
        self.current_epoch = 0
        # Set to True from outside when DeepSpeed ZeRO3 is active:
        # sparse routing causes NCCL deadlocks during
        # eval because different ranks call different experts → mismatched allgathers.
        self.force_dense_eval: bool = False

        self.num_patches = 1 + (seq_len - patch_size) // patch_stride
        if self.num_patches <= 0:
            raise ValueError(
                f"Invalid patching: seq_len={seq_len}, patch_size={patch_size}, "
                f"patch_stride={patch_stride}"
            )

        # Input embedding: each channel patch → d_model
        self.input_embedding = nn.Linear(patch_size, d_model)

        # Learnable positional encodings
        self.channel_pos_enc = nn.Parameter(torch.empty(1, num_channels, 1, d_model))
        self.patch_pos_enc   = nn.Parameter(torch.empty(1, 1, self.num_patches, d_model))
        nn.init.xavier_uniform_(self.channel_pos_enc)
        nn.init.xavier_uniform_(self.patch_pos_enc)

        self.dropout = nn.Dropout(dropout)

        # Context generator (two-track: scale + pattern)
        self.context_generator = ContextGenerator(
            K=K,
            d_model=d_model,
            seq_len=seq_len,
            d_node=d_node,
            n_gnn_layers=n_gnn_layers,
            gnn_alpha_init=gnn_alpha_init,
            gnn_tau_init=gnn_tau_init,
            scale_encoder=scale_encoder,
            dropout=dropout,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TeleEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                router_type=router_type,
                ctx_weight=ctx_weight,
                router_temperature=router_temperature,
                sparse_inference=sparse_inference,
                router_init_std=router_init_std,
                expert_init_gain=expert_init_gain,
                router_jitter_std=router_jitter_std,
                attn_mode=attn_mode,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Debug side-channels (populated during forward, read by main.py diagnostics)
        self._debug_gate_stats: list = []

        # Set to True (externally) to disable context generation entirely
        self.no_context: bool = False

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel z-score normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def compute_moe_balance_loss(self, gate_stats_list: list) -> torch.Tensor:
        """Switch Transformer auxiliary balance loss.

        L_balance = M · Σ_i f_i · P_i
        where f_i  = dispatch fraction (no grad, from sparse top-k gates)
              P_i  = mean full-softmax probability (has grad)
        """
        if not gate_stats_list:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total = 0.0
        for gate_tuple in gate_stats_list:
            if gate_tuple is None:
                continue
            gates, router_logits = gate_tuple
            B, L, M = gates.shape
            with torch.no_grad():
                dispatch = gates.mean(dim=[0, 1])           # [M], no grad
            full_probs = F.softmax(router_logits, dim=-1).mean(dim=[0, 1])  # [M]
            total += M * (dispatch * full_probs).sum()

        return total / max(len(gate_stats_list), 1)

    def compute_similarity_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """RBF channel similarity for cluster loss (TimeSeriesCCM)."""
        B, C, L = x.shape
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std  = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        x_n    = (x - x_mean) / x_std                      # [B, C, L]
        x_i    = x_n.unsqueeze(2)                           # [B, C, 1, L]
        x_j    = x_n.unsqueeze(1)                           # [B, 1, C, L]
        dist_sq = ((x_i - x_j) ** 2).sum(dim=-1)           # [B, C, C]
        return torch.exp(-dist_sq / (2 * 5.0 ** 2))        # sigma = 5

    def compute_cluster_loss(
        self,
        membership: torch.Tensor,      # [B, C, K]
        similarity: torch.Tensor,      # [B, C, C]
    ) -> torch.Tensor:
        """TimeSeriesCCM cluster loss (Eq. 4):
        L = -Tr(M^T S M) + Tr((I - M M^T) S)
        """
        B, C, K = membership.shape
        MtSM  = torch.bmm(torch.bmm(membership.transpose(1, 2), similarity), membership)  # [B, K, K]
        MMt   = torch.bmm(membership, membership.transpose(1, 2))                          # [B, C, C]
        I_MMt = torch.eye(C, device=similarity.device).unsqueeze(0) - MMt
        loss  = (-MtSM.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                 + (I_MMt * similarity).sum(dim=[-1, -2]))
        return loss.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                    # [B, C, L] unit-scaled input
        return_moe_loss: bool = False,
        compute_cluster_loss: bool = False,
        return_router_diagnostics: bool = False,  # kept for API compatibility
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Returns:
            encoded:   [B, C, S, d_model]  — all patch representations, no pooling
            aux_losses: (balance_loss, cluster_loss) scalars, or None
        """
        B, C, L = x.shape

        # Keep raw for scale-aware context (graph uses absolute KPI magnitudes)
        x_raw  = x
        x_norm = self._zscore(x)                             # [B, C, L]

        # Patchify normalized input
        x_patches = x_norm.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        # x_patches: [B, C, S, P]

        # Patch embedding + positional encoding
        h_patch = self.input_embedding(x_patches)            # [B, C, S, d_model]
        h_patch = h_patch + self.channel_pos_enc + self.patch_pos_enc
        h_patch = self.dropout(h_patch)

        # Two-track context generation
        if self.no_context:
            context_tokens, cluster_assignments = None, None
        else:
            context_tokens, cluster_assignments = self.context_generator(
                x_norm, x_raw
            )  # [B, C+K, d_model], [B, C, K]

        # Flatten to token sequence: [B, C*S, d_model]
        S = self.num_patches
        x_tokens = h_patch.reshape(B, C * S, self.d_model)

        # Transformer blocks
        gate_stats_list = []
        force_dense = self.force_dense_eval or (self.training and self.current_epoch < self.dense_routing_warmup_epochs)
        for block in self.blocks:
            x_tokens, gate_stats = block(
                x_tokens,
                context_tokens=context_tokens,
                num_channels=C,
                num_patches=S,
                return_gate_stats=return_moe_loss,
                force_dense=force_dense,
            )
            if return_moe_loss and gate_stats is not None:
                gate_stats_list.append(gate_stats)

        self._debug_gate_stats = gate_stats_list

        # Final norm + reshape: [B, C*S, d] → [B, C, S, d]
        # All patch representations are preserved — no pooling loss.
        # The head (outside the encoder) sees every (channel, patch) representation
        # and can learn which patches/channels are discriminative per class.
        encoded = self.norm(x_tokens).reshape(B, C, S, self.d_model)  # [B, C, S, d_model]

        if return_moe_loss:
            balance_loss = self.compute_moe_balance_loss(gate_stats_list)

            if compute_cluster_loss and cluster_assignments is not None:
                similarity = self.compute_similarity_matrix(x_norm)
                cluster_loss = self.compute_cluster_loss(cluster_assignments, similarity)
            else:
                cluster_loss = torch.tensor(0.0, device=encoded.device)

            return encoded, (balance_loss, cluster_loss)

        return encoded, None
