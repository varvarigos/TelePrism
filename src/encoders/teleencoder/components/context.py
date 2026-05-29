"""
Two-track context generation for TeleEncoder.

Scale track:   CrossChannelScaleGraph(x_raw) → EGNN → [B,C,d_node]
               → Linear+LN → scale_tokens [B,C,d_model]  (one token per channel)
Pattern track: x_norm [B,C,L] → CCM → pattern_tokens [B,K,d_model]
Assembly:      context_tokens = cat([scale_tokens, pattern_tokens], dim=1)  [B, C+K, d_model]

The scale track preserves absolute KPI magnitude (lost after z-scoring).
The pattern track encodes full temporal channel patterns via learned cluster prototypes
(masked cross-attention, Eq. 2-3 of the CCM paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Scale track
# ---------------------------------------------------------------------------

class EdgeConditionedGNN(nn.Module):
    """Single EGNN message-passing layer.

    h_i' = ReLU(LayerNorm(h_i + Σ_j s_ij · MLP([h_i ‖ h_j ‖ e_ij])))
    """

    def __init__(self, d_node: int, d_edge: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * d_node + d_edge, d_node),
            nn.GELU(),
            nn.Linear(d_node, d_node),
        )
        self.norm = nn.LayerNorm(d_node)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,   # [B, C, d_node]
        s: torch.Tensor,   # [B, C, C] edge weights
        e: torch.Tensor,   # [B, C, C, d_edge] edge features
    ) -> torch.Tensor:
        B, C, d = h.shape
        hi = h.unsqueeze(2).expand(-1, -1, C, -1)  # [B, C, C, d]
        hj = h.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, C, d]
        msgs = self.msg_mlp(torch.cat([hi, hj, e], dim=-1))  # [B, C, C, d]
        agg = (s.unsqueeze(-1) * msgs).sum(dim=2)             # [B, C, d]
        return F.relu(self.norm(h + self.dropout(agg)))


class CrossChannelScaleGraph(nn.Module):
    """Scale-aware dynamic graph over KPI channels.

    Node features:  [log10|μ|, log10σ, CV]            — absolute scale info
    Edge features:  [log(σ_i/σ_j), log(|μ_i|/|μ_j|)] — relative scale ratios
    Edge weights:   α|ρ_ij| + (1-α)exp(-‖e_ij‖/τ)

    Returns graph_channel_embeds [B, C, d_node].
    """

    def __init__(
        self,
        d_node: int = 32,
        n_layers: int = 2,
        alpha_init: float = 0.5,
        tau_init: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_node = d_node
        d_edge = 2
        self.node_proj = nn.Linear(3, d_node)
        self.gnn_layers = nn.ModuleList([
            EdgeConditionedGNN(d_node, d_edge, dropout) for _ in range(n_layers)
        ])
        self.gnn_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gnn_tau = nn.Parameter(torch.tensor(tau_init))

    def _channel_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return |μ|, σ per channel, clamped for log-safety."""
        mu  = x.mean(dim=-1).abs().clamp(min=1e-8)             # [B, C]
        sig = x.std(dim=-1, unbiased=False).clamp(min=1e-8)    # [B, C]
        return mu, sig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] raw time series → [B, C, d_node]"""
        B, C, L = x.shape
        mu, sig = self._channel_stats(x)

        # Node features: [log10|μ|, log10σ, CV]
        log_mu  = torch.log10(mu)
        log_sig = torch.log10(sig)
        cv      = (sig / mu).clamp(0.0, 100.0)
        node_feats = torch.stack([log_mu, log_sig, cv], dim=-1)  # [B, C, 3]
        h = self.node_proj(node_feats.to(x.dtype))               # [B, C, d_node]

        # Edge features: scale-ratio vectors [B, C, C, 2]
        log_sig_ratio = log_sig.unsqueeze(2) - log_sig.unsqueeze(1)  # [B, C, C]
        log_mu_ratio  = log_mu.unsqueeze(2)  - log_mu.unsqueeze(1)   # [B, C, C]
        e = torch.stack([log_sig_ratio, log_mu_ratio], dim=-1)        # [B, C, C, 2]

        # Pearson |ρ_ij| from x_raw
        x_f    = x.float()
        mean_f = x_f.mean(dim=-1)
        std_f  = x_f.std(dim=-1, unbiased=False).clamp(min=1e-8)
        x_n    = (x_f - mean_f.unsqueeze(-1)) / std_f.unsqueeze(-1)
        rho    = torch.bmm(x_n, x_n.transpose(1, 2)) / L  # [B, C, C]
        rho    = rho.abs()

        # Combined edge weights
        alpha = self.gnn_alpha.sigmoid()
        tau   = self.gnn_tau.clamp(min=0.01)
        s = alpha * rho + (1.0 - alpha) * torch.exp(-e.norm(dim=-1) / tau)  # [B, C, C]
        s = s.to(x.dtype)
        e = e.to(x.dtype)

        for layer in self.gnn_layers:
            h = layer(h, s, e)

        return h  # [B, C, d_node]


class NMEScaleEncoder(nn.Module):
    """Ablation: Normalized Magnitude Encoder (O(C), no graph).

    Concatenates absolute log-stats with cross-channel z-score normalized stats
    → per-channel embedding [B, C, d_node].
    """

    def __init__(self, d_node: int = 32, dropout: float = 0.1):
        super().__init__()
        # 6 features: [log|μ|, logσ, CV] (abs) + same 3 cross-channel z-scored
        self.proj = nn.Linear(6, d_node)
        self.norm = nn.LayerNorm(d_node)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _znorm(t: torch.Tensor) -> torch.Tensor:
        m = t.mean(dim=1, keepdim=True)
        s = t.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (t - m) / s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] → [B, C, d_node]"""
        mu  = x.mean(dim=-1).abs().clamp(min=1e-8)
        sig = x.std(dim=-1, unbiased=False).clamp(min=1e-8)
        cv  = (sig / mu).clamp(0.0, 100.0)

        log_mu  = torch.log10(mu)
        log_sig = torch.log10(sig)

        feats = torch.stack([
            log_mu, log_sig, cv,
            self._znorm(log_mu), self._znorm(log_sig), self._znorm(cv),
        ], dim=-1)  # [B, C, 6]

        h = self.proj(feats.to(x.dtype))
        return self.dropout(self.norm(h))  # [B, C, d_node]


class QueryPool(nn.Module):
    """K learned queries cross-attend over C channel embeddings.

    graph_channel_embeds [B, C, d_in] → scale_tokens [B, K, d_out]
    """

    def __init__(self, K: int, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.scale = d_out ** -0.5
        self.queries = nn.Parameter(torch.empty(K, d_out))
        nn.init.xavier_uniform_(self.queries.unsqueeze(0))
        self.kv_proj = nn.Linear(d_in, d_out)
        self.out_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, C, d_in] → [B, K, d_out]"""
        B = h.shape[0]
        kv = self.kv_proj(h)                                    # [B, C, d_out]
        q  = self.queries.unsqueeze(0).expand(B, -1, -1)        # [B, K, d_out]
        attn = torch.bmm(q, kv.transpose(1, 2)) * self.scale   # [B, K, C]
        attn = self.dropout(F.softmax(attn, dim=-1))
        out  = torch.bmm(attn, kv)                              # [B, K, d_out]
        return self.out_norm(q + out)                           # residual + norm


# ---------------------------------------------------------------------------
# Pattern track
# ---------------------------------------------------------------------------

class CCM(nn.Module):
    """Cross-Channel Matching with full-sequence channel encoder and masked cross-attention.

    Implements the three-step procedure from the CCM paper (Eq. 2-3):

    Step 1 — Channel encoding (paper: "Xi transformed through an MLP"):
        h_i = MLP(X_i)    X_i ∈ R^L → h_i ∈ R^d
        Uses the full time series, NOT a mean over patch embeddings.
        This preserves temporal anomaly shapes (exponential, sinusoidal, step)
        that mean-pooling destroys.

    Step 2 — Clustering probabilities (Eq. 2):
        p_ik  = softmax(cosine(c_k, h_i) / τ)    [B, C, K]
        M_ik ≈ Bernoulli(p_ik) via Gumbel-sigmoid (train) / hard threshold (eval)

    Step 3 — Masked cross-attention prototype update (Eq. 3):
        Cb = Normalize(exp(WQ·C · (WK·H)^T / √d_r) ⊙ M^T) · WV·H
        ≡  softmax(QK^T/√d_r + log(M^T)) · V
        With approximately-binary M, each prototype attends almost exclusively
        to its assigned channels — out-of-cluster interference is suppressed.
    """

    # Temperature for Gumbel-sigmoid relaxation of Bernoulli(p).
    # Lower = harder (closer to binary); 0.1 is sharp but still differentiable.
    BERNOULLI_TAU = 0.1

    def __init__(self, K: int, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        d_r = max(32, d_model // 4)

        # Step 1: encode full time series per channel  Xi [L] → hi [d_model]
        # Bottleneck (L → d_model//2 → d_model) + dropout regularises the large
        # input projection and prevents the encoder from memorising training patterns.
        d_enc = d_model // 2
        self.channel_encoder = nn.Sequential(
            nn.Linear(seq_len, d_enc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_enc, d_model),
            nn.LayerNorm(d_model),
        )

        # Cluster prototype embeddings C [K, d_model]
        self.prototypes = nn.Parameter(torch.empty(K, d_model))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0))
        self.cluster_log_tau = nn.Parameter(torch.zeros(1))  # τ=1.0 at init

        # Step 3: masked cross-attention  (Q from prototypes, K/V from channel embeds)
        self.W_q = nn.Linear(d_model, d_r, bias=False)
        self.W_k = nn.Linear(d_model, d_r, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.d_r = d_r
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _bernoulli_M(self, p: torch.Tensor) -> torch.Tensor:
        """Reparameterized approximately-binary membership matrix.

        Train: Gumbel-sigmoid relaxation — differentiable, values ∈ (0,1) but
               strongly peaked near 0/1 at BERNOULLI_TAU=0.1.
        Eval:  hard threshold at 0.5 — exactly binary, no gradients needed.
        """
        if self.training:
            logit = torch.log(p.clamp(1e-6, 1 - 1e-6)) \
                  - torch.log((1 - p).clamp(1e-6, 1 - 1e-6))
            noise = -torch.log(
                -torch.log(torch.rand_like(p).clamp(1e-8, 1 - 1e-8))
            )
            return torch.sigmoid((logit + noise) / self.BERNOULLI_TAU)
        else:
            return (p > 0.5).to(p.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, L] — full z-scored time series (scale-blind, shape-preserved)
        Returns:
            pattern_tokens:       [B, K, d_model]
            cluster_assignments:  [B, C, K]   (M, approximately binary)
        """
        B, C, L = x.shape

        # Step 1: channel embeddings from full time series
        h = self.channel_encoder(x.reshape(B * C, L)).reshape(B, C, -1)  # [B, C, d_model]

        # Step 2: clustering probabilities → approximately-binary M
        tau = torch.exp(self.cluster_log_tau).clamp(min=1e-3)
        h_n = F.normalize(h, p=2, dim=-1)                          # [B, C, d]
        p_n = F.normalize(self.prototypes, p=2, dim=-1)             # [K, d]
        sim = torch.matmul(h_n, p_n.T)                             # [B, C, K]
        p   = F.softmax(sim / tau, dim=-1)                         # [B, C, K] soft probs
        M   = self._bernoulli_M(p)                                 # [B, C, K] ≈ binary

        # Step 3: masked cross-attention prototype update (Eq. 3)
        proto = self.prototypes.unsqueeze(0).expand(B, -1, -1)     # [B, K, d_model]
        Q     = self.W_q(proto)                                     # [B, K, d_r]
        K_mat = self.W_k(h)                                         # [B, C, d_r]
        V     = self.W_v(h)                                         # [B, C, d_model]

        attn   = torch.bmm(Q, K_mat.transpose(1, 2)) * (self.d_r ** -0.5)  # [B, K, C]
        M_t    = M.transpose(1, 2)                                          # [B, K, C]
        # Normalize(exp(QK^T/√d) ⊙ M^T) ≡ softmax(QK^T/√d + log(M^T))
        masked_attn = F.softmax(attn + torch.log(M_t.clamp(min=1e-8)), dim=-1)
        masked_attn = self.dropout(masked_attn)

        pattern_tokens = self.out_norm(torch.bmm(masked_attn, V))  # [B, K, d_model]
        return pattern_tokens, M


# ---------------------------------------------------------------------------
# Context generator (two-track assembly)
# ---------------------------------------------------------------------------

class ContextGenerator(nn.Module):
    """Orchestrates two-track context generation.

    Scale track:   CrossChannelScaleGraph(x_raw) → QueryPool → scale_tokens [B,K,d]
    Pattern track: CCM(x_norm) → pattern_tokens [B,K,d]
    Output:        context_tokens [B, 2K, d_model]

    CCM receives the full z-scored time series x_norm so its channel encoder
    sees complete temporal shapes (not mean-pooled patch embeddings).
    """

    def __init__(
        self,
        K: int,
        d_model: int,
        seq_len: int,
        d_node: int = 32,
        n_gnn_layers: int = 2,
        gnn_alpha_init: float = 0.5,
        gnn_tau_init: float = 0.3,
        scale_encoder: str = "graph",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K

        if scale_encoder == "graph":
            self.scale_encoder = CrossChannelScaleGraph(
                d_node=d_node,
                n_layers=n_gnn_layers,
                alpha_init=gnn_alpha_init,
                tau_init=gnn_tau_init,
                dropout=dropout,
            )
        elif scale_encoder == "nme":
            self.scale_encoder = NMEScaleEncoder(d_node=d_node, dropout=dropout)
        else:
            raise ValueError(f"Unknown scale_encoder: {scale_encoder!r}. Use 'graph' or 'nme'.")

        # Direct per-channel projection: preserves channel identity unlike QueryPool.
        # Each scale_token[c] corresponds exclusively to channel c's EGNN embedding,
        # enabling DynamicRouter to learn channel-specific routing biases.
        self.scale_proj = nn.Sequential(
            nn.Linear(d_node, d_model),
            nn.LayerNorm(d_model),
        )
        self.ccm = CCM(K=K, d_model=d_model, seq_len=seq_len, dropout=dropout)

    def forward(
        self,
        x_norm: torch.Tensor,  # [B, C, L] z-scored time series
        x_raw: torch.Tensor,   # [B, C, L] raw time series
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context_tokens:      [B, C+K, d_model]  — C channel scale tokens + K cluster tokens
            cluster_assignments: [B, C, K]
        """
        # Scale track — one token per channel, identity preserved
        graph_embeds = self.scale_encoder(x_raw)              # [B, C, d_node]
        scale_tokens = self.scale_proj(graph_embeds)          # [B, C, d_model]

        # Pattern track — K cluster prototype tokens
        pattern_tokens, assignments = self.ccm(x_norm)        # [B, K, d_model], [B, C, K]

        context_tokens = torch.cat([scale_tokens, pattern_tokens], dim=1)  # [B, C+K, d_model]
        return context_tokens, assignments
