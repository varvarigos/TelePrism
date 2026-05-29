"""
Encoder adapter functions and wrapper classes for the TSLLM pipeline.

All adapters follow the convention that `get_embeddings(x)` accepts
x: [B, C, L] (channels-first) and returns a tensor the AlignLayer can consume.
"""

import torch
from chronos import ChronosPipeline
from transformers import T5Config, T5EncoderModel


# ---------------------------------------------------------------------------
# Toto adapter
# ---------------------------------------------------------------------------

def toto_get_embeddings(
    self, inputs, input_padding_mask, id_mask, kv_cache=None, scaling_prefix_length=None
):
    """Monkey-patch for TotoBackbone.get_embeddings.

    Runs the scaler + patch_embed + transformer and returns the raw token
    sequence (no pooling), consistent with the other encoders.
    """
    scaled_inputs, _, _ = self.scaler(
        inputs,
        weights=torch.ones_like(inputs, device=inputs.device),
        padding_mask=input_padding_mask,
        prefix_length=scaling_prefix_length,
    )
    embeddings, reduced_id_mask = self.patch_embed(scaled_inputs, id_mask)
    transformed = self.transformer(embeddings, reduced_id_mask, kv_cache)
    return transformed


# ---------------------------------------------------------------------------
# Mantis adapter
# ---------------------------------------------------------------------------

def mantis_get_embeddings(self, x):
    """Monkey-patch for Mantis8M / MantisV1.get_embeddings.

    x: [B, C, L]  (raw multi-channel time series)
    Returns: [B, C, num_patches * hidden_dim]
    Processes each channel independently through Mantis.
    """
    B, C, L = x.shape
    x = x.reshape(B * C, 1, L)
    if L != self.seq_len:
        x = torch.nn.functional.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
    emb = self(x)                       # [B*C, num_patches, hidden_dim]
    return emb.reshape(B, C, -1)        # [B, C, num_patches * hidden_dim]


# ---------------------------------------------------------------------------
# TSLib adapter (Autoformer, FEDformer, Informer, NonStationary_Transformer,
#               TimesNet)
# ---------------------------------------------------------------------------

TSLIB_MODELS = frozenset({
    "autoformer", "fedformer", "informer", "nonstationary_transformer", "timesnet"
})


def tslib_get_embeddings(self, x):
    """Monkey-patch for Time-Series-Library encoders.

    x: [B, C, T]  (channels-first, our pipeline convention)
    Returns: [B, T, d_model]  (time-first, TSLib encoder output)
    """
    x_t = x.transpose(1, 2)    # [B, T, C]
    out = self.forward_encoder(x_t)
    if isinstance(out, tuple):
        out = out[0]            # NonStationary_Transformer returns a 4-tuple
    return out                  # [B, T, d_model]


# ---------------------------------------------------------------------------
# Chronos wrapper
# ---------------------------------------------------------------------------

class ChronosEncoder(torch.nn.Module):
    """Wraps a Chronos T5 encoder + tokenizer.

    get_embeddings(x: [B, C, L]) → [B, C, n_patches * d_model]
    Each channel is tokenized independently. The EOS token appended by the
    Chronos tokenizer is dropped, leaving exactly seq_len tokens. These are
    then adaptively pooled to n_patches bins (default 8, matching the patch
    count of TeleEncoder and Toto) and flattened.
    hidden_dim = n_patches * d_model  (e.g. 8 * 192 = 1536).
    """

    def __init__(self, encoder: torch.nn.Module, tokenizer, d_model: int, n_patches: int = 8):
        super().__init__()
        self.encoder   = encoder
        self.tokenizer = tokenizer
        self.d_model   = d_model
        self.n_patches = n_patches
        self.hidden_dim = n_patches * d_model   # e.g. 8 * 192 = 1536

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] → [B, C, n_patches * d_model]"""
        B, C, L = x.shape
        x_flat = x.reshape(B * C, L).cpu()
        token_ids, attn_mask, _ = self.tokenizer.context_input_transform(x_flat)
        token_ids = token_ids[:, :-1].to(x.device)   # drop EOS → [B*C, L]
        attn_mask = attn_mask[:, :-1].to(x.device)
        enc_out = self.encoder(
            input_ids=token_ids, attention_mask=attn_mask
        ).last_hidden_state                           # [B*C, L, d_model]
        # Pool L tokens → n_patches bins (L is exactly divisible: 128/8=16)
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            enc_out.transpose(1, 2), self.n_patches
        ).transpose(1, 2)                             # [B*C, n_patches, d_model]
        return pooled.reshape(B, C, -1)               # [B, C, n_patches * d_model]


def build_chronos_encoder(cfg: dict, dtype: torch.dtype, device) -> "ChronosEncoder":
    """Construct a ChronosEncoder from a config dict (chronos_model section).

    cfg must contain:
      use_pretrained: bool
      scratch: {d_model, d_kv, d_ff, num_heads, num_layers}  (if use_pretrained=False)

    Returns (ts_encoder, embed_dim).
    """
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")
    tokenizer = pipeline.tokenizer
    if cfg["use_pretrained"]:
        encoder = pipeline.model.model.encoder
        d_model = pipeline.model.model.config.d_model
    else:
        sc = cfg["scratch"]
        t5_cfg = T5Config(
            vocab_size        = pipeline.model.model.config.vocab_size,
            d_model           = sc["d_model"],
            d_kv              = sc["d_kv"],
            d_ff              = sc["d_ff"],
            num_heads         = sc["num_heads"],
            num_layers        = sc["num_layers"],
            feed_forward_proj = "gated-gelu",
        )
        encoder = T5EncoderModel(t5_cfg).encoder
        d_model = sc["d_model"]

    ts_encoder = ChronosEncoder(encoder, tokenizer, d_model).to(device, dtype=dtype)
    embed_dim  = ts_encoder.hidden_dim   # n_patches * d_model
    return ts_encoder, embed_dim
