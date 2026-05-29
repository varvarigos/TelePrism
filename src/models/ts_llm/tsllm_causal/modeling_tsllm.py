"""
TSLLMForCausalLM — Time-Series Language Model for causal generation.

Dual-mode model class compatible with both vLLM and HuggingFace/VERL-FSDP.

Architecture:
    TS Encoder  : TOTO or TeleEncoder (selected via GRPO_TSLLM_CONFIG)
    Align Layer : Linear encoder_embed_dim -> hidden_size
    Base LLM    : Qwen3-4B

Weight sources:
    1. TS encoder + align layer: DeepSpeed ZeRO checkpoint (or baked safetensors)
    2. Base LLM: Qwen3-4B safetensors (via AutoWeightsLoader)
    3. LoRA deltas: DeepSpeed checkpoint, merged in-place into the LLM
"""

import logging
import os
import yaml
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import numpy as np

_tsllm_logger = logging.getLogger(__name__)


def _load_grpo_config():
    """Load encoder config from GRPO_TSLLM_CONFIG env var, or return defaults (toto)."""
    cfg_path = os.environ.get("GRPO_TSLLM_CONFIG")
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[modeling_tsllm] Loaded encoder config: {cfg_path} (encoder={cfg.get('encoder', 'toto')})")
        return cfg
    # Default: toto (backwards compatible)
    return {"encoder": "toto", "toto": {"use_pretrained": True, "embed_dim": 1536}}


def _diag_checksums(model):
    """Compute per-group weight sum + grad status for diagnostic logging."""
    groups = {"ts_encoder": [], "align_layer": [], "lora": []}
    for name, param in model.named_parameters():
        if name.startswith("ts_encoder.") or (hasattr(model, "base_model") and "ts_encoder" in name):
            groups["ts_encoder"].append((name, param))
        elif name.startswith("align_layer.") or (hasattr(model, "base_model") and "align_layer" in name):
            groups["align_layer"].append((name, param))
        elif "lora_" in name:
            groups["lora"].append((name, param))

    parts = []
    for g, params in groups.items():
        if params:
            n_grad = sum(1 for _, p in params if p.requires_grad)
            wsum = sum(p.data.float().sum().item() for _, p in params)
            parts.append(f"{g}: {len(params)} params ({n_grad} trainable), sum={wsum:.4f}")
    return " | ".join(parts)

# vLLM imports are optional. VERL FSDP workers run in a different venv
# that may not have vLLM installed. All vLLM symbols are imported inside
# a try/except so HF trust_remote_code loading works without vLLM.
try:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.interfaces import (
        MultiModalEmbeddings,
        SupportsMultiModal,
    )
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        WeightsMapper,
        init_vllm_registered_model,
        maybe_prefix,
    )
    from vllm.multimodal import MultiModalKwargsItems
    from vllm.sequence import IntermediateTensors
    _VLLM_AVAILABLE = True
except ImportError:
    VllmConfig = None
    SupportsMultiModal = object
    MultiModalEmbeddings = object
    IntermediateTensors = object
    _VLLM_AVAILABLE = False


# Debug flag: set TSLLM_DEBUG=1 to enable verbose diagnostic logging
_TSLLM_DEBUG = os.environ.get("TSLLM_DEBUG", "0") == "1"

# Global diagnostic counters (reset per step via train())
_FWD_DIAG_COUNT = 0
_FWD_DIAG_STEP = 0


# ── VERL FSDP worker patches ─────────────────────────────────────────────
# This module is loaded via trust_remote_code=True in EVERY process that
# loads the model, including FSDP WorkerDict processes.  The import-hook
# based patches in fsdp_patches.py are unreliable across Ray actor processes.
# We patch here instead, where execution is guaranteed.
import sys as _sys
import functools as _functools

_fsdp_mod = _sys.modules.get("verl.workers.fsdp_workers")
if _fsdp_mod is not None and hasattr(_fsdp_mod, "get_peft_model"):
    _orig_get_peft = _fsdp_mod.get_peft_model

    def _tsllm_get_peft_model(model, config, *args, **kwargs):
        # VERL passes target_modules as a comma-separated string; PEFT needs a list
        if hasattr(config, 'target_modules') and isinstance(config.target_modules, str) and ',' in config.target_modules:
            config.target_modules = [m.strip() for m in config.target_modules.split(',')]
        result = _orig_get_peft(model, config, *args, **kwargs)
        # Unfreeze ts_encoder/align_layer (PEFT froze them) unless config says not to
        _grpo_cfg_peft = _load_grpo_config()
        _train_encoder = _grpo_cfg_peft.get("training", {}).get("train_encoder", True)
        n = 0
        if _train_encoder:
            for name, p in result.named_parameters():
                if ("ts_encoder" in name or "align_layer" in name) and not p.requires_grad:
                    p.requires_grad_(True)
                    n += 1
        else:
            print("[modeling_tsllm] train_encoder=false → ts_encoder/align_layer stay FROZEN")
        # Cast ALL params to bf16 for FSDP gradient/optimizer dtype uniformity.
        cast = 0
        for p in result.parameters():
            if p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
                cast += 1
        # FSDP requires all params to be ≥1D. Reshape any scalar params
        # (e.g. TeleEncoder's gnn_alpha, gnn_tau) to 1D tensors.
        scalars = 0
        for name, p in result.named_parameters():
            if p.dim() == 0:
                p.data = p.data.unsqueeze(0)
                scalars += 1
        # Tag ts_encoder/align_layer param ids for separate LR in build_optimizer
        _ts_ids = set()
        for name, p in result.named_parameters():
            if "ts_encoder" in name or "align_layer" in name:
                _ts_ids.add(id(p))
        # Store on the build_optimizer function for later use
        _bo = _sys.modules.get("verl.workers.config.optimizer")
        if _bo and hasattr(_bo, "build_optimizer") and hasattr(_bo.build_optimizer, "_ts_param_ids"):
            _bo.build_optimizer._ts_param_ids = _ts_ids
            print(f"[modeling_tsllm] Tagged {len(_ts_ids)} ts_encoder/align_layer params for separate LR")
        print(f"[modeling_tsllm] get_peft_model: unfroze {n} TS params, cast {cast} to bf16, reshaped {scalars} scalars to 1D")
        return result

    _fsdp_mod.get_peft_model = _tsllm_get_peft_model

if _fsdp_mod is not None and hasattr(_fsdp_mod, "get_fsdp_wrap_policy"):
    _orig_get_wrap = _fsdp_mod.get_fsdp_wrap_policy

    def _tsllm_get_fsdp_wrap_policy(module, config=None, is_lora=False):
        policy = _orig_get_wrap(module, config=config, is_lora=is_lora)
        if not is_lora:
            return policy
        from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

        def _broad_lambda(module):
            # Skip nn.MultiheadAttention and its children — MHA accesses
            # out_proj.weight directly via F.multi_head_attention_forward,
            # bypassing module forward(). FSDP wrapping breaks this.
            if isinstance(module, torch.nn.MultiheadAttention):
                return False
            # Also skip children of MHA (out_proj, etc.)
            for parent in _mha_children:
                if module is parent:
                    return False

            # Wrap any module that directly owns at least one trainable param.
            direct_params = set(module.parameters()) - {
                p for c in module.children() for p in c.parameters()
            }
            if any(p.requires_grad for p in direct_params):
                return True
            # Leaf module with trainable params
            if len(list(module.children())) == 0:
                return any(p.requires_grad for p in module.parameters())
            return False

        # Collect all children of MHA modules to exclude them too
        _mha_children = set()
        for m in module.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                for child in m.modules():
                    if child is not m:
                        _mha_children.add(child)

        broad_policy = _functools.partial(lambda_auto_wrap_policy, lambda_fn=_broad_lambda)
        if policy is not None and hasattr(policy, "keywords") and "policies" in policy.keywords:
            orig_policies = list(policy.keywords["policies"])
            new_policies = []
            for p in orig_policies:
                if hasattr(p, "func") and p.func.__name__ == "lambda_auto_wrap_policy":
                    new_policies.append(broad_policy)
                else:
                    new_policies.append(p)
            result = _functools.partial(_or_policy, policies=new_policies)
        else:
            result = broad_policy
        print("[modeling_tsllm] Broadened FSDP wrap policy for TS encoder params ✓")
        return result

    _fsdp_mod.get_fsdp_wrap_policy = _tsllm_get_fsdp_wrap_policy
    print("[modeling_tsllm] Patched fsdp_workers (get_peft_model + get_fsdp_wrap_policy) ✓")

# ── Patch build_optimizer to use separate LR for ts_encoder/align_layer ──
# GRPO gradients for the encoder are much weaker than SFT (outcome-level
# reward diluted across ~540 response tokens), so the encoder needs a
# higher LR to make meaningful updates.
_opt_cfg_mod = _sys.modules.get("verl.workers.config.optimizer")
if _opt_cfg_mod is not None and hasattr(_opt_cfg_mod, "build_optimizer"):
    _orig_build_optimizer = _opt_cfg_mod.build_optimizer

    def _tsllm_build_optimizer(parameters, config):
        """Build optimizer with separate LR for ts_encoder/align_layer."""
        import importlib
        grpo_cfg = _load_grpo_config()
        encoder_lr_scale = float(grpo_cfg.get("training", {}).get("encoder_lr_scale", 1.0))
        print(f"[modeling_tsllm] _tsllm_build_optimizer called! encoder_lr_scale={encoder_lr_scale}")

        if encoder_lr_scale == 1.0:
            print(f"[modeling_tsllm] encoder_lr_scale=1.0, using uniform LR")
            return _orig_build_optimizer(parameters, config)

        # Consume the generator into a list
        params_list = list(parameters)
        # Use the tagged param ids from get_peft_model (pre-FSDP)
        # After FSDP with use_orig_params=True, the param objects should
        # be the same (FSDP keeps references to original params).
        ts_ids = _tsllm_build_optimizer._ts_param_ids
        ts_params = [p for p in params_list if id(p) in ts_ids]
        other_params = [p for p in params_list if id(p) not in ts_ids]

        base_lr = config.lr
        enc_lr = base_lr * encoder_lr_scale

        optimizer_name = config.optimizer.lower()
        optimizer_args = {"weight_decay": config.weight_decay}
        if "adam" in optimizer_name or "ademamix" in optimizer_name:
            optimizer_args["betas"] = config.betas
        if config.override_optimizer_config is not None:
            optimizer_args.update(config.override_optimizer_config)

        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)

        if ts_params:
            optimizer = optimizer_cls([
                {"params": other_params, "lr": base_lr},
                {"params": ts_params, "lr": enc_lr},
            ], **optimizer_args)
            print(f"[modeling_tsllm] Optimizer: {len(other_params)} params at lr={base_lr}, "
                  f"{len(ts_params)} ts_encoder/align params at lr={enc_lr} "
                  f"(scale={encoder_lr_scale}x)")
        else:
            # Fallback: FSDP changed param ids, can't identify encoder params
            optimizer = optimizer_cls(params_list, lr=base_lr, **optimizer_args)
            print(f"[modeling_tsllm] WARNING: Could not identify ts_encoder params "
                  f"(FSDP may have changed param ids). Using uniform lr={base_lr}")

        return optimizer

    _tsllm_build_optimizer._ts_param_ids = set()
    _opt_cfg_mod.build_optimizer = _tsllm_build_optimizer

    # Also patch in fsdp_workers since it imports build_optimizer by name
    if _fsdp_mod is not None:
        _fsdp_mod.build_optimizer = _tsllm_build_optimizer
        # Verify the patch took effect
        assert _fsdp_mod.build_optimizer is _tsllm_build_optimizer, "build_optimizer patch FAILED"

    print(f"[modeling_tsllm] Patched build_optimizer for separate encoder LR ✓ "
          f"(fsdp_mod patched: {_fsdp_mod is not None and _fsdp_mod.build_optimizer is _tsllm_build_optimizer})")

# ── dp_actor per-component gradient norms ───────────────────────
# Wrap _optimizer_step to compute per-component grad norms before clipping,
# and update_policy to inject them into the metrics dict sent to wandb.
# Keys logged: actor/grad_norm_ts_encoder, actor/grad_norm_align_layer,
#              actor/grad_norm_language_model
import math as _math

_dp_actor_mod = _sys.modules.get("verl.workers.actor.dp_actor")
if _dp_actor_mod is not None and hasattr(_dp_actor_mod, "DataParallelPPOActor"):
    _DPActor = _dp_actor_mod.DataParallelPPOActor
    _orig_optimizer_step = _DPActor._optimizer_step
    _orig_update_policy = _DPActor.update_policy

    # Read encoder L2 regularization strength from GRPO config
    _grpo_cfg_for_l2 = _load_grpo_config()
    _ENCODER_L2_LAMBDA = float(_grpo_cfg_for_l2.get("training", {}).get("encoder_l2_lambda", 0.0))
    if _ENCODER_L2_LAMBDA > 0:
        print(f"[modeling_tsllm] Encoder L2 regularization enabled: lambda={_ENCODER_L2_LAMBDA}")

    def _tsllm_optimizer_step(self):
        """Compute per-component grad norms, apply encoder L2 reg, then run original step."""
        import torch as _torch
        import torch.distributed as _dist

        # ── Snapshot initial encoder weights on first call (anchor for L2) ──
        if not hasattr(self, "_ts_encoder_init_weights"):
            self._ts_encoder_init_weights = {}
            for name, p in self.actor_module.named_parameters():
                if "ts_encoder" in name or "align_layer" in name:
                    self._ts_encoder_init_weights[name] = p.data.detach().clone()
            print(f"[modeling_tsllm] Snapshotted {len(self._ts_encoder_init_weights)} "
                  f"encoder/align params as L2 anchor")

            # ── AUDIT: Are ts_encoder/align_layer in the optimizer? ──
            ts_in_opt = 0
            ts_total = 0
            opt_param_ids = set()
            for group in self.actor_optimizer.param_groups:
                for op in group['params']:
                    opt_param_ids.add(id(op))
            for name, p in self.actor_module.named_parameters():
                if "ts_encoder" in name or "align_layer" in name:
                    ts_total += 1
                    in_opt = id(p) in opt_param_ids
                    if in_opt:
                        ts_in_opt += 1
                    if ts_total <= 5:
                        print(f"  [AUDIT] {name}: requires_grad={p.requires_grad}, "
                              f"in_optimizer={in_opt}, shape={tuple(p.shape)}")
            print(f"  [AUDIT] ts_encoder/align_layer in optimizer: {ts_in_opt}/{ts_total}")
            print(f"  [AUDIT] Total optimizer params: {len(opt_param_ids)}")

        # ── L2 regularization: add lambda*(param - param_init) to grads ──
        l2_lambda = _ENCODER_L2_LAMBDA
        if l2_lambda > 0:
            for name, p in self.actor_module.named_parameters():
                if p.grad is None:
                    continue
                if name in self._ts_encoder_init_weights:
                    p.grad.add_(l2_lambda * (p.data - self._ts_encoder_init_weights[name]))

        # ── Per-component grad norms (after L2 reg, before clip) ──
        groups = {"ts_encoder": 0.0, "align_layer": 0.0, "language_model": 0.0}
        for name, p in self.actor_module.named_parameters():
            if p.grad is None:
                continue
            sq = p.grad.detach().float().pow(2).sum()
            if "ts_encoder" in name:
                groups["ts_encoder"] += sq.item()
            elif "align_layer" in name:
                groups["align_layer"] += sq.item()
            else:
                groups["language_model"] += sq.item()

        # All-reduce squared norms across data-parallel ranks
        if _dist.is_initialized():
            for key in groups:
                t = _torch.tensor(groups[key], device="cuda")
                _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
                groups[key] = t.item()

        self._ts_component_grad_norms = {
            "actor/grad_norm_ts_encoder":      _math.sqrt(groups["ts_encoder"]),
            "actor/grad_norm_align_layer":     _math.sqrt(groups["align_layer"]),
            "actor/grad_norm_language_model":  _math.sqrt(groups["language_model"]),
        }

        # ── DIAG: snapshot param sums before step to verify optimizer updates ──
        if not hasattr(self, "_opt_step_count"):
            self._opt_step_count = 0
        self._opt_step_count += 1
        _pre_sums = {}
        if self._opt_step_count <= 3 or self._opt_step_count % 50 == 0:
            for name, p in self.actor_module.named_parameters():
                if "ts_encoder" in name or "align_layer" in name:
                    _pre_sums[name] = p.data.float().sum().item()

        result = _orig_optimizer_step(self)

        if _pre_sums:
            delta_sum = 0.0
            n_changed = 0
            for name, p in self.actor_module.named_parameters():
                if name in _pre_sums:
                    post = p.data.float().sum().item()
                    d = abs(post - _pre_sums[name])
                    delta_sum += d
                    if d > 1e-10:
                        n_changed += 1
            print(f"[DIAG optimizer_step #{self._opt_step_count}] "
                  f"ts_encoder/align_layer: {n_changed}/{len(_pre_sums)} params changed, "
                  f"total |delta|={delta_sum:.6f}")

        return result

    def _tsllm_update_policy(self, data):
        metrics = _orig_update_policy(self, data)
        if hasattr(self, "_ts_component_grad_norms"):
            metrics.update(self._ts_component_grad_norms)
        return metrics

    _DPActor._optimizer_step = _tsllm_optimizer_step
    _DPActor.update_policy = _tsllm_update_policy
    print("[modeling_tsllm] Patched dp_actor for per-component grad norm tracking ✓")

# ── Patch compute_ref_log_prob to use initial ts_encoder/align_layer weights ──
# With LoRA, VERL's reference model = actor with disable_adapter(). But
# ts_encoder and align_layer are NOT LoRA adapters, so disable_adapter()
# doesn't affect them. Both actor and ref share the SAME updated encoder,
# making KL blind to encoder drift. Fix: swap in initial encoder weights
# when computing reference log_probs, then swap back.
if _fsdp_mod is not None and hasattr(_fsdp_mod, "ActorRolloutRefWorker"):
    _ARRWorker = _fsdp_mod.ActorRolloutRefWorker
    _orig_compute_ref_log_prob = _ARRWorker.compute_ref_log_prob

    def _tsllm_compute_ref_log_prob(self, data):
        """Swap ts_encoder/align_layer to initial weights for reference log_prob."""
        import torch as _torch

        if not self._is_lora:
            return _orig_compute_ref_log_prob(self, data)

        # Snapshot initial weights on first call
        if not hasattr(self, "_ts_ref_weights"):
            self._ts_ref_weights = {}
            for name, p in self.actor.actor_module.named_parameters():
                if "ts_encoder" in name or "align_layer" in name:
                    self._ts_ref_weights[name] = p.data.detach().clone()
            print(f"[modeling_tsllm] Snapshotted {len(self._ts_ref_weights)} "
                  f"ts_encoder/align_layer params as reference weights")

        # Swap in initial weights
        current_weights = {}
        n_swapped = 0
        drift_sum = 0.0
        for name, p in self.actor.actor_module.named_parameters():
            if name in self._ts_ref_weights:
                current_weights[name] = p.data.detach().clone()
                drift_sum += (p.data - self._ts_ref_weights[name]).float().abs().sum().item()
                p.data.copy_(self._ts_ref_weights[name])
                n_swapped += 1

        if not hasattr(self, "_ref_swap_count"):
            self._ref_swap_count = 0
        self._ref_swap_count += 1
        if self._ref_swap_count <= 3 or self._ref_swap_count % 10 == 0:
            print(f"[modeling_tsllm] ref_log_prob: swapped {n_swapped} params to initial "
                  f"(encoder drift L1={drift_sum:.4f}, call #{self._ref_swap_count})")

        try:
            result = _orig_compute_ref_log_prob(self, data)
        finally:
            # Swap back current weights
            for name, p in self.actor.actor_module.named_parameters():
                if name in current_weights:
                    p.data.copy_(current_weights[name])

        return result

    _ARRWorker.compute_ref_log_prob = _tsllm_compute_ref_log_prob
    print("[modeling_tsllm] Patched compute_ref_log_prob to use initial ts_encoder/align_layer ✓")

# ── Patch collect_lora_params to include ts_encoder/align_layer in weight sync ─
# VERL's collect_lora_params only sends LoRA adapter weights on subsequent syncs,
# but ts_encoder and align_layer are trained as full parameters (not LoRA).
# Without this patch, the vLLM rollout worker never sees their updated weights.
_fsdp_utils_mod = _sys.modules.get("verl.utils.fsdp_utils")
if _fsdp_utils_mod is not None and hasattr(_fsdp_utils_mod, "collect_lora_params"):
    _orig_collect_lora = _fsdp_utils_mod.collect_lora_params

    def _tsllm_collect_lora_params(module, layered_summon, base_sync_done):
        """Include ts_encoder/align_layer full params alongside LoRA weights."""
        params = _orig_collect_lora(module, layered_summon, base_sync_done)

        if base_sync_done:
            # The original only returns LoRA weights. We need to also include
            # ts_encoder and align_layer so the vLLM worker gets their updates.
            from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
            peft_model = getattr(module, "_fsdp_wrapped_module", module)
            inner_model = peft_model.base_model.model  # TSLLMForCausalLM

            with _FSDP.summon_full_params(module, writeback=False):
                for name, p in inner_model.named_parameters():
                    if name.startswith("ts_encoder.") or name.startswith("align_layer."):
                        full = p.full_tensor().detach().cpu() if hasattr(p, "full_tensor") else p.detach().cpu()
                        params[name] = full

            n_ts = sum(1 for k in params if k.startswith("ts_encoder.") or k.startswith("align_layer."))
            print(f"[modeling_tsllm] collect_lora_params: added {n_ts} ts_encoder/align_layer params to sync")

        return params

    _fsdp_utils_mod.collect_lora_params = _tsllm_collect_lora_params
    # Also patch the local reference in fsdp_workers (imported by name, not by module)
    if _fsdp_mod is not None:
        _fsdp_mod.collect_lora_params = _tsllm_collect_lora_params
    # And in the engine transformer_impl if loaded
    _engine_mod = _sys.modules.get("verl.workers.engine.fsdp.transformer_impl")
    if _engine_mod is not None and hasattr(_engine_mod, "collect_lora_params"):
        _engine_mod.collect_lora_params = _tsllm_collect_lora_params
    print("[modeling_tsllm] Patched collect_lora_params (fsdp_utils + fsdp_workers + engine) ✓")
# ──────────────────────────────────────────────────────────────────────────


class TSLLMForCausalLM(nn.Module, SupportsMultiModal):
    """
    Time-Series Language Model compatible with vLLM and HF/VERL-FSDP.

    vLLM mode:
        model = TSLLMForCausalLM(vllm_config=...)

    HF/FSDP mode (trust_remote_code):
        model = TSLLMForCausalLM.from_pretrained(model_path)
    """

    # Required by HF's custom_object_save / _set_auto_map_in_config
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, *, vllm_config=None, hf_config=None, prefix: str = ""):
        super().__init__()

        if vllm_config is not None:
            config = vllm_config.model_config.hf_config
        elif hf_config is not None:
            config = hf_config
        else:
            raise ValueError("Either vllm_config or hf_config must be provided")

        self.config = config
        self._hf_mode = vllm_config is None

        if vllm_config is not None:
            model_dtype = vllm_config.model_config.dtype
        else:
            _dt = getattr(config, "torch_dtype", "bfloat16")
            model_dtype = getattr(torch, _dt) if isinstance(_dt, str) else _dt

        # TS encoder (selected via GRPO_TSLLM_CONFIG)
        grpo_cfg = _load_grpo_config()
        self._encoder_type = grpo_cfg.get("encoder", "toto")

        if self._encoder_type == "toto":
            from toto.model.toto import Toto
            from toto.inference.forecaster import TotoForecaster

            toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
            self.ts_encoder = TotoForecaster(toto.model).model

            if model_dtype != torch.float32:
                self.ts_encoder = self.ts_encoder.to(dtype=model_dtype)

            self.ts_encoder.get_embeddings = (
                self._get_embeddings_fn().__get__(self.ts_encoder, type(self.ts_encoder))
            )
            ts_embed_dim = 768 * 2  # 1536

        elif self._encoder_type == "teleencoder":
            from teleprism.encoders.teleencoder.teleEncoder import TeleEncoder

            te_cfg = grpo_cfg["teleencoder"]
            self.ts_encoder = TeleEncoder(
                num_channels=te_cfg["num_channels"],
                seq_len=te_cfg["seq_len"],
                d_model=te_cfg["d_model"],
                num_heads=te_cfg["num_heads"],
                num_layers=te_cfg["num_layers"],
                d_ff=te_cfg["d_ff"],
                num_experts=te_cfg["num_experts"],
                top_k=te_cfg["top_k"],
                K=te_cfg["K"],
                patch_size=te_cfg["patch_size"],
                patch_stride=te_cfg["patch_stride"],
                dropout=te_cfg["dropout"],
                router_type=te_cfg["router_type"],
                ctx_weight=te_cfg["ctx_weight"],
                router_temperature=te_cfg["router_temperature"],
                attn_mode=te_cfg["attn_mode"],
                scale_encoder=te_cfg["scale_encoder"],
                d_node=te_cfg["d_node"],
                n_gnn_layers=te_cfg["n_gnn_layers"],
                gnn_alpha_init=te_cfg["gnn_alpha_init"],
                gnn_tau_init=te_cfg["gnn_tau_init"],
                sparse_inference=te_cfg["sparse_inference"],
                router_init_std=te_cfg["router_init_std"],
                expert_init_gain=te_cfg["expert_init_gain"],
                router_jitter_std=te_cfg["router_jitter_std"],
                dense_routing_warmup_epochs=te_cfg["dense_routing_warmup_epochs"],
            )
            if model_dtype != torch.float32:
                self.ts_encoder = self.ts_encoder.to(dtype=model_dtype)

            ts_embed_dim = te_cfg["embed_dim"]
            print(f"[modeling_tsllm] TeleEncoder: embed_dim={ts_embed_dim}")
        else:
            raise ValueError(f"Unknown encoder type: {self._encoder_type}")

        # Align layer: encoder output dim -> LLM hidden dim
        from teleprism.utils.utils import AlignLayer
        self.align_layer = AlignLayer(ts_embed_dim, config.hidden_size)
        if model_dtype != torch.float32:
            self.align_layer = self.align_layer.to(dtype=model_dtype)

        # Load TS encoder + align layer + cache LoRA from DS checkpoint
        self._pending_lora: dict[str, torch.Tensor] = {}
        self._weights_baked = getattr(config, "weights_baked", False)
        ts_checkpoint_dir = os.environ.get("TS_CHECKPOINT_DIR")
        ts_checkpoint_tag = os.environ.get("TS_CHECKPOINT_TAG")
        verl_skip = os.environ.get("VERL_SKIP_TS_CHECKPOINT", "0") == "1"
        if not self._weights_baked and not verl_skip and ts_checkpoint_dir and ts_checkpoint_tag:
            self._load_ts_checkpoint(ts_checkpoint_dir, ts_checkpoint_tag)

        # Safety: when weights_baked=True (even in vLLM mode), pre-load
        # ts_encoder and align_layer from the baked safetensors NOW.
        # vLLM colocated mode uses load_format=dummy → language_model gets
        # random init (overwritten by IPC sync).  But if VERL's IPC sync
        # doesn't include ts_encoder/align_layer weights, they'd stay as
        # generic pretrained TOTO / random align_layer.  Pre-loading here
        # ensures they're always correct.  If IPC sync DOES include them,
        # load_weights() will harmlessly overwrite with the same values.
        if self._weights_baked and vllm_config is not None:
            self._preload_ts_from_baked(vllm_config)

        # Language model
        if vllm_config is not None:
            with self._mark_language_model(vllm_config):
                self.language_model = init_vllm_registered_model(
                    vllm_config=vllm_config,
                    hf_config=config,
                    prefix=maybe_prefix(prefix, "language_model"),
                    architectures=["Qwen3ForCausalLM", "Qwen2ForCausalLM"],
                )
            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors
            )
        else:
            from transformers import Qwen3ForCausalLM as _Qwen3
            self.language_model = _Qwen3(config).to(dtype=model_dtype)
            self.make_empty_intermediate_tensors = None

    # -- SupportsMultiModal interface --

    @classmethod
    def is_backend_compatible(cls) -> bool:
        """Return False so vLLM skips the Transformers-backend path and
        falls through to the ModelRegistry lookup for TSLLMModel."""
        return False

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "timeseries":
            return "<|begin_of_TS|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Process time-series data and return per-item embeddings.

        Each item returns [20, D]:
            [0]    = begin token embedding
            [1:19] = 18 TOTO encoder embeddings (one per channel)
            [19]   = end token embedding
        """
        timeseries = kwargs.pop("timeseries", None)
        if timeseries is None:
            return []

        ts_input = timeseries

        if isinstance(ts_input, list):
            ts_input = np.stack(
                [x.numpy() if isinstance(x, torch.Tensor) else x for x in ts_input],
                axis=0,
            )
        if isinstance(ts_input, np.ndarray):
            ts_input = torch.from_numpy(ts_input.copy()).float()

        if ts_input.dim() == 2:
            ts_input = ts_input.unsqueeze(0)
        elif ts_input.dim() > 3:
            ts_input = ts_input.view(-1, ts_input.shape[-2], ts_input.shape[-1])

        B = ts_input.shape[0]
        ts_embeds = self._encode_timeseries(ts_input)
        device, dtype = ts_embeds.device, ts_embeds.dtype

        embed_tokens = self.language_model.model.embed_tokens
        begin_embed = embed_tokens(
            torch.tensor([151669], device=device)
        ).to(dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        end_embed = embed_tokens(
            torch.tensor([151670], device=device)
        ).to(dtype=dtype).unsqueeze(0).expand(B, -1, -1)

        return torch.cat([begin_embed, ts_embeds, end_embed], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ts_values: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self._hf_mode:
            # ── Always-on diagnostic: confirm ts_values reaches FSDP actor ──
            if not hasattr(self, '_ts_fwd_diag_done'):
                self._ts_fwd_diag_done = True
                print(f"\n{'='*60}")
                print(f"[DIAG forward()] FIRST HF-mode forward call")
                print(f"  ts_values is None: {ts_values is None}")
                if ts_values is not None:
                    print(f"  ts_values shape: {ts_values.shape}, dtype: {ts_values.dtype}")
                    print(f"  ts_values[0,0,:5]: {ts_values[0,0,:5].tolist()}")
                    print(f"  ts_values range: [{ts_values.min():.4f}, {ts_values.max():.4f}]")
                if input_ids is not None:
                    ids = input_ids[0].tolist()
                    n_begin = ids.count(151669)
                    n_end = ids.count(151670)
                    print(f"  input_ids shape: {input_ids.shape}")
                    print(f"  TS markers: {n_begin} begin (151669), {n_end} end (151670)")
                print(f"{'='*60}\n")

            if ts_values is not None and input_ids is not None:
                # ── DIAG: log actor forward input (first 1 per step) ──
                global _FWD_DIAG_COUNT, _FWD_DIAG_STEP
                _FWD_DIAG_COUNT += 1
                if _TSLLM_DEBUG and _FWD_DIAG_COUNT <= 1:
                    ids_list = input_ids[0].tolist()
                    PAD_ID = 151643
                    first_non_pad = next((i for i, t in enumerate(ids_list) if t != PAD_ID), 0)
                    n_begin = ids_list.count(151669)
                    n_end = ids_list.count(151670)
                    print(f"\n{'='*60}")
                    print(f"[DIAG forward()] step~{_FWD_DIAG_STEP}")
                    print(f"  input_ids shape: {input_ids.shape}")
                    print(f"  left-pad tokens: {first_non_pad}")
                    print(f"  TS markers: {n_begin} begin, {n_end} end")
                    print(f"  ts_values shape={ts_values.shape}, range=[{ts_values.min():.2f}, {ts_values.max():.2f}]")
                    print(f"{'='*60}\n")

                old_len = input_ids.shape[1]
                _orig_input_ids = input_ids  # keep for mask rebuild
                inputs_embeds = self._merge_ts_embeddings(input_ids, ts_values)
                new_len = inputs_embeds.shape[1]
                input_ids = None

                if _TSLLM_DEBUG and _FWD_DIAG_COUNT <= 1:
                    print(f"[DIAG forward()] after merge: old_len={old_len}, new_len={new_len}, inputs_embeds={inputs_embeds.shape}")

                # _merge_ts_embeddings inserts TS tokens (delta extra).
                # Rebuild attention_mask: insert delta ones at each sample's
                # begin_of_TS position so the mask stays aligned with the
                # rearranged embeddings (matching SFT's tsllm.py approach).
                if new_len != old_len:
                    B = inputs_embeds.shape[0]
                    delta = new_len - old_len
                    device = inputs_embeds.device

                    if attention_mask is not None:
                        new_masks = []
                        for b in range(B):
                            ids = _orig_input_ids[b]
                            sp = (ids == 151669).nonzero(as_tuple=False)
                            if sp.numel() > 0:
                                s = sp[0, 0].item()
                                ts_ones = attention_mask.new_ones(1, delta)
                                new_masks.append(torch.cat([
                                    attention_mask[b:b+1, :s+1],
                                    ts_ones,
                                    attention_mask[b:b+1, s+1:],
                                ], dim=1))
                            else:
                                ext = attention_mask.new_ones(1, delta)
                                new_masks.append(torch.cat([attention_mask[b:b+1], ext], dim=1))
                        attention_mask = torch.cat(new_masks, dim=0)

                    # Recompute position_ids from the corrected mask.
                    # With left-padding, cumsum gives content positions
                    # starting at 0 (matching pretraining). Padding positions
                    # get a fixed value (1) and are masked out in attention.
                    if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                        position_ids = attention_mask.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attention_mask == 0, 1)
                        kwargs = dict(kwargs)
                        kwargs["position_ids"] = position_ids

            fwd_kwargs = {k: v for k, v in kwargs.items()
                          if k in ("position_ids", "use_cache", "output_attentions",
                                   "output_hidden_states", "return_dict")}
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **fwd_kwargs,
            )
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    # -- Training mode: unfreeze TS components after PEFT freezing --

    def train(self, mode=True):
        """Override train() to re-enable gradients on ts_encoder and align_layer.

        When VERL wraps the model with PEFT LoRA, it freezes ALL non-adapter
        parameters (requires_grad=False).  ts_encoder and align_layer are
        excluded from LoRA (via exclude_modules), so they get frozen too.

        VERL calls model.train() before training starts.  This override
        re-enables gradients so ts_encoder and align_layer are jointly
        trained alongside the LoRA adapters.

        The optimizer already has references to ALL params (VERL calls
        build_optimizer(module.parameters()) before train()).  Adam will
        lazily initialize optimizer state on the first gradient.
        """
        result = super().train(mode)
        _grpo_cfg_train = _load_grpo_config()
        _should_train_encoder = _grpo_cfg_train.get("training", {}).get("train_encoder", True)
        # When encoder is frozen, keep it in eval mode to avoid dropout
        # mismatch between compute_log_prob (eval) and update_policy (train).
        # TeleEncoder has dropout=0.1; Qwen3 LLM has dropout=0.0.
        if self._hf_mode and not _should_train_encoder:
            self.ts_encoder.eval()
            self.align_layer.eval()
        if mode and self._hf_mode and _should_train_encoder:
            # Re-enable gradients on TS components (PEFT froze them)
            n_unfrozen = 0
            for p in self.ts_encoder.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    n_unfrozen += 1
            for p in self.align_layer.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    n_unfrozen += 1
            for name, param in self.named_parameters():
                if ("ts_encoder" in name or "align_layer" in name) and not param.requires_grad:
                    param.requires_grad_(True)
                    n_unfrozen += 1

            if n_unfrozen > 0:
                print(f"[train()] Unfroze {n_unfrozen} ts_encoder/align_layer params for joint training")

            global _FWD_DIAG_COUNT, _FWD_DIAG_STEP
            _FWD_DIAG_STEP += 1
            _FWD_DIAG_COUNT = 0

            if _TSLLM_DEBUG:
                all_names = [(n, p.requires_grad, p.shape) for n, p in self.named_parameters()]
                ts_names = [n for n, g, s in all_names if "ts_encoder" in n or "align_layer" in n]
                _tsllm_logger.warning(
                    f"[DIAG train() ALL] total={len(all_names)}, "
                    f"ts/align visible={len(ts_names)}, "
                    f"first 10 names: {[n for n,_,_ in all_names[:10]]}"
                )
                if ts_names:
                    _tsllm_logger.warning(f"[DIAG train() TS] ts/align names: {ts_names[:5]}")
                if n_unfrozen == 0:
                    _tsllm_logger.warning("[DIAG train()] n_unfrozen=0 — ts/align already unfrozen")

                diag = _diag_checksums(self)
                _tsllm_logger.warning(f"[DIAG train()] step={_FWD_DIAG_STEP} | {diag}")
                try:
                    for name, param in self.named_parameters():
                        if "align_layer" in name and "weight" in name:
                            vals = param.data.float().flatten()[:5].tolist()
                            print(f"[DIAG train() step={_FWD_DIAG_STEP}] {name} first 5: {vals}")
                            break
                    for name, param in self.named_parameters():
                        if "ts_encoder" in name and "patch_embed" in name:
                            vals = param.data.float().flatten()[:5].tolist()
                            print(f"[DIAG train() step={_FWD_DIAG_STEP}] {name} first 5: {vals}")
                            break
                except Exception as e:
                    print(f"[DIAG train()] error: {e}")
        return result

    # -- Gradient checkpointing (VERL FSDP) --

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {}
        )

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()

    # -- PEFT/LoRA helpers --

    def enable_input_require_grads(self):
        """Enable gradient computation on inputs for LoRA adapters."""
        def _make_inputs_require_grad(module, input, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
            elif isinstance(output, tuple):
                for o in output:
                    if isinstance(o, torch.Tensor) and o.is_floating_point():
                        o.requires_grad_(True)
        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(
            _make_inputs_require_grad
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.language_model.prepare_inputs_for_generation(*args, **kwargs)

    def can_generate(self) -> bool:
        """HF GenerationMixin compatibility — required by VERL's checkpoint manager."""
        return self.language_model.can_generate()

    def _get_no_split_modules(self, device_map=None):
        if hasattr(self.language_model, "_get_no_split_modules"):
            return self.language_model._get_no_split_modules(device_map)
        return []

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    # -- HF/FSDP: from_pretrained --

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """HuggingFace-compatible loader for VERL FSDP."""
        import glob
        from safetensors.torch import load_file
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )

        model = cls(hf_config=config)

        sf_files = sorted(
            glob.glob(os.path.join(pretrained_model_name_or_path, "*.safetensors"))
        )
        if not sf_files:
            raise FileNotFoundError(
                f"No .safetensors found in {pretrained_model_name_or_path}"
            )
        for sf_path in sf_files:
            sd = load_file(sf_path, device="cpu")
            mapped = {}
            for k, v in sd.items():
                if k.startswith("model.") or k.startswith("lm_head."):
                    mapped["language_model." + k] = v
                else:
                    mapped[k] = v
            missing, unexpected = model.load_state_dict(mapped, strict=False)

        if model._pending_lora:
            lora_alpha = float(getattr(config, "lora_alpha", 16))
            model._merge_lora_into_model_hf(model._pending_lora, lora_alpha)
            model._pending_lora = {}

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        if torch_dtype is not None:
            model.to(torch_dtype)

        # FSDP requires all params ≥1D. Unsqueeze scalar params AFTER loading.
        for name, p in model.named_parameters():
            if p.dim() == 0:
                p.data = p.data.unsqueeze(0)

        return model

    # -- TS embedding merge (HF mode) --
    def _merge_ts_embeddings(self, input_ids, ts_values):
        """Replace <|begin_of_TS|>...<|end_of_TS|> span with TS embeddings."""
        TS_START, TS_END = 151669, 151670
        device = input_ids.device
        dtype = next(self.language_model.parameters()).dtype

        word_embeds = self.language_model.get_input_embeddings()(input_ids)
        ts_embed = self._encode_timeseries(ts_values.to(device))

        B, L, D = word_embeds.shape
        rows = []
        for b in range(B):
            ids = input_ids[b]
            sp = (ids == TS_START).nonzero(as_tuple=False)
            ep = (ids == TS_END).nonzero(as_tuple=False)
            if sp.numel() == 0 or ep.numel() == 0:
                rows.append(word_embeds[b])
                continue
            s, e = sp[0, 0].item(), ep[0, 0].item()
            rows.append(torch.cat([
                word_embeds[b, :s + 1],
                ts_embed[b].to(dtype),
                word_embeds[b, e:],
            ], dim=0))

        max_len = max(r.shape[0] for r in rows)
        out = torch.zeros(B, max_len, D, dtype=dtype, device=device)
        for b, row in enumerate(rows):
            out[b, :row.shape[0]] = row
        return out

    # -- LoRA merge: HF unfused projections --

    def _merge_lora_into_model_hf(self, lora_state, lora_alpha):
        """Merge LoRA deltas into HF Qwen3ForCausalLM (unfused projections)."""
        DS_PREFIX = "base_model.base_model.model."

        def get_ab(path):
            a = lora_state.get(f"{DS_PREFIX}{path}.lora_A.default.weight")
            b = lora_state.get(f"{DS_PREFIX}{path}.lora_B.default.weight")
            return a, b

        sample_a = next(v for k, v in lora_state.items() if ".lora_A." in k)
        lora_r = sample_a.shape[0]
        scaling = lora_alpha / lora_r

        lm_params = dict(self.language_model.named_parameters())
        num_layers = self.config.num_hidden_layers
        errors = []
        merged = 0

        for i in range(num_layers):
            ckpt = f"model.layers.{i}"
            attn = f"model.layers.{i}.self_attn"
            mlp = f"model.layers.{i}.mlp"

            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                a, b = get_ab(f"{ckpt}.self_attn.{proj}")
                p = lm_params.get(f"{attn}.{proj}.weight")
                if a is None or p is None:
                    errors.append(f"layer {i}: missing {proj}")
                else:
                    p.data += ((b @ a) * scaling).to(p.device, p.dtype)
                    merged += 1

            for proj in ("gate_proj", "up_proj", "down_proj"):
                a, b = get_ab(f"{ckpt}.mlp.{proj}")
                p = lm_params.get(f"{mlp}.{proj}.weight")
                if a is None or p is None:
                    errors.append(f"layer {i}: missing {proj}")
                else:
                    p.data += ((b @ a) * scaling).to(p.device, p.dtype)
                    merged += 1

        if errors:
            raise RuntimeError(
                f"LoRA HF merge failed ({len(errors)} errors):\n" + "\n".join(errors)
            )
        print(f"[HF mode] Merged {merged} LoRA modules (r={lora_r}, alpha={lora_alpha}, scaling={scaling:.4f})")

    # -- LoRA merge: vLLM fused projections --

    def _merge_lora_into_model(self, lora_state, lora_alpha):
        """Merge LoRA deltas into vLLM language model (fused qkv_proj, gate_up_proj)."""
        DS_PREFIX = "base_model.base_model.model."

        def get_ab(key_path):
            a = lora_state.get(f"{DS_PREFIX}{key_path}.lora_A.default.weight")
            b = lora_state.get(f"{DS_PREFIX}{key_path}.lora_B.default.weight")
            return a, b

        sample_a = next(v for k, v in lora_state.items() if ".lora_A." in k)
        lora_r = sample_a.shape[0]
        scaling = lora_alpha / lora_r

        lm_params = dict(self.language_model.named_parameters())
        num_layers = self.config.num_hidden_layers
        errors = []
        merged = 0

        for i in range(num_layers):
            attn = f"model.layers.{i}.self_attn"
            mlp = f"model.layers.{i}.mlp"
            ckpt = f"model.layers.{i}"

            # qkv_proj (fused: q + k + v)
            qa, qb = get_ab(f"{ckpt}.self_attn.q_proj")
            ka, kb = get_ab(f"{ckpt}.self_attn.k_proj")
            va, vb = get_ab(f"{ckpt}.self_attn.v_proj")
            if qa is None or ka is None or va is None:
                errors.append(f"layer {i}: missing q/k/v lora")
            else:
                dq = (qb @ qa) * scaling
                dk = (kb @ ka) * scaling
                dv = (vb @ va) * scaling
                p = lm_params.get(f"{attn}.qkv_proj.weight")
                if p is None:
                    errors.append(f"layer {i}: qkv_proj not found")
                else:
                    p.data += torch.cat([dq, dk, dv], dim=0).to(p.device, p.dtype)
                    merged += 3

            # o_proj (unfused)
            oa, ob = get_ab(f"{ckpt}.self_attn.o_proj")
            if oa is not None:
                p = lm_params.get(f"{attn}.o_proj.weight")
                if p is not None:
                    p.data += ((ob @ oa) * scaling).to(p.device, p.dtype)
                    merged += 1

            # gate_up_proj (fused: gate + up)
            ga, gb = get_ab(f"{ckpt}.mlp.gate_proj")
            ua, ub = get_ab(f"{ckpt}.mlp.up_proj")
            if ga is None or ua is None:
                errors.append(f"layer {i}: missing gate/up lora")
            else:
                dg = (gb @ ga) * scaling
                du = (ub @ ua) * scaling
                p = lm_params.get(f"{mlp}.gate_up_proj.weight")
                if p is None:
                    errors.append(f"layer {i}: gate_up_proj not found")
                else:
                    p.data += torch.cat([dg, du], dim=0).to(p.device, p.dtype)
                    merged += 2

            # down_proj (unfused)
            da, db = get_ab(f"{ckpt}.mlp.down_proj")
            if da is not None:
                p = lm_params.get(f"{mlp}.down_proj.weight")
                if p is not None:
                    p.data += ((db @ da) * scaling).to(p.device, p.dtype)
                    merged += 1

        if errors:
            raise RuntimeError(
                f"LoRA merge failed ({len(errors)} errors):\n" + "\n".join(errors)
            )
        expected = num_layers * 7
        assert merged == expected, f"Expected {expected} LoRA merges but got {merged}"
        print(f"Merged {merged} LoRA modules (r={lora_r}, alpha={lora_alpha:.0f}, scaling={scaling:.3f})")

    # -- vLLM weight loader --

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from safetensors, VERL PEFT sync, or legacy DS checkpoint."""
        _mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.": "language_model.model.",
                "lm_head.": "language_model.lm_head.",
            }
        )

        weights_list = list(weights)
        has_peft = any(".base_layer." in name or ".lora_A." in name
                       for name, _ in weights_list)

        if has_peft:
            weights_list = self._merge_peft_weights(weights_list)

        # Separate ts_encoder / align_layer weights from LLM weights.
        # AutoWeightsLoader dispatches to submodules via their load_weights()
        # method, but ts_encoder (TOTO) and align_layer are plain nn.Modules
        # that don't implement that protocol — so AutoWeightsLoader silently
        # skips them.  We load them manually here.
        ts_align_weights: dict[str, torch.Tensor] = {}
        llm_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights_list:
            if name.startswith("ts_encoder.") or name.startswith("align_layer."):
                ts_align_weights[name] = tensor
            else:
                llm_weights.append((name, tensor))

        if self._weights_baked or has_peft:
            loader = AutoWeightsLoader(self)
            loaded = loader.load_weights(iter(llm_weights), mapper=_mapper)
        else:
            loader = AutoWeightsLoader(
                self,
                skip_prefixes=["ts_encoder.", "align_layer."],
            )
            loaded = loader.load_weights(iter(llm_weights), mapper=_mapper)
            if self._pending_lora:
                lora_alpha = float(getattr(self.config, "lora_alpha", 16))
                self._merge_lora_into_model(self._pending_lora, lora_alpha)
                self._pending_lora = {}

        # Manually load ts_encoder and align_layer weights
        if ts_align_weights:
            model_params = dict(self.named_parameters())
            loaded_ts = 0
            for name, tensor in ts_align_weights.items():
                param = model_params.get(name)
                if param is not None:
                    t = tensor.to(device=param.device, dtype=param.dtype)
                    # Handle scalar↔1D mismatch (FSDP reshapes scalars to 1D)
                    if param.shape != t.shape:
                        t = t.reshape(param.shape)
                    param.data.copy_(t)
                    loaded_ts += 1
            loaded |= set(ts_align_weights.keys())
            print(f"[load_weights] Manually loaded {loaded_ts} ts_encoder/align_layer params")
        elif not self._weights_baked:
            # Non-baked: ts_encoder/align_layer were loaded from DS checkpoint
            pre_loaded = {
                name for name, _ in self.named_parameters()
                if name.startswith("ts_encoder.") or name.startswith("align_layer.")
            }
            loaded |= pre_loaded

        return loaded

    def _merge_peft_weights(self, weights):
        """Preprocess PEFT-format weights: strip .base_layer., merge LoRA deltas."""
        base_weights = {}
        lora_a = {}
        lora_b = {}
        other_weights = []

        for name, tensor in weights:
            if ".base_layer." in name:
                base_weights[name.replace(".base_layer.", ".")] = tensor
            elif ".lora_A." in name:
                lora_a[name.split(".lora_A.")[0]] = tensor
            elif ".lora_B." in name:
                lora_b[name.split(".lora_B.")[0]] = tensor
            elif ".lora_embedding" in name or ".modules_to_save." in name:
                continue
            else:
                other_weights.append((name, tensor))

        if lora_a:
            sample_a = next(iter(lora_a.values()))
            lora_r = sample_a.shape[0]
            lora_alpha = float(getattr(self.config, "lora_alpha", 16))
            scaling = lora_alpha / lora_r
        else:
            scaling = 0.0

        for mod_path in lora_a:
            a = lora_a[mod_path]
            b = lora_b.get(mod_path)
            if b is None:
                continue
            weight_key = f"{mod_path}.weight"
            base = base_weights.get(weight_key)
            if base is not None:
                delta = (b @ a) * scaling
                base_weights[weight_key] = base + delta.to(dtype=base.dtype, device=base.device)

        result = list(base_weights.items())
        result.extend(other_weights)
        return result

    # -- Pre-load TS weights from baked safetensors (vLLM mode) --

    def _preload_ts_from_baked(self, vllm_config):
        """Pre-load ts_encoder and align_layer from baked safetensors.

        In vLLM colocated mode, the model starts with load_format=dummy
        (random language_model weights).  The IPC sync overwrites those.
        But ts_encoder and align_layer are initialized in __init__ as
        generic pretrained TOTO / random Linear — NOT from safetensors.

        If VERL's IPC sync doesn't include ts_encoder/align_layer weights,
        they'd stay wrong.  This method pre-loads them from the baked
        safetensors so they're always correct, regardless of IPC behavior.
        """
        import glob
        from safetensors.torch import load_file

        model_path = vllm_config.model_config.model
        sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not sf_files:
            print(f"[preload_ts] WARNING: no safetensors found in {model_path}")
            return

        loaded = 0
        model_params = dict(self.named_parameters())
        for sf_path in sf_files:
            sd = load_file(sf_path, device="cpu")
            for key, tensor in sd.items():
                if key.startswith("ts_encoder.") or key.startswith("align_layer."):
                    param = model_params.get(key)
                    if param is not None:
                        t = tensor.to(dtype=param.dtype)
                        # Handle scalar↔1D mismatch (scalars were unsqueezed
                        # during baking for FSDP/vLLM compatibility)
                        if param.dim() == 0 and t.dim() == 1 and t.numel() == 1:
                            t = t.squeeze(0)
                        elif param.dim() == 1 and t.dim() == 0:
                            param.data = param.data.unsqueeze(0)
                        param.data.copy_(t)
                        loaded += 1
        print(f"[preload_ts] Pre-loaded {loaded} ts_encoder/align_layer params from baked safetensors")

    # -- DS checkpoint loading --

    def _load_ts_checkpoint(self, checkpoint_dir, tag):
        """Load ts_encoder, align_layer, and cache LoRA from a DeepSpeed checkpoint."""
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        print(f"Loading DS checkpoint: {checkpoint_dir}, tag={tag}")
        sd = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=tag)

        ts_state = {
            k.replace("ts_encoder.", "", 1): v
            for k, v in sd.items() if k.startswith("ts_encoder.")
        }
        self.ts_encoder.load_state_dict(ts_state, strict=True)
        print(f"TS encoder: loaded {len(ts_state)} params")

        align_state = {
            k.replace("align_layer.", "", 1): v
            for k, v in sd.items() if k.startswith("align_layer.")
        }
        self.align_layer.load_state_dict(align_state, strict=True)
        print(f"Align layer: loaded {len(align_state)} params")

        lora_state = {
            k: v for k, v in sd.items() if ".lora_A." in k or ".lora_B." in k
        }
        if lora_state:
            self._pending_lora = lora_state
            n_pairs = sum(1 for k in lora_state if ".lora_A." in k)
            print(f"Cached {n_pairs} LoRA pairs for merging after safetensors load")
        else:
            print("WARNING: no LoRA weights found in checkpoint")

    # -- TS encoding internals --

    @staticmethod
    def _get_embeddings_fn():
        """Return the TOTO get_embeddings helper (only used when encoder=toto)."""
        def get_embeddings(self, inputs, input_padding_mask, id_mask,
                           kv_cache=None, scaling_prefix_length=None):
            scaled_inputs, _, _ = self.scaler(
                inputs,
                weights=torch.ones_like(inputs, device=inputs.device),
                padding_mask=input_padding_mask,
                prefix_length=scaling_prefix_length,
            )
            embeddings, reduced_id_mask = self.patch_embed(scaled_inputs, id_mask)
            transformed = self.transformer(embeddings, reduced_id_mask, kv_cache)
            return transformed
        return get_embeddings

    def _encode_timeseries(self, ts):
        """[B, C, T] -> [B, C, hidden_size]. Lazy-moves ts_encoder to GPU."""
        B, C, T = ts.shape
        device = self.align_layer.linear.weight.device
        dtype = self.align_layer.linear.weight.dtype

        # NOTE: Do NOT call self.ts_encoder.to(device) during forward pass —
        # it can break FSDP's parameter management. In FSDP mode, all params
        # are already on the correct device. The lazy-move was only needed
        # for vLLM init where the encoder starts on CPU.
        if not self._hf_mode:
            enc_device = next(self.ts_encoder.parameters()).device
            if enc_device != device:
                self.ts_encoder = self.ts_encoder.to(device=device)

        ts = ts.to(device=device, dtype=dtype)

        if self._encoder_type == "toto":
            from toto.data.util.dataset import MaskedTimeseries

            inputs = MaskedTimeseries(
                series=ts,
                padding_mask=torch.ones_like(ts, dtype=torch.bool),
                id_mask=torch.zeros_like(ts),
                timestamp_seconds=torch.zeros(B, C, T, device=device, dtype=dtype),
                time_interval_seconds=torch.full((C,), 0.1, device=device, dtype=dtype),
            )
            raw = self.ts_encoder.get_embeddings(
                inputs.series, inputs.padding_mask, inputs.id_mask
            )
            # raw: [B, C, N, P] -> flatten to [B, C, N*P]
            flat = raw.contiguous().view(B, C, raw.shape[2] * raw.shape[3])

        elif self._encoder_type == "teleencoder":
            raw, _ = self.ts_encoder(ts)  # [B, C, S, d_model]
            # Flatten to [B, C, S*d_model]
            flat = raw.contiguous().view(B, C, raw.shape[2] * raw.shape[3])

        aligned = self.align_layer(flat)

        if _TSLLM_DEBUG and _FWD_DIAG_COUNT <= 1:
            with torch.no_grad():
                print(f"[DIAG _encode_timeseries] encoder={self._encoder_type}, "
                      f"input ts: {ts.shape}, raw: {raw.shape}, "
                      f"flat: {flat.shape}, aligned: {aligned.shape}")
                print(f"  raw mean={raw.float().mean():.4f}, std={raw.float().std():.4f}, "
                      f"norm={raw.float().norm():.4f}")
                print(f"  aligned mean={aligned.float().mean():.4f}, std={aligned.float().std():.4f}, "
                      f"norm={aligned.float().norm():.4f}")
                if B > 1:
                    diff = (aligned[0] - aligned[1]).float().norm()
                    print(f"  sample[0] vs sample[1] embedding diff norm: {diff:.4f}")

        return aligned
