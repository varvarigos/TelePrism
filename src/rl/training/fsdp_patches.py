"""
Monkey-patches for TSLLM GRPO training in VERL.

Three patches:

1. get_peft_model  →  After PEFT freezes all non-LoRA params, re-enable
   requires_grad on ts_encoder and align_layer so they are trainable,
   and cast them to bf16 for gradient dtype uniformity.

2. get_fsdp_wrap_policy  →  The default LoRA lambda only wraps leaf modules
   with a `.weight` attribute.  Some TS encoder norm layers (e.g. TOTO's
   `.scale`) don't match, falling through to a parent FSDP flat-param group
   that mixes requires_grad=True/False, which crashes with
   use_orig_params=False.  We replace the lambda to match ANY leaf module
   that has at least one trainable parameter.  This works for both TOTO
   and TeleEncoder.

3. _get_grad_norm  →  FSDP's clip_grad_norm_ requires uniform gradient dtype.
   Custom multimodal models (TSLLM with TOTO or TeleEncoder) can produce
   mixed bf16/fp32 gradients.  We patch the PyTorch internal to cast fp32
   grads to bf16 before the check.
"""
import functools
import sys
import builtins
import torch

_UNFREEZE_PATTERNS = ("ts_encoder", "align_layer")
_TARGET = "verl.workers.fsdp_workers"
_patched = False
_original_import = builtins.__import__


def _do_patch():
    global _patched
    if _patched:
        return

    mod = sys.modules.get(_TARGET)
    if mod is None or not hasattr(mod, "get_peft_model") or not hasattr(mod, "get_fsdp_wrap_policy"):
        return  # Not ready yet — module still importing

    # get_peft_model patch
    _orig_get_peft = mod.get_peft_model

    def _wrapped_get_peft_model(model, config, *args, **kwargs):
        import torch

        # VERL passes target_modules as a comma-separated string; PEFT needs a list
        if hasattr(config, 'target_modules') and isinstance(config.target_modules, str) and ',' in config.target_modules:
            config.target_modules = [m.strip() for m in config.target_modules.split(',')]

        result = _orig_get_peft(model, config, *args, **kwargs)

        # Check train_encoder config
        import os, yaml
        _cfg_path = os.environ.get("GRPO_TSLLM_CONFIG")
        _train_enc = True
        if _cfg_path and os.path.isfile(_cfg_path):
            with open(_cfg_path) as _f:
                _train_enc = yaml.safe_load(_f).get("training", {}).get("train_encoder", True)

        n = 0
        if _train_enc:
            for name, p in result.named_parameters():
                if any(pat in name for pat in _UNFREEZE_PATTERNS) and not p.requires_grad:
                    p.requires_grad_(True)
                    n += 1
        else:
            print("[fsdp_patches] train_encoder=false → ts_encoder/align_layer stay FROZEN")

        # Cast ts_encoder/align_layer to bf16 to match language_model dtype.
        cast = 0
        for name, p in result.named_parameters():
            if any(pat in name for pat in _UNFREEZE_PATTERNS) and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
                cast += 1

        # FSDP requires all params to be ≥1D. Reshape any scalar params
        # (e.g. TeleEncoder's gnn_alpha, gnn_tau) to 1D tensors.
        scalars = 0
        for name, p in result.named_parameters():
            if p.dim() == 0:
                p.data = p.data.unsqueeze(0)
                scalars += 1

        print(f"[fsdp_patches] After get_peft_model: unfroze {n} ts_encoder/align_layer params, cast {cast} to bf16, reshaped {scalars} scalars to 1D")

        # Disable encoder stochasticity during GRPO training.
        #
        # The TeleEncoder has two training-only stochastic operations:
        #   1. Dropout (p=0.1) — randomly zeros activations in train() mode
        #   2. Router jitter (std=0.01) — adds Gaussian noise to MoE router
        #      logits in train() mode, causing different top-k experts to fire
        #
        # In GRPO, rollout and compute_log_prob run in eval() mode (no
        # stochasticity) while update_policy runs in train() mode (stochastic).
        # The log_prob ratio exp(new - old) then picks up this stochasticity
        # as if it were a policy change, and gradients push the encoder toward
        # "matching" random routing/dropout — destroying it over time.
        #
        # MoE routing is especially bad: jitter can flip which experts are
        # selected, producing discretely different outputs for the same input.
        #
        # Fix (safe for FSDP/NCCL): disable the operations at the parameter
        # level so they're inactive regardless of train/eval mode.  We do this
        # by zeroing dropout probabilities and jitter std on encoder submodules.
        # This is idempotent and doesn't touch module.training, so it won't
        # interfere with FSDP's expected behavior.
        if _train_enc:
            n_dropouts = 0
            n_jitters = 0
            n_dense_warmup = 0
            n_sparse_inf = 0
            for name, module in result.named_modules():
                if "ts_encoder" not in name and "align_layer" not in name:
                    continue
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
                    n_dropouts += 1
                if hasattr(module, "router_jitter_std"):
                    try:
                        module.router_jitter_std = 0.0
                        n_jitters += 1
                    except Exception:
                        pass
                # TeleEncoder has a `force_dense` flag that activates DENSE MoE
                # routing during training when `current_epoch < dense_routing_
                # warmup_epochs` (default 3).  Nothing calls set_current_epoch()
                # in RL, so current_epoch stays 0 and update_policy (train mode)
                # uses DENSE routing while rollout (eval mode) uses SPARSE top-k.
                # This produces different encoder outputs in the two passes,
                # making the GRPO log-prob ratio meaningless.  Fix: bump
                # current_epoch past the warmup so SPARSE is used in both modes.
                if hasattr(module, "current_epoch") and hasattr(module, "dense_routing_warmup_epochs"):
                    try:
                        module.current_epoch = max(
                            int(getattr(module, "dense_routing_warmup_epochs", 0)) + 1,
                            int(module.current_epoch) + 1,
                        )
                        n_dense_warmup += 1
                    except Exception:
                        pass
                # Disable `sparse_inference` because it makes the MoE forward
                # pass execute a DIFFERENT set of experts per sample (those
                # with non-zero gate weight).  Under FSDP, different ranks
                # have different inputs and therefore skip different experts,
                # so the gradient allgather hangs (one rank tries to gather
                # gradients for an expert another rank never executed).  With
                # sparse_inference=False, all experts run on every input
                # (multiplied by zero gate weight when not selected), so the
                # computation graph is identical across ranks.
                if hasattr(module, "sparse_inference"):
                    try:
                        if module.sparse_inference:
                            module.sparse_inference = False
                            n_sparse_inf += 1
                    except Exception:
                        pass
            print(f"[fsdp_patches] Disabled encoder stochasticity: "
                  f"{n_dropouts} dropouts → p=0, {n_jitters} MoE routers → jitter=0, "
                  f"{n_dense_warmup} encoders → past dense warmup, "
                  f"{n_sparse_inf} MoE → sparse_inference=False (avoids FSDP allgather hang)")

        return result

    mod.get_peft_model = _wrapped_get_peft_model

    # get_fsdp_wrap_policy patch
    _orig_get_wrap = mod.get_fsdp_wrap_policy

    def _wrapped_get_fsdp_wrap_policy(module, config=None, is_lora=False):
        policy = _orig_get_wrap(module, config=config, is_lora=is_lora)

        if not is_lora:
            return policy

        # Replace the LoRA lambda policy with a broader one that catches
        # ANY leaf module with trainable params (not just those with .weight)
        from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

        # Collect MHA children to exclude from wrapping
        _mha_children = set()
        for m in module.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                for child in m.modules():
                    if child is not m:
                        _mha_children.add(child)

        def _broad_lambda(module):
            # Skip nn.MultiheadAttention and its children — MHA accesses
            # out_proj.weight directly, bypassing module forward().
            if isinstance(module, torch.nn.MultiheadAttention):
                return False
            if module in _mha_children:
                return False

            direct_params = set(module.parameters()) - {
                p for c in module.children() for p in c.parameters()
            }
            if any(p.requires_grad for p in direct_params):
                return True
            if len(list(module.children())) == 0:
                return any(p.requires_grad for p in module.parameters())
            return False

        broad_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=_broad_lambda)

        # Rebuild the _or_policy keeping any size/transformer policies
        # from the original, but swapping in our broader lambda
        if policy is not None and hasattr(policy, "keywords") and "policies" in policy.keywords:
            orig_policies = list(policy.keywords["policies"])
            new_policies = []
            for p in orig_policies:
                # Detect the old lambda_auto_wrap_policy and replace it
                if hasattr(p, "func") and p.func.__name__ == "lambda_auto_wrap_policy":
                    new_policies.append(broad_policy)
                else:
                    new_policies.append(p)
            result = functools.partial(_or_policy, policies=new_policies)
        else:
            # Fallback: just use our broad policy
            result = broad_policy

        print(f"[patches] Broadened FSDP wrap lambda (catches TS encoder params)")
        return result

    mod.get_fsdp_wrap_policy = _wrapped_get_fsdp_wrap_policy

    _patched = True
    builtins.__import__ = _original_import
    print("[patches] Patched fsdp_workers (get_peft_model + get_fsdp_wrap_policy)")


def _hooked_import(name, *args, **kwargs):
    result = _original_import(name, *args, **kwargs)
    if not _patched:
        _do_patch()
    return result


# If already fully loaded, patch now; otherwise install hook
_do_patch()
if not _patched:
    builtins.__import__ = _hooked_import
    print("[fsdp_patches] Installed import hook for fsdp_workers patching")


# We patch _get_grad_norm to cast fp32 grads → bf16 before the dtype check.
from torch.distributed.fsdp import fully_sharded_data_parallel as _fsdp_mod

_orig_get_grad_norm = _fsdp_mod._get_grad_norm


def _mixed_dtype_safe_get_grad_norm(params, *args, **kwargs):
    params_list = list(params)
    # Cast any fp32 gradients to bf16 for dtype uniformity
    for p in params_list:
        if p.grad is not None and p.grad.dtype == torch.float32:
            p.grad.data = p.grad.data.to(torch.bfloat16)
    return _orig_get_grad_norm(params_list, *args, **kwargs)


_fsdp_mod._get_grad_norm = _mixed_dtype_safe_get_grad_norm
print("[fsdp_patches] Patched FSDP _get_grad_norm for mixed-dtype gradient safety")


# ── Patch collect_lora_params to include ts_encoder/align_layer ──────────
# When base_sync_done=True, get_peft_model_state_dict only returns LoRA
# adapter weights.  ts_encoder and align_layer are trained as full params
# (not LoRA), so they are silently dropped from the weight sync to vLLM.
# This means the rollout engine keeps using STALE encoder weights from
# step 0, while the FSDP actor trains with updated encoder weights —
# a systematic mismatch that corrupts learning.
#
# Fix: after collecting LoRA params, also collect ts_encoder/align_layer
# params and include them in the sync payload.
import verl.utils.fsdp_utils as _fsdp_utils_mod

_TRAINABLE_FULL_PARAM_PREFIXES = ("ts_encoder.", "align_layer.")

_orig_collect_lora_params = _fsdp_utils_mod.collect_lora_params


def _patched_collect_lora_params(module, layered_summon, base_sync_done):
    """Wrapper that adds ts_encoder/align_layer to the LoRA-only sync."""
    from collections import OrderedDict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    lora_params = _orig_collect_lora_params(module, layered_summon, base_sync_done)

    # Only need to augment when base_sync_done=True (LoRA-only sync).
    # When base_sync_done=False, all params are already included.
    if not base_sync_done:
        return lora_params

    # Check if train_encoder is enabled
    import os, yaml
    _cfg_path = os.environ.get("GRPO_TSLLM_CONFIG")
    _train_enc = True
    if _cfg_path and os.path.isfile(_cfg_path):
        with open(_cfg_path) as _f:
            _train_enc = yaml.safe_load(_f).get("training", {}).get("train_encoder", True)

    if not _train_enc:
        return lora_params  # encoder is frozen, no need to sync

    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    inner = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model

    extra = OrderedDict()
    is_fsdp = _fsdp_utils_mod.fsdp_version(module) > 0

    if is_fsdp:
        with FSDP.summon_full_params(module, writeback=False):
            for name, param in inner.named_parameters():
                if any(name.startswith(pfx) or f".{pfx.rstrip('.')}" in name
                       for pfx in _TRAINABLE_FULL_PARAM_PREFIXES):
                    clean = name.replace("_fsdp_wrapped_module.", "")
                    if hasattr(param, "full_tensor"):
                        extra[clean] = param.full_tensor().detach().cpu()
                    else:
                        extra[clean] = param.detach().cpu()
    else:
        for name, param in inner.named_parameters():
            if any(name.startswith(pfx) or f".{pfx.rstrip('.')}" in name
                   for pfx in _TRAINABLE_FULL_PARAM_PREFIXES):
                clean = name.replace("_fsdp_wrapped_module.", "")
                extra[clean] = param.detach().cpu()

    if extra:
        lora_params.update(extra)
        print(f"[fsdp_patches] collect_lora_params: added {len(extra)} ts_encoder/align_layer "
              f"params to LoRA sync (base_sync_done={base_sync_done})")

    return lora_params


_fsdp_utils_mod.collect_lora_params = _patched_collect_lora_params
print("[fsdp_patches] Patched collect_lora_params to include ts_encoder/align_layer in weight sync")


# Note: the force-encoder-eval-mode fix is applied in the get_peft_model
# wrapper above. See the `_train_with_frozen_encoder_mode` override there.
# This disables:
#   1. Dropout in ts_encoder (dropout=0.1 in TeleEncoder config)
#   2. MoE router jitter (router_jitter_std=0.01 in config)
# Both only activate when module.training=True. Forcing eval mode on
# encoder submodules keeps them deterministic between rollout/compute_log_prob
# (eval) and update_policy (train) forward passes, so the GRPO log_prob
# ratio isn't corrupted by encoder stochasticity.


# Patch transformers.dynamic_module_utils.custom_object_save so that when the
# model is wrapped in PeftModelForCausalLM, HF inspects the *inner* model class
# (TSLLMForCausalLM in modeling_tsllm.py) rather than peft_model.py.
try:
    import transformers.dynamic_module_utils as _dmu

    _orig_custom_object_save = _dmu.custom_object_save

    def _tsllm_custom_object_save(obj, folder, config=None):
        # Unwrap PeftModel → inner TSLLMForCausalLM so inspect.getfile()
        # returns modeling_tsllm.py (no relative imports) instead of
        # peft_model.py (relative imports referencing peft/utils.py).
        inner = obj
        for _ in range(5):  # guard against deeply nested wrappers
            if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
                inner = inner.base_model.model
            else:
                break
        return _orig_custom_object_save(inner, folder, config=config)

    _dmu.custom_object_save = _tsllm_custom_object_save
    print("[fsdp_patches] Patched custom_object_save to unwrap PeftModel before HF file inspection")
except Exception as _e:
    print(f"[fsdp_patches] WARNING: could not patch custom_object_save: {_e}")


# ── Patch compute_data_metrics to add sequence-level advantage stats ──
# The default metric averages advantages per-token (length-weighted),
# which is misleading. Add a proper per-sequence mean.
try:
    from verl.trainer.ppo import metric_utils as _metric_mod
    _orig_compute_data_metrics = _metric_mod.compute_data_metrics

    def _tsllm_compute_data_metrics(batch, use_critic=False):
        import torch
        metrics = _orig_compute_data_metrics(batch, use_critic=use_critic)

        advantages = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"].bool()
        seq_adv = (advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)

        metrics["critic/seq_advantages/mean"] = seq_adv.mean().detach().item()
        metrics["critic/seq_advantages/std"] = seq_adv.std().detach().item()
        return metrics

    _metric_mod.compute_data_metrics = _tsllm_compute_data_metrics
    print("[fsdp_patches] Patched compute_data_metrics: added sequence-level advantage stats ✓")
except Exception as _e:
    print(f"[fsdp_patches] WARNING: could not patch compute_data_metrics: {_e}")
