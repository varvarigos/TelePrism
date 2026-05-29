"""
Custom vLLM worker extension for TSLLM.

Extends VERL's vLLMColocateWorkerExtension to handle LoRA weight sync
without requiring vLLM's native LoRA API (--enable-lora).

When VERL sets base_sync_done=True (after the first full weight sync),
it sends only LoRA adapter weights and expects the worker to use
add_lora()/remove_lora().  Our TSLLMModel doesn't support vLLM's native
LoRA infrastructure, so this extension:

  1. Caches "clean" base weights (CPU) after the first full load.
  2. On subsequent LoRA-only updates: restores base weights, then merges
     the new LoRA deltas in-place.

This avoids the add_lora/remove_lora API entirely while still being
efficient (only ~30 MiB of LoRA params are transferred per sync).

Configured via engine_kwargs in run_grpo_training.sh:
    +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=\
        teleprism.rl.training.worker_extension.TSColocateWorkerExtension
"""

import gc
import logging

import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

logger = logging.getLogger(__name__)


def _weight_checksums(model, prefix_groups=("ts_encoder.", "align_layer.", "language_model.")):
    """Compute per-group weight checksums for diagnostic logging."""
    sums = {}
    for group in prefix_groups:
        total = 0.0
        count = 0
        for name, param in model.named_parameters():
            if name.startswith(group):
                total += param.data.float().sum().item()
                count += 1
        if count > 0:
            sums[group.rstrip(".")] = (count, total)
    return sums


def _log_checksums(label, model):
    """Log weight checksums for verification."""
    checksums = _weight_checksums(model)
    parts = [f"{k}: {n} params, sum={s:.4f}" for k, (n, s) in checksums.items()]
    logger.warning(f"[DIAG] {label} | " + " | ".join(parts))


class TSColocateWorkerExtension(vLLMColocateWorkerExtension):
    """Worker extension that handles LoRA weight sync without vLLM's LoRA API.

    Overrides update_weights_from_ipc and _update_weights to avoid
    add_lora/remove_lora calls, which require --enable-lora.
    """

    def _ensure_ts_state(self):
        """Lazily initialize TS-specific state (mixin __init__ may not run)."""
        if not hasattr(self, "_base_weight_cache"):
            self._base_weight_cache: dict[str, torch.Tensor] = {}
        if not hasattr(self, "_first_sync_done"):
            self._first_sync_done = False

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Override to skip remove_lora() call when base_sync_done=True."""
        import zmq
        from vllm.platforms import current_platform
        from verl.utils.torch_functional import get_torch_device

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        self._ensure_ts_state()

        # NOTE: We intentionally skip the parent's `self.remove_lora()` call here.
        # That call requires --enable-lora which our model doesn't support.

        # build communication buffer
        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(self._get_zmq_handle())

        comm_metadata = socket.recv_pyobj()
        buffer, shm = None, None
        if not use_shm:
            from verl.workers.rollout.vllm_rollout.utils import rebuild_ipc
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            from verl.workers.rollout.vllm_rollout.utils import rebuild_shared_memory
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        socket.send(b"")

        # receive bucket and update weights
        while True:
            metadata = socket.recv_pyobj()
            weights, tensor = [], None
            for name, meta in metadata["bucket_meta"].items():
                shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                if not use_shm:
                    tensor = tensor.clone()
                else:
                    tensor = tensor.to(self.device)
                weights.append((name, tensor))
            get_torch_device().synchronize()
            socket.send(b"")
            self._update_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done)
            del weights, tensor
            if metadata["is_last"]:
                break

        # Cache base weights AFTER all buckets are loaded (not per-bucket).
        # With load_format=dummy, vLLM starts with random weights.  The IPC
        # sync overwrites them bucket-by-bucket.  We must wait until ALL
        # buckets have been applied before snapshotting the base weights,
        # otherwise the cache would contain random (dummy) values for any
        # parameters that hadn't been loaded yet.
        if not base_sync_done:
            self._cache_base_weights()
            self._first_sync_done = True
            _log_checksums("after full sync (base_sync_done=False)", self.model_runner.model)
        else:
            _log_checksums("after LoRA sync (base_sync_done=True)", self.model_runner.model)

        # clean up
        socket.close()
        del buffer
        if shm is not None:
            shm.close()
            del shm
        get_torch_device().synchronize()
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        """Override to merge LoRA deltas directly instead of using add_lora."""
        if peft_config and base_sync_done:
            # LoRA-only update: merge deltas into model weights
            self._apply_lora_to_model(weights, peft_config)
            logger.info(f"TS worker: merged {len(weights)} LoRA params into model")
        else:
            # First full sync: load all weights normally, then cache base weights
            from verl.workers.rollout.vllm_rollout.utils import is_fp8_model, load_quanted_weights
            if is_fp8_model(self.model_runner.vllm_config):
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (TS async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("TS worker: loading full weights (first sync)")
                self.model_runner.model.load_weights(weights)

    def _cache_base_weights(self):
        """Cache current model weights (CPU) as the 'clean base' for LoRA merges.

        Caches ALL parameters — language_model, ts_encoder, and align_layer.
        Only language_model weights are restored from cache during LoRA merges;
        ts_encoder/align_layer are trained as full params and receive updates
        directly from FSDP (not restored from cache).
        """
        model = self.model_runner.model
        cache = {}
        for name, param in model.named_parameters():
            # Store on CPU to save GPU memory
            cache[name] = param.data.detach().cpu().clone()
        self._base_weight_cache = cache
        logger.info(
            f"TS worker: cached {len(cache)} base weight tensors "
            f"(ts_encoder: {sum(1 for k in cache if k.startswith('ts_encoder.'))}, "
            f"align_layer: {sum(1 for k in cache if k.startswith('align_layer.'))}, "
            f"language_model: {sum(1 for k in cache if k.startswith('language_model.'))})"
        )

    def _get_tp_info(self) -> tuple[int, int]:
        """Return (tp_rank, tp_size) for this vLLM worker.

        With TP>1 each vLLM worker holds a *shard* of every weight tensor.
        We must slice the LoRA delta to match the shard before applying it.
        """
        try:
            tp_size = self.model_runner.vllm_config.parallel_config.tensor_parallel_size
            # Within a single vLLM replica the workers are numbered 0..TP-1,
            # which maps to local_rank.
            tp_rank = getattr(self, "local_rank", 0) % tp_size
        except Exception:
            tp_size = 1
            tp_rank = 0
        return tp_rank, tp_size

    def _apply_lora_to_model(self, lora_weights: list[tuple[str, torch.Tensor]], peft_config: dict):
        """Merge LoRA A/B weights into model parameters.

        1. Restore base weights from cache (undo previous LoRA merge).
        2. Parse LoRA A/B pairs from the PEFT state dict.
        3. Compute full delta = lora_B @ lora_A * scaling.
        4. Slice delta to match this worker's TP shard, then apply in-place.
           • Column-parallel weights (qkv_proj, gate_up_proj): slice rows.
           • Row-parallel weights (o_proj, down_proj): slice columns.

        Handles both TP=1 (no slicing) and TP>1 (shard-aware slicing).
        """
        model = self.model_runner.model
        model_params = dict(model.named_parameters())
        tp_rank, tp_size = self._get_tp_info()

        # Step 1: Restore base weights (undo previous LoRA merge)
        # Skip ts_encoder and align_layer — they are trained as full parameters
        # (not via LoRA) and receive their updated weights directly from FSDP.
        # Restoring them from cache would discard the training updates.
        restored = 0
        ts_skipped = 0
        for name, base_tensor in self._base_weight_cache.items():
            if name.startswith("ts_encoder.") or name.startswith("align_layer."):
                ts_skipped += 1
                continue
            if name in model_params:
                model_params[name].data.copy_(base_tensor.to(
                    device=model_params[name].device,
                    dtype=model_params[name].dtype,
                ))
                restored += 1

        # Step 1b: Load ts_encoder/align_layer params directly (trained as full params)
        ts_loaded = 0
        lora_dict = dict(lora_weights)
        # Debug: log first few ts-related param names to diagnose prefix issues
        ts_names = [n for n in lora_dict if "ts_encoder" in n or "align_layer" in n]
        if ts_names:
            logger.warning(f"TS worker: received {len(ts_names)} ts/align params, first 3 names: {ts_names[:3]}")
        for name, tensor in list(lora_dict.items()):
            # Strip PEFT and FSDP prefixes to get clean param name
            clean = name
            for prefix in ("base_model.model.", "model."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
            clean = clean.replace("_fsdp_wrapped_module.", "")
            if clean.startswith("ts_encoder.") or clean.startswith("align_layer."):
                # Try both the cleaned name and original name for lookup
                param = model_params.get(clean)
                if param is None:
                    param = model_params.get(name)
                if param is not None:
                    t = tensor.to(device=param.device, dtype=param.dtype)
                    if param.shape != t.shape:
                        t = t.reshape(param.shape)
                    param.data.copy_(t)
                    ts_loaded += 1
                del lora_dict[name]  # remove so LoRA parsing skips it

        # Step 2: Parse LoRA A/B pairs
        lora_a = {}  # module_path → tensor
        lora_b = {}  # module_path → tensor

        for name, tensor in lora_dict.items():
            if ".lora_A." in name:
                mod_path = name.split(".lora_A.")[0]
                lora_a[mod_path] = tensor
            elif ".lora_B." in name:
                mod_path = name.split(".lora_B.")[0]
                lora_b[mod_path] = tensor

        if not lora_a:
            logger.warning("TS worker: no LoRA A/B pairs found in update!")
            return

        # Infer scaling
        sample_a = next(iter(lora_a.values()))
        lora_r = sample_a.shape[0]
        lora_alpha = peft_config.get("lora_alpha", 16) if isinstance(peft_config, dict) else getattr(peft_config, "lora_alpha", 16)
        scaling = float(lora_alpha) / lora_r

        # ── helpers for TP-aware delta application ──────────────────────────
        def _apply_col_parallel(p: torch.Tensor, delta: torch.Tensor):
            """Apply a column-parallel delta (rows split across TP workers)."""
            if tp_size > 1:
                total_rows = delta.shape[0]
                chunk = total_rows // tp_size
                delta = delta[tp_rank * chunk:(tp_rank + 1) * chunk]
            p.data += delta.to(device=p.device, dtype=p.dtype)

        def _apply_row_parallel(p: torch.Tensor, delta: torch.Tensor):
            """Apply a row-parallel delta (columns split across TP workers)."""
            if tp_size > 1:
                total_cols = delta.shape[1]
                chunk = total_cols // tp_size
                delta = delta[:, tp_rank * chunk:(tp_rank + 1) * chunk]
            p.data += delta.to(device=p.device, dtype=p.dtype)

        # Step 3 & 4: Compute and apply deltas
        # PEFT names use unfused projections (q_proj, k_proj, v_proj, gate_proj, up_proj)
        # vLLM uses fused projections (qkv_proj, gate_up_proj)
        merged = 0
        config = model.config
        num_layers = config.num_hidden_layers

        for i in range(num_layers):
            prefix = f"base_model.model.language_model.model.layers.{i}"
            attn = f"language_model.model.layers.{i}.self_attn"
            mlp = f"language_model.model.layers.{i}.mlp"

            # ── qkv_proj: q + k + v fused — column-parallel (rows split) ──
            qa = lora_a.get(f"{prefix}.self_attn.q_proj")
            qb = lora_b.get(f"{prefix}.self_attn.q_proj")
            ka = lora_a.get(f"{prefix}.self_attn.k_proj")
            kb = lora_b.get(f"{prefix}.self_attn.k_proj")
            va = lora_a.get(f"{prefix}.self_attn.v_proj")
            vb = lora_b.get(f"{prefix}.self_attn.v_proj")

            if qa is not None and ka is not None and va is not None:
                p = model_params.get(f"{attn}.qkv_proj.weight")
                if p is not None:
                    dq = (qb @ qa) * scaling
                    dk = (kb @ ka) * scaling
                    dv = (vb @ va) * scaling
                    delta = torch.cat([dq, dk, dv], dim=0)
                    _apply_col_parallel(p, delta)
                    merged += 3

            # ── o_proj — row-parallel (columns split) ──
            oa = lora_a.get(f"{prefix}.self_attn.o_proj")
            ob = lora_b.get(f"{prefix}.self_attn.o_proj")
            if oa is not None:
                p = model_params.get(f"{attn}.o_proj.weight")
                if p is not None:
                    _apply_row_parallel(p, (ob @ oa) * scaling)
                    merged += 1

            # ── gate_up_proj: gate + up fused — column-parallel ──
            ga = lora_a.get(f"{prefix}.mlp.gate_proj")
            gb = lora_b.get(f"{prefix}.mlp.gate_proj")
            ua = lora_a.get(f"{prefix}.mlp.up_proj")
            ub = lora_b.get(f"{prefix}.mlp.up_proj")

            if ga is not None and ua is not None:
                p = model_params.get(f"{mlp}.gate_up_proj.weight")
                if p is not None:
                    dg = (gb @ ga) * scaling
                    du = (ub @ ua) * scaling
                    delta = torch.cat([dg, du], dim=0)
                    _apply_col_parallel(p, delta)
                    merged += 2

            # ── down_proj — row-parallel (columns split) ──
            da = lora_a.get(f"{prefix}.mlp.down_proj")
            db = lora_b.get(f"{prefix}.mlp.down_proj")
            if da is not None:
                p = model_params.get(f"{mlp}.down_proj.weight")
                if p is not None:
                    _apply_row_parallel(p, (db @ da) * scaling)
                    merged += 1

        logger.info(
            f"TS worker: restored {restored} base weights, loaded {ts_loaded} ts/align params, "
            f"merged {merged} LoRA modules "
            f"(r={lora_r}, alpha={lora_alpha}, scaling={scaling:.4f}, "
            f"tp_rank={tp_rank}/{tp_size})"
        )
