"""
Time-Series vLLM Rollout for VERL GRPO training.

Provides TSVllmReplica (and its TSVllmHttpServer) that extend VERL's rollout
classes to support the TSLLMModel multimodal architecture.

Design: ALL vLLM class definitions are deferred inside _ts_replica_factory()
so that importing this module in the main process / FSDP-env does NOT trigger
the vLLM import chain (which requires _sqlite3 and is unavailable there).
The factory is only called when get_rollout_replica_class("ts_vllm") is
invoked — which happens inside a Ray actor that runs in the correct vLLM env.

Usage in the VERL launcher (run_grpo_training.sh):
    actor_rollout_ref.rollout.name=ts_vllm
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.timeseries=1
"""

import os

# Source file locations (resolved at import time; no vLLM needed).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_HERE, "../.."))  # the `teleprism` package dir (./src)
_SRC_DIR = os.path.dirname(_PKG_DIR)                      # repo root (fallback added to sys.path)
_VLLM_DIR = os.path.join(_PKG_DIR, "models", "ts_llm", "tsllm_causal")
_CONFIG_SRC = os.path.join(_VLLM_DIR, "config.json")
_TOKENIZER_SRC = os.path.join(_VLLM_DIR, "tokenizer.json")

# Lightweight registry imports (no vLLM chain triggered here).
from verl.workers.rollout.replica import RolloutReplicaRegistry
from verl.workers.rollout.base import _ROLLOUT_REGISTRY


def _ts_replica_factory():
    """Build and return TSVllmReplica (imports vLLM only at call time)."""
    import shutil

    import ray

    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        vLLMHttpServer,
        vLLMReplica,
    )

    # Capture module-level constants in closure
    _config_src = _CONFIG_SRC
    _tokenizer_src = _TOKENIZER_SRC
    _repo_root = _SRC_DIR

    class TSVllmHttpServer(vLLMHttpServer):
        """vLLM HTTP server with TSLLM multimodal (timeseries) support."""

        def __init__(self, *args, **kwargs):
            import sys

            # With baked weights (weights_baked=True in config.json), DS
            # checkpoint loading is automatically skipped. No env var needed.

            # Ensure repo root is importable in this process and subprocesses
            if _repo_root not in sys.path:
                sys.path.insert(0, _repo_root)
            existing = os.environ.get("PYTHONPATH", "")
            if _repo_root not in existing.split(os.pathsep):
                os.environ["PYTHONPATH"] = (
                    f"{_repo_root}{os.pathsep}{existing}" if existing else _repo_root
                )

            # TSLLMModel registration is handled automatically by the
            # vllm.general_plugins entry point in pyproject.toml.
            super().__init__(*args, **kwargs)

        async def generate(self, *args, timeseries_data=None, **kwargs):
            """Override to inject timeseries into multi_modal_data."""
            if timeseries_data is not None:
                # Intercept: add timeseries to image_data slot isn't right,
                # so we override the full method to inject it properly.
                from vllm import SamplingParams
                from vllm.inputs import TokensPrompt
                from verl.workers.rollout.vllm_rollout.vllm_async_server import _qwen2_5_vl_dedup_image_tokens
                from verl.workers.rollout.vllm_rollout.utils import (
                    VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH,
                )
                from vllm.lora.request import LoRARequest
                from verl.workers.rollout.replica import TokenOutput

                # Extract positional/keyword args matching parent signature
                prompt_ids = kwargs.get("prompt_ids") or args[0]
                sampling_params = kwargs.get("sampling_params") or args[1]
                request_id = kwargs.get("request_id") or args[2]
                image_data = kwargs.get("image_data")
                video_data = kwargs.get("video_data")
                priority = kwargs.get("priority", 0)

                max_possible_tokens = self.config.max_model_len - len(prompt_ids)
                if max_possible_tokens < 0:
                    raise ValueError(
                        f"Prompt length ({len(prompt_ids)}) exceeds max_model_len ({self.config.max_model_len})."
                    )
                if "max_tokens" in sampling_params:
                    max_tokens = sampling_params.pop("max_tokens")
                elif "max_new_tokens" in sampling_params:
                    max_tokens = sampling_params.pop("max_new_tokens")
                else:
                    max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)
                max_tokens = max(0, min(max_tokens, max_possible_tokens))

                sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
                sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
                sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
                prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)

                multi_modal_data = {}
                if image_data is not None:
                    multi_modal_data["image"] = image_data
                if video_data is not None:
                    multi_modal_data["video"] = video_data
                multi_modal_data["timeseries"] = timeseries_data

                prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

                lora_request = None
                if (
                    self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
                ) and not self.model_config.lora.get("merge", False):
                    lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
                    if lora_loaded:
                        lora_request = LoRARequest(
                            lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                        )

                generator = self.engine.generate(
                    prompt=prompt, sampling_params=sampling_params,
                    request_id=request_id, lora_request=lora_request, priority=priority,
                )
                final_res = None
                async for output in generator:
                    final_res = output
                assert final_res is not None

                token_ids = final_res.outputs[0].token_ids
                log_probs = None
                if sampling_params.logprobs is not None:
                    log_probs = [lp[token_ids[i]].logprob for i, lp in enumerate(final_res.outputs[0].logprobs)]

                routed_experts = None
                if self.config.enable_rollout_routing_replay:
                    routed_experts = final_res.outputs[0].routed_experts

                finish_reason = final_res.outputs[0].finish_reason
                stop_reason = "aborted" if finish_reason == "abort" else (
                    "completed" if finish_reason in ("stop", "length") else finish_reason
                )
                num_preempted = getattr(final_res.outputs[0], "num_preempted", None)

                return TokenOutput(
                    token_ids=token_ids, log_probs=log_probs,
                    routed_experts=routed_experts, stop_reason=stop_reason,
                    num_preempted=num_preempted,
                )
            else:
                return await super().generate(*args, **kwargs)

    class TSVllmReplica(vLLMReplica):
        """vLLM replica that boots TSVllmHttpServer workers."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Replace the default server class with our TS-aware subclass
            self.server_class = ray.remote(TSVllmHttpServer)

    return TSVllmReplica


# Registry entries (set at module import time, no vLLM needed).
RolloutReplicaRegistry.register("ts_vllm", _ts_replica_factory)

# FSDP workers use the standard vLLM ServerAdapter; TS logic lives in TSVllmReplica.
_ROLLOUT_REGISTRY[("ts_vllm", "async")] = "verl.workers.rollout.vllm_rollout.ServerAdapter"
