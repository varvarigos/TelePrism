"""
vLLM plugin for TSLLMModel — multimodal processor, model registration, and plugin entry point.

This module is guarded by _VLLM_AVAILABLE so VERL FSDP workers (which run in
a separate env without vLLM) can safely import modeling_tsllm.py without
triggering vLLM imports.
"""

from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from teleprism.models.ts_llm.tsllm_causal.modeling_tsllm import TSLLMForCausalLM, _VLLM_AVAILABLE

__all__ = ["TSLLMForCausalLM", "register_teleprism_model"]

if _VLLM_AVAILABLE:
    from vllm.model_executor.models import ModelRegistry
    from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargsItems
    from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig
    from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser, ProcessorBatchItems
    from vllm.multimodal.processing import (
        BaseDummyInputsBuilder,
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdate,
    )

    class TSLLMProcessingInfo(BaseProcessingInfo):
        def get_hf_processor(self, **kwargs):
            return self.ctx.get_tokenizer()

        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"timeseries": None}

        def get_num_ts_tokens(self, *, num_channels: int = 18) -> int:
            """1 begin + num_channels TS + 1 end = 20 total."""
            return 1 + num_channels + 1

    class TSLLMDummyInputsBuilder(BaseDummyInputsBuilder):
        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            return "<|begin_of_TS|><|end_of_TS|>" * mm_counts.get("timeseries", 0)

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Optional[Mapping[str, object]] = None,
        ) -> MultiModalDataDict:
            n = mm_counts.get("timeseries", 0)
            return {} if n == 0 else {"timeseries": torch.randn(n, 18, 128)}

    class TSLLMMultiModalProcessor(BaseMultiModalProcessor):
        def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs=None):
            tokenizer = self.info.get_hf_processor()
            result = tokenizer(text=prompt, return_tensors="pt", truncation=False)
            ts = mm_data.get("timeseries") or mm_data.get("timeseriess")
            if ts is not None:
                result["timeseries"] = ts
            return result

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            return {"timeseries": MultiModalFieldConfig.batched("timeseries")}

        def _hf_processor_applies_updates(
            self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        ) -> bool:
            return False

        def _get_prompt_updates(
            self,
            mm_items: MultiModalKwargsItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            tokenizer = self.info.get_tokenizer()
            begin_id = tokenizer.convert_tokens_to_ids("<|begin_of_TS|>")
            end_id = tokenizer.convert_tokens_to_ids("<|end_of_TS|>")
            num_slots = self.info.get_num_ts_tokens(num_channels=18)
            return [
                PromptReplacement(
                    modality="timeseries",
                    target=[begin_id, end_id],
                    replacement=lambda item_idx: [begin_id] * num_slots,
                ),
            ]

    @MULTIMODAL_REGISTRY.register_processor(
        TSLLMMultiModalProcessor,
        info=TSLLMProcessingInfo,
        dummy_inputs=TSLLMDummyInputsBuilder,
    )
    class TSLLMModel(TSLLMForCausalLM):
        """Registered entry point for vLLM."""
        pass

    ModelRegistry.register_model("TSLLMModel", TSLLMModel)

    class TimeseriesProcessorItems(ProcessorBatchItems[torch.Tensor]):
        """Wraps a batch of [C, T] tensors as a vLLM multi-modal data item."""
        def __init__(self, data: list[torch.Tensor] | None) -> None:
            super().__init__(data, modality="timeseries")

    def _parse_timeseries_data(data) -> "TimeseriesProcessorItems":
        if data is None or (isinstance(data, list) and len(data) == 0):
            return TimeseriesProcessorItems([])
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.copy()).float()
        elif isinstance(data, list):
            data = torch.stack(
                [torch.from_numpy(d.copy()).float() if isinstance(d, np.ndarray) else d.float()
                 for d in data]
            )
        if data.dim() == 2:
            data = data.unsqueeze(0)
        return TimeseriesProcessorItems(list(data))

    _orig_get_subparsers = MultiModalDataParser._get_subparsers

    def _patched_get_subparsers(self):
        sp = _orig_get_subparsers(self)
        sp["timeseries"] = _parse_timeseries_data
        return sp

    MultiModalDataParser._get_subparsers = _patched_get_subparsers

    __all__ += [
        "TSLLMModel", "TSLLMProcessingInfo",
        "TSLLMDummyInputsBuilder", "TSLLMMultiModalProcessor",
        "TimeseriesProcessorItems",
    ]


def register_teleprism_model() -> None:
    """vLLM general plugin entry point.

    Called by vLLM's load_general_plugins() in every process (main HTTP-server,
    EngineCoreProc, WorkerProc). Importing this module triggers the
    ``if _VLLM_AVAILABLE:`` block which registers TSLLMModel in ModelRegistry.

    Registered in pyproject.toml:
        [project.entry-points."vllm.general_plugins"]
        teleprism = "teleprism.models.ts_llm.tsllm_causal.vllm_plugin:register_teleprism_model"
    """
    # Bypass ShmObjectStoreReceiverCache (deadlocks in VERL setup).
    # TSLLMModel passes timeseries as raw float tensors and does not need
    # the SHM object store. Returning None skips SHM cache setup.
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY as _mm_reg
        _mm_reg.worker_receiver_cache_from_config = lambda vllm_cfg, lock: None
    except Exception:
        pass
