"""
Time-Series Single-Turn Agent Loop for VERL GRPO training.

Subclasses VERL's SingleTurnAgentLoop to extract serialized timeseries
from the per-sample extra_info dict and pass it to vLLM as multi_modal_data.

The timeseries is stored in the parquet as:
    extra_info["timeseries"] = base64(pickle(numpy_array))  # shape [C, T]

It is passed to vLLM's generate() as:
    timeseries_data=[numpy_array]

vLLM maps this to:
    multi_modal_data={"timeseries": [numpy_array]}

which TSLLMMultiModalProcessor receives and passes to the TS encoder.
"""

import base64
import logging
import os
import pickle
from typing import Any
from uuid import uuid4

import numpy as np

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
    register,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput


# Monkey-patch AsyncLLMServerManager.generate to accept timeseries_data
# and forward it to the server actor.
@rollout_trace_op
async def _ts_generate(
    self,
    request_id,
    *,
    prompt_ids,
    sampling_params,
    image_data=None,
    video_data=None,
    timeseries_data=None,
) -> TokenOutput:
    from uuid import uuid4

    server = self._choose_server(request_id)
    output = await server.generate.remote(
        request_id=uuid4().hex,
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        image_data=image_data,
        video_data=video_data,
        timeseries_data=timeseries_data,
    )
    return output


AsyncLLMServerManager.generate = _ts_generate

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _scale_kpi_units(ts: np.ndarray) -> np.ndarray:
    """Apply unit scaling to match supervised training pipeline.

    Channel indices follow KPI_LIST order used in preprocessing:
      7=TX_Bytes, 8=RX_Bytes, 9=Estimated_UL_Buffer,
      15=UL_NumberOfPackets, 17=DL_NumberOfPackets
    """
    ts = ts.copy()
    ts[7, :] /= 1e6   # TX_Bytes → MB
    ts[8, :] /= 1e6   # RX_Bytes → MB
    ts[9, :] /= 1000   # Estimated_UL_Buffer → MB
    ts[15, :] /= 100   # UL_NumberOfPackets
    ts[17, :] /= 100   # DL_NumberOfPackets
    return ts


def _deserialize_timeseries(extra_info: dict) -> np.ndarray | None:
    """Deserialize a base64+pickle encoded timeseries array from extra_info.

    Args:
        extra_info: Per-sample extra_info dict from the parquet dataset.

    Returns:
        numpy array of shape [C, T] with KPI units scaled, or None if not present / malformed.
    """
    ts_b64 = extra_info.get("timeseries") if isinstance(extra_info, dict) else None
    if ts_b64 is None:
        return None
    try:
        ts = pickle.loads(base64.b64decode(ts_b64))
        return _scale_kpi_units(ts)
    except Exception as e:
        logger.warning(f"Failed to deserialize timeseries: {e}")
        return None


@register("ts_single_turn_agent")
class TSSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop that injects time-series data into the vLLM generate call.

    The prompt must already contain the <|begin_of_TS|> placeholder token(s) (added
    during data pre-processing).  vLLM's TSLLMMultiModalProcessor will expand the
    placeholder into the full set of TS embedding tokens and the TS encoder will
    process the raw numpy arrays passed via multi_modal_data["timeseries"].
    """

    _diag_counter = 0  # class-level counter for diagnostic logging

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # Extract images/videos from messages (inherited, does nothing if no vision data)
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # Extract and deserialize timeseries from extra_info
        extra_info = kwargs.get("extra_info", {})
        timeseries = _deserialize_timeseries(extra_info)
        timeseries_data = [timeseries] if timeseries is not None else None

        if timeseries_data is not None:
            multi_modal_data["timeseries"] = timeseries_data

        # DIAG: log first 2 samples (only when TSLLM_DEBUG=1)
        if os.environ.get("TSLLM_DEBUG", "0") == "1":
            TSSingleTurnAgentLoop._diag_counter += 1
            if TSSingleTurnAgentLoop._diag_counter <= 2:
                print(f"\n{'='*60}")
                print(f"[DIAG agent_loop.run] sample #{TSSingleTurnAgentLoop._diag_counter}")
                print(f"  messages: {messages}")
                if timeseries is not None:
                    print(f"  timeseries: shape={timeseries.shape}, dtype={timeseries.dtype}")
                    print(f"  ts[0,:5] = {timeseries[0,:5]}")
                    print(f"  ts min={timeseries.min():.4f}, max={timeseries.max():.4f}")
                else:
                    print(f"  timeseries: NONE")
                print(f"{'='*60}\n")

        # The parquet raw_prompt is a pre-formatted string:
        #   <|im_start|>user\n<|begin_of_TS|><|end_of_TS|>{q}<|im_end|>\n
        #   <|im_start|>assistant\n
        # Cold-start model already knows <think>...</think> format from SFT.
        raw = kwargs["raw_prompt"]
        if isinstance(raw, str):
            prompt_text = raw
        elif isinstance(raw, list) and len(raw) > 0:
            user_content = raw[0]["content"] if isinstance(raw[0], dict) else str(raw[0])
            prompt_text = (
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt_text = ""

        import asyncio
        prompt_ids = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.tokenizer.encode(prompt_text, add_special_tokens=False),
        )

        # Generate with timeseries injected into multi_modal_data
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                timeseries_data=timeseries_data,
            )

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_mask = [1] * len(output.token_ids)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )


# Monkey-patch AgentLoopWorker._compute_multi_modal_inputs to carry
# timeseries data through to the FSDP training batch.
# Without this patch, _compute_multi_modal_inputs returns {} when
# self.processor is None (no vision processor for TS models), so the
# timeseries in multi_modal_data["timeseries"] is silently dropped.
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

_orig_compute_mm = AgentLoopWorker._compute_multi_modal_inputs


def _ts_compute_multi_modal_inputs(self, output, input_ids):
    result = _orig_compute_mm(self, output, input_ids)

    # Inject timeseries as ts_values tensor if present in multi_modal_data
    # Note: KPI unit scaling is already applied in _deserialize_timeseries
    if output.multi_modal_data and "timeseries" in output.multi_modal_data:
        ts_list = output.multi_modal_data["timeseries"]
        ts_array = ts_list[0] if isinstance(ts_list, list) else ts_list
        if isinstance(ts_array, np.ndarray):
            result["ts_values"] = torch.from_numpy(ts_array.copy()).float().unsqueeze(0)  # [1, C, T]
        elif isinstance(ts_array, torch.Tensor):
            result["ts_values"] = ts_array.float().unsqueeze(0)
        logger.info(f"[agent_loop] Injected ts_values {result['ts_values'].shape} into multi_modal_inputs")

    return result


AgentLoopWorker._compute_multi_modal_inputs = _ts_compute_multi_modal_inputs
print("[agent_loop] Patched AgentLoopWorker._compute_multi_modal_inputs for timeseries → ts_values ✓")
