"""Time-Series LLM models.

``FullModel`` is the early-fusion TSLLM used for cold-start SFT and evaluation
(TeleEncoder output + Qwen3-4B via a learned linear alignment layer).
``TSLLMModel`` is exposed as an alias for it. ``TSLLMForCausalLM`` is the
vLLM/HuggingFace causal variant used for batched serving.

Imports are lazy (PEP 562) so importing this package does not require the heavy
optional dependencies until a class is accessed.
"""

__all__ = ["FullModel", "TSLLMModel", "TSLLMForCausalLM"]


def __getattr__(name):
    if name in ("FullModel", "TSLLMModel"):
        from .tsllm import FullModel

        return FullModel
    if name == "TSLLMForCausalLM":
        from .tsllm_causal.modeling_tsllm import TSLLMForCausalLM

        return TSLLMForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
