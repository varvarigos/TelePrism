"""TelePrism: a multi-modal foundation model for telecom time-series reasoning.

Key classes are exposed lazily so that ``import teleprism`` stays lightweight and
does not pull in heavy optional dependencies (torch, vLLM, baseline encoders)
until a class is actually accessed.
"""

__version__ = "0.1.0"

__all__ = ["TeleEncoder", "TSLLMModel", "FullModel", "PretrainingDataset", "__version__"]


def __getattr__(name):  # PEP 562 lazy attribute loading
    if name == "TeleEncoder":
        from teleprism.encoders.teleencoder.teleEncoder import TeleEncoder

        return TeleEncoder
    if name in ("TSLLMModel", "FullModel"):
        # Early-fusion TSLLM: TeleEncoder output + Qwen3-4B via a learned linear
        # alignment layer. ``TSLLMModel`` is an alias for ``FullModel``.
        from teleprism.models.ts_llm.tsllm import FullModel

        return FullModel
    if name == "PretrainingDataset":
        from teleprism.dataset.dataset import PretrainingDataset

        return PretrainingDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
