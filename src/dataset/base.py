from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Optional
import numpy.typing as npt
from torch.utils.data import Dataset


@dataclass
class TimeseriesData:
    """
    Represents a single or batched timeseries data sample used in TS+LLM tasks.

    Attributes:
        questions (Optional[str or List[str]]): The associated natural language question(s).
        answers (Optional[str or List[str]]): The corresponding answer(s).
        question_category (Optional[str]): Category label for the question type.
        timeseries (Optional[NDArray]): Time series data [B, C, T] or [C, T].
        timeseries_input_mask (Optional[NDArray]): Mask for valid time-series values.
        input_ids (Optional[NDArray]): Tokenized input IDs for the question prompt.
        attention_mask (Optional[NDArray]): Attention mask for input IDs.
        labels (Optional[NDArray]): Labels used for supervised generation (e.g., shifted input_ids).
    """
    questions: Optional[Union[str, List[str]]] = None
    answers: Optional[Union[str, List[str]]] = None
    question_category: Optional[str] = None
    description: Optional[str] = None
    timeseries: Optional[npt.NDArray] = None                # Shape: [B, C, T] or [C, T]
    timeseries_input_mask: Optional[npt.NDArray] = None     # Shape: [B, T] or [T]
    input_ids: Optional[npt.NDArray] = None                 # Shape: [B, L] or [L]
    attention_mask: Optional[npt.NDArray] = None            # Shape: [B, L] or [L]
    labels: Optional[npt.NDArray] = None                    # Shape: [B, L] or [L]
    anomaly_type: Optional[str] = None
    timestamp: Optional[List[List]] = None
    reasoning: Optional[str] = None                        # Chain-of-thought reasoning text
    parsed_answer: Optional[str] = None                    # Parsed final answer from reasoning


class TaskDataset(ABC, Dataset):
    """
    Abstract base class for datasets combining time series with natural language supervision.
    """
    def __init__(self):
        super(TaskDataset, self).__init__()

    def _read_data(self) -> TimeseriesData:
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError

    def plot(self, idx):
        return NotImplementedError

    def _check_and_remove_nans(self):
        return NotImplementedError

    def _subsample(self):
        return NotImplementedError
