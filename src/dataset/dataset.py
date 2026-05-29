import json
import logging
import os
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from typing import Optional, List
from .load import process_dataset
from .qa_templates import QA_templates
from .base import TimeseriesData, TaskDataset

random.seed(42)

class PretrainingDataset(TaskDataset):
    """
    Dataset for pretraining models on time series + natural language QnA tasks.
    Includes normalization, interpolation, and optional upsampling.
    """

    def __init__(
        self,
        seq_len_channel: int = 128,
        data_split: str = "train",
        scale: bool = False,
        task_name: str = "pretraining",
        train_ratio: float = 0.8,
        val_ratio: float = 0,
        test_ratio: float = 0.2,
        upsampling_pad_direction: str = "backward",
        upsampling_type: str = "pad",
        downsampling_type: str = "interpolate",
        pad_mode: str = "constant",
        pad_constant_values: int = 0,
        return_meta_data: bool = False,
        classification_head: tuple = (False, "root_cause"),
        KPI_list: Optional[List[str]] = None,
        descr_pretrain: bool = False,
        use_thinking: bool = False,
        use_cot_labels: bool = False,
        task_list: Optional[List[str]] = None,
        balance: bool = True,
        skip_done_traces_path: Optional[str] = None,
    ):
        super().__init__()
        self.seq_len_channel = seq_len_channel
        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values
        self.return_meta_data = return_meta_data
        self.classification_head = classification_head
        self.KPI_list = KPI_list
        self.descr_pretrain = descr_pretrain
        self.use_thinking = use_thinking
        self.use_cot_labels = use_cot_labels
        self.task_list = task_list
        self.balance = balance
        self.skip_done_traces_path = skip_done_traces_path

        self._check_inputs()
        self._read_data()
        self._split_data()
        if self.balance:
            self._balance_data(description=self.descr_pretrain, use_thinking=self.use_thinking)
        if self.skip_done_traces_path:
            self._filter_done_traces()

    def _check_inputs(self) -> None:
        """Validate `data_split` argument."""
        assert self.data_split in ["train", "val", "test"], \
            "data_split must be one of ['train', 'val', 'test']"

    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        """Map labels to a contiguous integer range starting from 0."""
        label_map = {label: i for i, label in enumerate(np.unique(train_labels))}
        train_labels = np.vectorize(label_map.get)(train_labels)
        test_labels = np.vectorize(label_map.get)(test_labels)
        return train_labels, test_labels

    def _split_data(self):
        """Split data into train, validation, and test sets."""
        train_end = int(self.data_size * self.train_ratio)

        if self.data_split == "train":
            self.data = self.data[:train_end]
        else:  # test
            self.data = self.data[train_end:]

        self.data_size = len(self.data)

    def _balance_data(
        self, inter_category_balance=False, description=False,
        use_thinking=False, use_cot_labels=False
    ) -> None:
        """Balance the dataset across different question categories and anomaly types."""
        if description:
            unique_descriptions = set()
            unique_data = []
            for s in self.data:
                desc = s.description
                if desc not in unique_descriptions:
                    unique_descriptions.add(desc)
                    unique_data.append(s)
            self.data = unique_data
            self.data_size = len(self.data)

        if not description:
            # === Split out categories to balance ===
            root_cause = [s for s in self.data if s.question_category == "root_cause"]
            anomaly_detection = [s for s in self.data if s.question_category == "anomaly_detection"]
            normal_categories = ["zone", "activity", "cong", "motion"]

            # === Balance root_cause and anomaly_detection (equal samples per anomaly_type) ===
            if root_cause:
                random.shuffle(root_cause)
                type_to_samples = {}
                for s in root_cause:
                    type_to_samples.setdefault(s.anomaly_type, []).append(s)
                min_count = min(len(v) for v in type_to_samples.values())
                balanced_root_cause = [s for lst in type_to_samples.values() for s in lst[:min_count]]

            if anomaly_detection:
                non_list = [s for s in anomaly_detection if s.anomaly_type == 0]
                anom_list = [s for s in anomaly_detection if s.anomaly_type != 0]
                target = min(len(non_list), len(anom_list))
                random.shuffle(non_list)
                random.shuffle(anom_list)
                balanced_anomaly_detection = non_list[:target] + anom_list[:target]
                random.shuffle(balanced_anomaly_detection)

            # === Balance normal QA categories ===
            balanced_by_category = {}

            for category in normal_categories:
                if not (use_thinking or use_cot_labels):
                    key_to_answers = {
                        key: {ans for _, ans in QA_templates[category][key]}
                        for key in QA_templates[category]
                    }

                # Group samples by key
                if not (use_thinking or use_cot_labels):
                    type_to_samples = {key: [] for key in QA_templates[category]}
                else:
                    # We can use the parsed_answer directly as key
                    type_to_samples = {}
                for s in self.data:
                    if s.question_category != category:
                        continue
                    if not (use_thinking or use_cot_labels):
                        for key, ans_set in key_to_answers.items():
                            answers = s.answers

                            if answers in ans_set:
                                type_to_samples[key].append(s)
                                break
                    else:
                        key = s.parsed_answer
                        type_to_samples.setdefault(key, []).append(s)

                counts = [len(v) for v in type_to_samples.values() if v]

                if not counts:
                    # No samples to balance in this category
                    continue

                target = min(counts)
                balanced = []
                for lst in type_to_samples.values():
                    random.shuffle(lst)
                    balanced.extend(lst[:target])
                random.shuffle(balanced)
                balanced_by_category[category] = balanced

            # === Balance "trends" category (equal samples per trend per KPI) ===
            trend_samples = [s for s in self.data if s.question_category == "trends"]
            if trend_samples:
                random.shuffle(trend_samples)
                kpi_to_trend_samples = {}
                for s in trend_samples:
                    if use_thinking or use_cot_labels:
                        answers = s.parsed_answer
                    else:
                        answers = s.answers
                    parts = answers.split()
                    try:
                        if not (use_thinking or use_cot_labels):
                            kpi_name = parts[3]
                            trend_value = parts[-1]
                        else:
                            kpi_name = parts[-1]
                            trend_value = parts[0]
                    except IndexError:
                        continue
                    kpi_to_trend_samples.setdefault(kpi_name, {}).setdefault(trend_value, []).append(s)

                balanced_trends = []
                for kpi, trend_dict in kpi_to_trend_samples.items():
                    # Only proceed if we have at least 2+ trend groups
                    if len(trend_dict) < 2:
                        continue
                    # Find smallest trend count
                    target = min(len(lst) for lst in trend_dict.values())
                    if target == 0:
                        continue
                    # For each trend value, randomly select 'target' samples
                    for trend_value, lst in trend_dict.items():
                        random.shuffle(lst)
                        balanced_trends.extend(lst[:target])

                random.shuffle(balanced_trends)
                balanced_by_category["trends"] = balanced_trends


            # === Remove all categories being balanced (once) ===
            to_remove = {"root_cause", "anomaly_detection"} | set(normal_categories) | {"trends"}
            self.data = [s for s in self.data if s.question_category not in to_remove]

            # === Add all balanced samples back, only if they exist ===
            if 'balanced_root_cause' in locals():
                self.data.extend(balanced_root_cause)
            if 'balanced_anomaly_detection' in locals():
                self.data.extend(balanced_anomaly_detection)
            for cat in normal_categories:
                if cat in balanced_by_category:
                    self.data.extend(balanced_by_category[cat])
            if "trends" in balanced_by_category:
                self.data.extend(balanced_by_category["trends"])


            self.data_size = len(self.data)
            # === Inter-category balancing (optional) ===
            if inter_category_balance == True:
                category_count = {}
                for sample in self.data:
                    category = sample.question_category
                    category_count[category] = category_count.get(category, 0) + 1

                if category_count != {}:
                    min_count = min(category_count.values())

                    for i in reversed(range(self.data_size)):
                        sample = self.data[i]
                        category = sample.question_category
                        if category_count[category] == min_count:
                            continue
                        if category_count[category] > min_count:
                            self.data.pop(i)
                            category_count[category] -= 1

            random.shuffle(self.data)
            self.data_size = len(self.data)


    def _read_data(self) -> None:
        """Load and preprocess data from file."""
        self.scaler = StandardScaler()
        self.data = load_dataset("AliMaatouk/TelecomTS", data_files="**/processed/chunked.jsonl")
        self.data = self.data["train"]
        self.data = process_dataset(
            self.data, self.classification_head, use_thinking=self.use_thinking,
            use_cot_labels=self.use_cot_labels, task_list=self.task_list
        )
        random.shuffle(self.data)
        self._check_and_remove_nans()
        if self.scale:
            for i, ts in enumerate(self.data):
                ts_arr = ts.timeseries.T  # [L, C]
                ts_scaled = self.scaler.fit_transform(ts_arr).T  # [C, L]
                self.data[i].timeseries = ts_scaled

        self.data_size = len(self.data)

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> TimeseriesData:
        """Return a single timeseries + QnA sample, with optional padding."""
        assert index < self.__len__()

        sample = self.data[index]
        timeseries = sample.timeseries  # shape: [C, L]
        input_mask = np.ones((timeseries.shape[0], timeseries.shape[1]))

        return TimeseriesData(
            timeseries=timeseries,
            timeseries_input_mask=input_mask,
            description=sample.description,
            questions=sample.questions,
            answers=sample.answers,
            question_category=sample.question_category,
            anomaly_type=sample.anomaly_type,
            timestamp=sample.timestamp,
            reasoning=sample.reasoning if self.use_thinking else None,
            parsed_answer=sample.parsed_answer if self.use_thinking else None
        )

    def _filter_done_traces(self) -> None:
        """Drop samples whose (start_idx, category) already has a trace in the JSONL.

        Done at load time so downstream batches stay full-sized instead of
        shrinking after post-batch filtering.
        """
        if not os.path.exists(self.skip_done_traces_path):
            logging.info(
                f"Skip-done filter: {self.skip_done_traces_path} does not exist; nothing to skip"
            )
            return
        done_keys = set()
        with open(self.skip_done_traces_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (
                        str(rec.get("start_idx", "")),
                        str(rec.get("category", "")).strip().lower(),
                    )
                    done_keys.add(key)
                except Exception:
                    continue
        before = len(self.data)
        self.data = [
            s for s in self.data
            if (str(s.timestamp[0]), str(s.question_category).strip().lower()) not in done_keys
        ]
        self.data_size = len(self.data)
        logging.info(
            f"Skip-done filter: removed {before - self.data_size} samples already in "
            f"{self.skip_done_traces_path}; {self.data_size} remaining"
        )

    def _check_and_remove_nans(self) -> None:
        """Interpolate and replace NaNs in the dataset."""
        for i, sample in enumerate(self.data):
            ts = sample.timeseries
            if np.isnan(ts).any():
                ts = interpolate_timeseries(ts, interp_length=ts.shape[-1])
                ts = np.nan_to_num(ts)
                self.data[i].timeseries = ts


    def _check_if_equal_length(self) -> None:
        """Force all time series to have the same length using interpolation."""
        if isinstance(self.data, list):
            max_len = max([sample.timeseries.shape[-1] for sample in self.data])
            for sample in self.data:
                sample.timeseries = interpolate_timeseries(sample.timeseries, interp_length=max_len)
            logging.info(f"Time-series have unequal lengths. Reshaped to length {max_len}")
