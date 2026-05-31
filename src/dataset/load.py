import random
import numpy as np
import pandas as pd
import hashlib
from typing import List
from .base import TimeseriesData
from tqdm import tqdm

from .qa_templates import QA_templates, anomalies_type_2_id

random.seed(42)

def get_question_category(question: str, answer: str) -> str:
    q = question.strip().lower()
    a = answer.strip().lower()

    if "anomaly detected in this sample" in q or "was an anomaly detected" in q:
        return "anomaly_detection", None
    if "root cause of the anomaly" in q:
        return "root_cause", None
    if "index range of the anomaly" in q:
        return "anomaly_bounds", None
    if "how long did the anomaly last" in q or "anomaly lasted" in q:
        return "anomaly_length", None

    for category, label_map in QA_templates.items():
        for label, qas in label_map.items():
            for q_template, a_template in qas:
                if q == q_template.strip().lower() and a == a_template.strip().lower():
                    return category, label

    if "average value" in q:
        return "mean", None

    elif "variance" in q:
        return "variance", None

    elif "periodicity" in q:
        return "periodicity", None

    elif "trend" in q:
        return "trends", None

    return "unknown", None

def stable_hashing(col: List[str], num_buckets: int = 100) -> np.ndarray:
    return np.array([
        int(hashlib.sha256(val.encode()).hexdigest(), 16) % num_buckets
        for val in col
    ])

def get_anomaly_type_answer(data):
    anomaly_type = data["anomalies"]["type"]
    if isinstance(anomaly_type, str):
        anomaly_type = anomaly_type.strip()
        if not anomaly_type:
            return "The type of the anomaly is unknown."
        return f"The anomaly is of type '{anomaly_type}'."
    else:
        return "The type of the anomaly is unknown."


def process_dataset(dataset: List[dict],
                    use_thinking=False, task_list: List[str] = None) -> List[TimeseriesData]:
    """Build TimeseriesData rows from AliMaatouk/TelecomTS chunks.

    Reasoning lives directly on each Q&A entry (`QA["reasoning"]`) — no
    cold_start/traces.jsonl lookup. The anomalies array on the HF dataset
    already contains anomaly_detection (all chunks) + root_cause + anomaly_bounds
    (anomalous chunks); we iterate it the same way as network/timeseries.
    """
    proto_vocab = {"": 0, "UDP": 1, "None": 0, "TCP": 2}
    kpi_set = dataset[0]["KPIs"].keys()
    processed_dataset = []

    for data in tqdm(dataset):
        description = data["description"]
        anomaly_type = anomalies_type_2_id[data["anomalies"]["type"]] if data["anomalies"]["exists"] else 0
        timestamp = pd.date_range(
            start=pd.to_datetime(data["start_time"]),
            end=pd.to_datetime(data["end_time"]),
            freq=pd.to_timedelta(1 / data["sampling_rate"], unit="s"),
        ).to_numpy()

        if "UL_Protocol" in kpi_set:
            data["KPIs"]["UL_Protocol"] = [proto_vocab.get(x, 0) for x in data["KPIs"]["UL_Protocol"]]
        if "DL_Protocol" in kpi_set:
            data["KPIs"]["DL_Protocol"] = [proto_vocab.get(x, 0) for x in data["KPIs"]["DL_Protocol"]]

        ts_array = np.asarray([data["KPIs"][kpi] for kpi in kpi_set])

        qna = data.get("QnA", {}) or {}
        all_qa = (qna.get("network") or []) + (qna.get("timeseries") or []) + (qna.get("anomalies") or [])

        for QA in all_qa:
            category, label_map = get_question_category(QA["q"], QA["a"])
            a = QA["a"]

            if task_list is not None and category not in task_list:
                continue
            if category == "jam":
                continue
            if category == "zone" and label_map == "User is in motion.":
                continue

            reasoning_text = (QA.get("reasoning") or "").strip()
            parsed_answer = a  # answer field on the QA entry IS the parsed answer

            if use_thinking:
                # Keep only samples that have a reasoning trace.
                if reasoning_text == "":
                    continue

            processed_dataset.append(
                TimeseriesData(
                    timeseries=ts_array,
                    questions=QA["q"],
                    description=description,
                    answers=a,
                    question_category=category,
                    anomaly_type=anomaly_type,
                    timestamp=timestamp,
                    reasoning=reasoning_text if use_thinking else None,
                    parsed_answer=parsed_answer if use_thinking else None,
                )
            )

        # Add description task
        if task_list is not None and 'description' in task_list:
            processed_dataset.append(
                TimeseriesData(
                    timeseries=ts_array,
                    questions="Describe the current network conditions based on the provided KPIs.",
                    description=description,
                    answers=description,
                    question_category="description",
                    anomaly_type=anomaly_type,
                    timestamp=timestamp
                )
            )

    return processed_dataset
