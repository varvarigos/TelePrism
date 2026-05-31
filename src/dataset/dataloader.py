import torch
import functools
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from .base import TimeseriesData
from .dataset import PretrainingDataset
from transformers import PreTrainedTokenizer
from typing import List
from teleprism.evaluation.tasks.prompts.anomaly_tasks_prompts import anomaly_list
from .qa_templates import QA_templates, answer_normalization, normalize_target


def build_reverse_map(qa_templates, answer_normalization):
    reverse_map = defaultdict(dict)
    for category in qa_templates:
        for label, answer_options in qa_templates[category].items():
            normalized = answer_normalization[category][label]
            for qa in answer_options:
                reverse_map[category][qa[1].lower().strip()] = normalized
    return reverse_map


def collate_fn_timeseries_pretraining(
    batch: List[TimeseriesData], tokenizer: PreTrainedTokenizer,
    model_type: str = "llama",
    device: torch.device = torch.device("cpu"),
    description: bool = False,
    use_thinking: bool = False,
    use_vllm: bool = False,
) -> TimeseriesData:
    """
    Collate function for pretraining with instruct-style prompting.

    Returns:
        TimeseriesData: Contains normalized time-series, tokenized input_ids,
                        attention_mask, and labels for language modeling.
    """

    # Stack and normalize time series: [B, C, T]
    ts_tensor = torch.stack(
        [torch.tensor(sample.timeseries, dtype=torch.float32) for sample in batch]
    )


    # Construct chat-style or plain prompt+answer sequences
    if tokenizer.chat_template is None:
        full_texts = [
            "<|begin_of_TS|><|end_of_TS|> " + sample.questions + sample.answers
            for sample in batch
        ]
        prompt_texts = [
            "<|begin_of_TS|><|end_of_TS|> " + sample.questions for sample in batch
        ]
        full_inputs = tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True
        )
        prompt_inputs = tokenizer(prompt_texts, padding=False, truncation=True)
        prompt_lengths = [len(p) for p in prompt_inputs["input_ids"]]
        input_ids = full_inputs["input_ids"]

    else:
        # Construct chat-format messages
        if description == True:
            if model_type == "llama":
                conversations = [
                    (
                        f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n<|begin_of_TS|><|end_of_TS|><|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n{sample.description}<|eot_id|>"
                    )
                    for sample in batch
                ]
            elif model_type == "qwen":
                conversations = [
                    (
                        f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|><|im_end|>\n"
                        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{sample.description}<|im_end|>"
                    )
                    for sample in batch
                ]

            input_ids = tokenizer(
                conversations, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]

            # Compute prompt lengths for masking
            if model_type == "llama":
                user_only = [
                    (
                        f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n<|begin_of_TS|><|end_of_TS|><|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    for sample in batch

                ]
            elif model_type == "qwen":
                user_only = [
                    (
                        f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|><|im_end|>\n"
                        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    )
                    for sample in batch
                ]
            user_ids = [
                tokenizer(u, padding=False, truncation=True, return_tensors="pt")["input_ids"]
                for u in user_only
            ]
            prompt_lengths = [u.shape[1] for u in user_ids]

        else:
            if model_type == "llama":
                conversations = [
                    (
                        f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n{sample.answers}<|eot_id|>"
                    )
                    for sample in batch
                ]
            elif model_type == "qwen":
                if use_thinking:
                    conversations = [
                        (
                            # CASE 1: reasoning exists → use <think>
                            f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n{sample.reasoning}\n</think>\n\n{sample.answers}<|im_end|>"
                        ) if sample.reasoning.strip() else (
                            # CASE 2: reasoning missing → normal response (thinking disabled)
                            f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{sample.answers}<|im_end|>"
                        )
                        for sample in batch
                    ]
                else:
                    conversations = [
                        (
                            f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{sample.answers}<|im_end|>"
                        )
                        for sample in batch
                    ]

            input_ids = tokenizer(
                conversations, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]

            # Compute prompt lengths for masking
            if model_type == "llama":
                user_only = [
                    (
                        f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    for sample in batch
                ]
            elif model_type == "qwen":
                if use_thinking:
                    user_only = [
                        (
                            f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|im_end|>\n"
                            f"<|im_start|>assistant\n"
                        ) if sample.reasoning.strip() else (
                            f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        )
                        for sample in batch
                    ]
                else:
                    user_only = [
                        (
                            f"<|im_start|>user\n<|begin_of_TS|><|end_of_TS|>{sample.questions}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        )
                        for sample in batch
                    ]
            user_ids = [
                tokenizer(u, padding=False, truncation=True, return_tensors="pt")["input_ids"]
                for u in user_only
            ]
            prompt_lengths = [u.shape[1] for u in user_ids]

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    if not use_vllm:
        ts_tensor = ts_tensor.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device, dtype=torch.long)

    # Compute labels with prompt masking
    target_ids = input_ids.clone()
    for i, prompt_len in enumerate(prompt_lengths):
        target_ids[i, :prompt_len] = -100  # Mask out prompt tokens for loss computation
    target_ids[input_ids == tokenizer.pad_token_id] = -100


    if not use_vllm:
        return TimeseriesData(
            timeseries=ts_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            timestamp=[item.timestamp for item in batch],
        )
    else:
        return TimeseriesData(
            timeseries=ts_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            timestamp=[item.timestamp for item in batch],
            questions=[item.questions for item in batch],
            answers=[item.answers for item in batch],
            description=[item.description for item in batch],
            question_category=[item.question_category for item in batch],
            anomaly_type=[item.anomaly_type for item in batch],
        )


def get_dataloader(args, tokenizer: PreTrainedTokenizer, data_split: str) -> DataLoader:
    """
    Creates a PyTorch DataLoader for timeseries-language pretraining.

    Args:
        args: Argument namespace or dict with training configuration.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        data_split (str): Dataset split to load, e.g., 'train', 'val'.

    Returns:
        DataLoader: A configured PyTorch DataLoader.
    """
    dataset = PretrainingDataset(
        seq_len_channel=args.seq_len_channel,
        data_split=data_split,
        scale=args.scale,
        upsampling_pad_direction=args.upsampling_pad_direction,
        upsampling_type=args.upsampling_type,
        downsampling_type=args.downsampling_type,
        pad_mode=args.pad_mode,
        KPI_list=args.data["KPI_list"],
        descr_pretrain=args.descr_pretrain,
        use_thinking=args.use_thinking,
        task_list=args.task_list,
        balance=getattr(args, "balance_dataset", True),
        skip_done_traces_path=getattr(args, "skip_done_traces_path", None),
        train_ratio=getattr(args, "train_ratio", 0.8),
        test_ratio=getattr(args, "test_ratio", 0.2),
    )

    # unique_root_cause_answers = set()
    for sample in dataset.data:
        if sample.question_category == "root_cause":
            sample.questions += f" The valid anomalies are {', '.join(anomaly_list)}"

    model_type = "llama" if "llama" in args.llm_model.lower() else "qwen" if "qwen" in args.llm_model.lower() else "other"
    use_vllm = getattr(args, "use_vllm", False)

    collate_fn = functools.partial(
        collate_fn_timeseries_pretraining, tokenizer=tokenizer, model_type=model_type,
        device=args.device, description=args.descr_pretrain,
        use_thinking=args.use_thinking, use_vllm=use_vllm
    )

    if getattr(args, "distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=args.shuffle,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            sampler=sampler,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
        )

    return dataloader


def get_eval_dataloader(args, tokenizer: PreTrainedTokenizer, data_split: str) -> dict:
    """
    Creates a set of DataLoaders for evaluation, partitioned by `question_category`.

    Args:
        args: Argument namespace or dict with training configuration.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        data_split (str): Dataset split to load, e.g., 'val', 'test'.

    Returns:
        dict[str, DataLoader]: Dictionary mapping question_category to its DataLoader.
    """
    dataset = PretrainingDataset(
        seq_len_channel=args.seq_len_channel,
        data_split=data_split,
        scale=args.scale,
        upsampling_pad_direction=args.upsampling_pad_direction,
        upsampling_type=args.upsampling_type,
        downsampling_type=args.downsampling_type,
        pad_mode=args.pad_mode,
        KPI_list=args.data["KPI_list"],
        use_thinking=args.use_thinking,
    )

    for sample in dataset.data:
        if sample.question_category == "root_cause":
            sample.questions += f" The valid anomalies are {', '.join(anomaly_list)}"


    category_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.data):
        cat = sample.question_category
        category_to_indices[cat].append(idx)

    model_type = "llama" if "llama" in args.llm_model.lower() else "qwen" if "qwen" in args.llm_model.lower() else "other"
    use_vllm = getattr(args, "use_vllm", False)

    collate_fn = functools.partial(
        collate_fn_timeseries_pretraining, tokenizer=tokenizer, model_type=model_type,
        use_thinking=args.use_thinking,
        device=args.device, use_vllm=use_vllm
    )

    reverse_map = build_reverse_map(QA_templates, answer_normalization)

    dataloaders_by_category = {}
    for category, indices in category_to_indices.items():
        partition = defaultdict(list)
        args.sample_size = None 
        if args.sample_size is None:
            partition["full"] = indices
        else:
            for idx in indices:
                answer = dataset.data[idx].answers.lower()
                if category in ["jamming", "cong"]:
                    answer = normalize_target(answer.strip(), category, reverse_map)
                    if "yes" in answer and "no" not in answer:
                        partition["yes"].append(idx)
                    elif "no" in answer and "yes" not in answer:
                        partition["no"].append(idx)
                elif category == "motion":
                    answer = normalize_target(answer.strip(), category, reverse_map)
                    if "mobile" in answer and "static" not in answer:
                        partition["yes"].append(idx)
                    elif "static" in answer and "mobile" not in answer:
                        partition["no"].append(idx)
                elif category in ["activity", "zone"]:
                    true_answer = normalize_target(
                        answer.strip(), category, reverse_map
                    )
                    if true_answer != "motion":
                        partition[true_answer].append(idx)
                elif category == "anomaly_detection":
                    partition["normal"].append(idx)
                else:
                    if "jamming" in answer:
                        continue
                    partition["full"].append(idx)

            if category in ["root_cause", "anomaly_length", "anomaly_bounds"]:
                non_empty_partitions = {k: v for k, v in partition.items() if k != ""}
                min_len = (
                    min(len(v) for v in non_empty_partitions.values())
                    if non_empty_partitions
                    else 0
                )
                for k in non_empty_partitions:
                    partition[k] = partition[k][:min_len]
                anomaly_indices = []
                for k in list(partition.keys()):
                    if k != "":
                        anomaly_indices.extend(partition[k])
                        del partition[k]
                partition["anomaly"] = anomaly_indices

            num_classes = len(partition)
            max_per_class = args.sample_size // num_classes
            min_len_across_partitions = min(len(v) for v in partition.values())
            final_sample_size = min(min_len_across_partitions, max_per_class)

            if final_sample_size == 0:
                print(f"Insufficient Samples in {category}")
                continue
            else:
                print(f"Total {final_sample_size * num_classes} samples for {category}")

            for k in partition:
                partition[k] = partition[k][:final_sample_size]

        partition_indices = [i for indices in partition.values() for i in indices]
        subset = Subset(dataset, partition_indices)

        shuffle = True if args.shuffle is None else args.shuffle
        sampler = DistributedSampler(subset, shuffle=shuffle, drop_last=True)


        dataloader = DataLoader(
            subset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn
        )

        dataloaders_by_category[category] = dataloader

    return dataloaders_by_category
