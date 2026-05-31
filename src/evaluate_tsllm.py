import os
import json
import yaml
import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np
import regex as re
import torch.distributed as dist
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import deepspeed

from tqdm import tqdm
from collections import defaultdict
from functools import partial

from teleprism.inference_tsllm import load_model
from teleprism.evaluation.tasks.time_series_qa import extract_number
from teleprism.evaluation.tasks.prompts.anomaly_tasks_prompts import parse_anomaly_detection, parse_anomaly_bounds, parse_root_cause, parse_anomaly_length, anomaly_list, convert_intervals_to_binary
from teleprism.dataset.output_prompts import parse_tagged_response
from teleprism.dataset.qa_templates import QA_templates, answer_normalization, normalize_target

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def scale_kpi_units_(x: torch.tensor, KPI_list: list[str]) -> torch.tensor:
    """In-place unit scaling."""
    if "TX_Bytes" in KPI_list:
        tx_index = KPI_list.index("TX_Bytes")
        x[:, tx_index, :] = x[:, tx_index, :] / 1e6  # Convert bytes → MB
    if "RX_Bytes" in KPI_list:
        rx_index = KPI_list.index("RX_Bytes")
        x[:, rx_index, :] = x[:, rx_index, :] / 1e6  # Convert bytes → MB
    if "Estimated_UL_Buffer" in KPI_list:
        ul_index = KPI_list.index("Estimated_UL_Buffer")
        x[:, ul_index, :] = x[:, ul_index, :] / 1000  # Convert KB → MB
    if "DL_NumberOfPackets" in KPI_list:
        ul_index = KPI_list.index("DL_NumberOfPackets")
        x[:, ul_index, :] = x[:, ul_index, :] / 100
    if "UL_NumberOfPackets" in KPI_list:
        ul_index = KPI_list.index("UL_NumberOfPackets")
        x[:, ul_index, :] = x[:, ul_index, :] / 100
    return x

def build_reverse_map(qa_templates, answer_normalization):
    reverse_map = defaultdict(dict)
    for category in qa_templates:
        for label, answer_options in qa_templates[category].items():
            normalized = answer_normalization[category][label]
            for qa in answer_options:
                reverse_map[category][qa[1].lower().strip()] = normalized
    return reverse_map

eval_parse_dict = {
    'cong': partial(parse_tagged_response, tag="cong", options=["Yes", "No"]),
    'activity': partial(parse_tagged_response, tag="activity", options=["Youtube", "Twitch", "File"]),
    'zone': partial(parse_tagged_response, tag="zone", options=["A", "B", "C"]),
    'motion': partial(parse_tagged_response, tag="motion", options=["mobile", "static"]),
    'mean': extract_number,
    'variance': extract_number,
    'trends': extract_number,
    'periodicity': extract_number,
    'root_cause': parse_root_cause,
    'anomaly_bounds': parse_anomaly_bounds,
    'anomaly_detection': parse_anomaly_detection,
    'anomaly_length': parse_anomaly_length,
    'jamming': partial(parse_tagged_response, tag="jamming", options=["Yes", "No"]),
}

def main():
    # Load DeepSpeed config
    with open("configs/ds_conf.json", "r") as f:
        ds_config = json.load(f)

    parser = argparse.ArgumentParser()

    # Model & Checkpoint
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="epoch-0")

    # Prediction Saving
    parser.add_argument("--save_predictions", type=bool, default=False)
    parser.add_argument("--predictions_dir", type=str, default="predictions")

    # Dataloader & Env
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=ds_config["train_micro_batch_size_per_gpu"])
    parser.add_argument("--num_workers", type=int, default=os.environ["CUDA_VISIBLE_DEVICES"].count(',') + 1)
    parser.add_argument("--sample_size", type=int, default=None)

    # TS & LoRA hyperparameters
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # TS preprocessing
    parser.add_argument("--seq_len_channel", type=int, default=128)
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--upsampling_pad_direction", type=str, default="forward")
    parser.add_argument("--upsampling_type", type=str, default="pad")
    parser.add_argument("--downsampling_type", type=str, default="average")
    parser.add_argument("--pad_mode", type=str, default="constant")

    # Misc
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--pin_memory", type=bool, default=True)

    deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Merge args with defaults
    with open("configs/evaluate_tsllm.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**{**cfg, **vars(args)})

    print("Finished Loading Args")
    # Initialize model + loader
    print(args.model_name)
    model, tokenizer, dataloader, device, dtype = load_model(args, ds_config, eval=True)

    # Initialize results storage

    from collections import defaultdict
    results = defaultdict(list)

    kpi_list = args.data["KPI_list"]
    for stat in ['mean', 'variance', 'periodicity', 'trends']:
        if not args.eval_tasks[stat]:
            continue
        results[stat] = {}
        
        for kpi in kpi_list:
            results[stat][kpi] = defaultdict(list)
    if args.eval_tasks['anomaly_length']:
        results['anomaly_length'] = {'mae': [], 'mse': []}



    all_predictions = []

    reverse_map = build_reverse_map(QA_templates, answer_normalization)

    print("Beginning Evaluation")
    for category, category_loader in dataloader.items():
        if not args.eval_tasks[category]:
            continue
        if len(category_loader) == 0:
            continue
        print(f"Beginning evaluation of {category}")
        for step, batch in enumerate(tqdm(category_loader)):
            ts = batch.timeseries.to(device, dtype=dtype)
            ts = scale_kpi_units_(ts, args.data["KPI_list"])
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            text_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            labels = batch.labels.to(device)
            initial_labels = labels

            #B, L = input_ids.shape
            if 'llama' in args.llm_model.lower():
                strip = torch.tensor(
                    tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False),
                    device=input_ids.device
                )
            elif 'qwen' in args.llm_model.lower():
                if args.use_thinking:
                    strip = torch.tensor(
                        tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
                        device=input_ids.device
                    )
                else:
                    strip = torch.tensor(
                        tokenizer.encode("</think>\n\n", add_special_tokens=False),
                        device=input_ids.device
                    )

            # Find the last token index that matches the strip
            match = (input_ids.unfold(1, strip.size(0), 1) == strip).all(dim=2)
            last_token_idx = torch.tensor(
                [m.nonzero(as_tuple=True)[0][-1].item() + strip.size(0) for m in match],
                device=input_ids.device
            )

            max_len = last_token_idx.max().item()

            # Truncate and pad input_ids
            truncated_input_ids = [
                F.pad(input_ids[i, :last_token_idx[i]], (0, max_len - last_token_idx[i]), value=tokenizer.pad_token_id)
                for i in range(input_ids.size(0))
            ]

            # Truncate and pad attention_mask (pad with 0)
            truncated_attention_mask = [
                F.pad(attention_mask[i, :last_token_idx[i]], (0, max_len - last_token_idx[i]), value=0)
                for i in range(attention_mask.size(0))
            ]

            # Truncate and pad labels (pad with -100 to ignore in loss)
            truncated_labels = [
                F.pad(labels[i, :last_token_idx[i]], (0, max_len - last_token_idx[i]), value=-100)
                for i in range(labels.size(0))
            ]

            labels = torch.stack(truncated_labels)

            input_ids = torch.stack(truncated_input_ids)
            attention_mask = torch.stack(truncated_attention_mask)

            text_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            model.eval()
            with torch.no_grad():
                scaled_params = None

                output = model.generate(
                    ts=ts,
                    text_inputs=text_inputs,
                    scaled_params=scaled_params,
                    tokenizer=tokenizer,
                    device=device,
                    dtype=dtype,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=1000,
                    do_sample=True,
                    synced_gpus=True,
                    kwargs={"model_name": args.model_name},
                )

            for idx, (sample_input_ids, sample_labels, sample_output) in enumerate(zip(truncated_input_ids, initial_labels, output)):
                prompt = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
                label_ids_clean = sample_labels[sample_labels != -100]
                true_answer_full = tokenizer.decode(label_ids_clean, skip_special_tokens=True)
                if args.use_thinking:
                    true_thinking, true_ans = true_answer_full.split("</think>")
                    true_thinking = true_thinking.replace("<think>", "").strip()
                    true_answer_full = true_ans.strip()
                try:
                    true_answer = float(true_answer_full)
                except (ValueError, TypeError):
                    if category == "root_cause":
                        true_answer = None
                        for k in anomaly_list:
                            if k.lower() in true_answer_full.lower():
                                true_answer = k
                                break
                        if true_answer is None:
                            continue
                    elif category == "anomaly_detection":
                        if "yes" in true_answer_full.lower() and "no" not in true_answer_full.lower():
                            true_answer = "yes"
                        elif "no" in true_answer_full.lower() and "yes" not in true_answer_full.lower():
                            true_answer = "no"
                        else:
                            continue
                    elif category == "anomaly_length":
                        match = re.search(r"The anomaly lasted for (\d+) time steps", true_answer_full)
                        if match:
                            true_answer = int(match.group(1))
                        else:
                            true_answer = None
                            continue
                    elif category == "anomaly_bounds":
                        # Robust parser handles both the legacy
                        # "starts at X and ends at Y" wording and the new
                        # "X, Y" index-pair format from the cold-start traces.
                        true_answer = parse_anomaly_bounds(true_answer_full)
                    elif category in ['mean','variance','trends','periodicity']:
                        true_answer = eval_parse_dict[category](true_answer_full.lower().strip())
                    else:
                        true_answer = normalize_target(true_answer_full.lower().strip(), category, reverse_map)

                parsed_response = eval_parse_dict[category](sample_output)


                print(f"\n[Full True answer][rank {args.local_rank}]:\n{true_answer_full}")
                if args.use_thinking:
                    print(f"\n[True reasoning][rank {args.local_rank}]:\n{true_thinking}")
                print(f"\n[Prompt][rank {args.local_rank}]:\n{prompt}")
                print(f"\n[Generated output][rank {args.local_rank}]:\n{sample_output}")
                print(f"\n[Parsed response][rank {args.local_rank}]:\n{parsed_response}")
                print(f"\n[True answer][rank {args.local_rank}]:\n{true_answer}")


                if parsed_response is None:
                    continue

                if category in ["activity", "zone", "root_cause"]:
                    results[category].append(parsed_response.lower().strip() == true_answer.lower().strip())
                elif category in ["jamming", "cong", "motion", "anomaly_detection"]:
                    if category == "motion":
                        gt = true_answer.lower().strip() == "mobile"
                        pred = parsed_response.lower().strip() == "mobile"
                    else:
                        gt = true_answer.lower().strip() == "yes"
                        pred = parsed_response.lower().strip() == "yes"
                    results[category].append((gt, pred))
                elif category in ["mean", "variance", "periodicity", "trends"]:
                    kpi = None
                    for k in kpi_list:
                        if k in prompt:
                            kpi = k
                            break
                    if kpi is not None:
                        if category == "trends":
                            if parsed_response == true_answer:
                                results[category][kpi][true_answer].append(1)
                            else:
                                results[category][kpi][true_answer].append(0)
                        else:
                            error = parsed_response - true_answer
                            results[category][kpi]["mae"].append(abs(error))
                            results[category][kpi]["mse"].append(error ** 2)
                    else:
                        raise ValueError(f"No KPI found in prompt: '{prompt}'")
                elif category == "anomaly_length":
                    error = parsed_response - true_answer
                    results[category]["mae"].append(abs(error))
                    results[category]["mse"].append(error ** 2)
                elif category == "anomaly_bounds":
                    if len(parsed_response) == 2:
                        vec = np.zeros(128, dtype=int)  # create binary vector from 0 to 127
                        vec[true_answer[0]:true_answer[1]+1] = 1
                        vec_parse = np.zeros(128, dtype=int)  # create binary vector from 0 to 127
                        vec_parse[parsed_response[0]:parsed_response[1]+1] = 1
                        results[category].append((vec.tolist(), vec_parse.tolist()))
                    
                # Save predictions if enabled
                if args.save_predictions:
                    all_predictions.append({
                        "step": step,
                        "prompt": prompt,
                        "generated": sample_output,
                        "parsed_response": str(parsed_response),
                        "true_answer": str(true_answer),
                        "category": category,
                        "rank": args.local_rank,
                    })

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered_results = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_results, results)
        gathered_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)

        # Merge results
        if torch.distributed.get_rank() == 0:
            merged = {}  # plain dict avoids type collisions

            for r in gathered_results:
                for cat, val in r.items():
                    if isinstance(val, dict):
                        cat_dict = merged.setdefault(cat, {})
                        for subcat, subval in val.items():
                            if isinstance(subval, dict):
                                sub_dict = cat_dict.setdefault(subcat, {})
                                for metric, arr in subval.items():
                                    sub_dict.setdefault(metric, []).extend(arr)
                            else:
                                cat_dict.setdefault(subcat, []).extend(subval)
                    else:
                        merged.setdefault(cat, []).extend(val)

            results = merged

            merged_predictions = []
            for p in gathered_predictions:
                merged_predictions.extend(p)
            all_predictions = merged_predictions

    metrics = {}
    for category in ["activity", "zone", "root_cause"]:
        if not args.eval_tasks[category] or category not in results:
            continue
        acc = np.mean(results[category])
        metrics[category] = {"accuracy": acc}

    for category in ["jamming", "cong", "motion", "anomaly_detection"]:
        if not args.eval_tasks[category] or category not in results:
            continue
        gts, preds = zip(*results[category])
        precision, recall, f1, _ = precision_recall_fscore_support(gts, preds, average='binary', zero_division=0)
        accuracy = accuracy_score(gts, preds)
        metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    if args.eval_tasks["anomaly_bounds"] and "anomaly_bounds" in results:
        pairs = results["anomaly_bounds"]
        # Separate and concatenate all ground truth and predicted vectors
        gts = [gt for gt, _ in pairs]
        preds = [pred for _, pred in pairs]
        gts_flat = np.concatenate(gts)
        preds_flat = np.concatenate(preds)
        precision, recall, f1, _ = precision_recall_fscore_support(gts_flat, preds_flat, average='binary', zero_division=0)
        accuracy = accuracy_score(gts, preds)
        metrics["anomaly_bounds"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    if "trends" in results:
        trend_metrics = {}
        for kpi, label_dict in results["trends"].items():
            label_accuracies = {}
            total_correct = 0
            total_count = 0

            for label, correct_list in label_dict.items():
                count = len(correct_list)
                correct = sum(correct_list)

                acc = correct / count if count > 0 else 0.0
                label_accuracies[label] = acc

                total_correct += correct
                total_count += count

            overall_acc = total_correct / total_count if total_count > 0 else 0.0
            trend_metrics[kpi] = {
                "per_class_accuracy": label_accuracies,
                "overall_accuracy": overall_acc
            }
        metrics["trends"] = trend_metrics

    for category in ["mean", "variance", "periodicity"]:
        if not args.eval_tasks[category] or category not in results:
            continue
        category_metrics = {}
        for kpi, err_dict in results[category].items():
            if "mae" not in err_dict:
                continue
            mae_list = err_dict["mae"]
            mse_list = err_dict["mse"]
            mae = np.mean(mae_list) if mae_list else None
            mse = np.mean(mse_list) if mse_list else None
            category_metrics[kpi] = {"mae": mae, "mse": mse}
        metrics[category] = category_metrics

    if args.eval_tasks["anomaly_length"] and "anomaly_length" in results:
        mae = np.mean(results["anomaly_length"]["mae"]) if results["anomaly_length"]["mae"] else None
        mse = np.mean(results["anomaly_length"]["mse"]) if results["anomaly_length"]["mse"] else None
        metrics["anomaly_length"] = {"mae": mae, "mse": mse}

    if args.save_predictions and torch.distributed.get_rank() == 0:
        os.makedirs(args.predictions_dir, exist_ok=True)

        with open(os.path.join(args.predictions_dir, f"all_predictions_{args.tag}.json"), "w") as f:
            json.dump(all_predictions, f, indent=2)

        with open(os.path.join(args.predictions_dir, f"intermediate_results_{args.tag}.json"), "w") as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(args.predictions_dir, f"metrics_{args.tag}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()