"""
Preprocess TelecomTS dataset for VERL GRPO training.

This script:
1. Loads TelecomTS dataset using existing teleprism pipeline
2. Applies thinking traces from cold start CoT generation
3. Converts to VERL-compatible parquet format with:
   - prompt: User message with TS placeholder
   - target: Ground truth answer with thinking tags
   - data_source: 'telecom_qa'
   - extra_info: Metadata including timeseries, category, etc.

Usage:
    python preprocess_telecom_data.py --output_dir ./checkpoints/grpo/data --split train
    python preprocess_telecom_data.py --output_dir ./checkpoints/grpo/data --split test
"""

import argparse
import os

import json
import pickle
import base64
import logging
import random
from typing import Dict, List, Any
from tqdm import tqdm
import pandas as pd
import numpy as np

from datasets import load_dataset as hf_load_dataset
import importlib.util
from teleprism.evaluation.tasks.prompts.anomaly_tasks_prompts import anomaly_list

# Load TelePrism modules directly by file path
teleprism_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
class TelePrismNamespace:
    """Namespace to hold TelePrism modules."""
    pass
teleprism = TelePrismNamespace()
base_spec = importlib.util.spec_from_file_location(
    "teleprism_base",
    os.path.join(teleprism_src_path, 'dataset', 'base.py')
)
base_module = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base_module)
teleprism.base = base_module
qa_spec = importlib.util.spec_from_file_location(
    "teleprism_qa_templates",
    os.path.join(teleprism_src_path, 'dataset', 'qa_templates.py')
)
qa_module = importlib.util.module_from_spec(qa_spec)
qa_spec.loader.exec_module(qa_module)
teleprism.qa_templates = qa_module
load_py_path = os.path.join(teleprism_src_path, 'dataset', 'load.py')
with open(load_py_path, 'r') as f:
    load_source = f.read()
load_source = load_source.replace('from .base import TimeseriesData', '# from .base import TimeseriesData')
load_source = load_source.replace('from .qa_templates import QA_templates', '# from .qa_templates import QA_templates')
load_source = load_source.replace('from datasets import load_dataset', '# from datasets import load_dataset')

# Create and execute the modified module
load_module = type('module', (), {})()
load_module.__dict__.update({
    'TimeseriesData': base_module.TimeseriesData,
    'QA_templates': qa_module.QA_templates,
    'anomalies_type_2_id': qa_module.anomalies_type_2_id,
    'load_dataset': hf_load_dataset,
    'logging': logging,
    'random': random,
    'np': np,
    'json': json,
    'pd': pd,
    'tqdm': tqdm,
    'hashlib': __import__('hashlib'),
    'List': List,
})
exec(load_source, load_module.__dict__)
teleprism.load = load_module

# Load dataset.py source and fix relative imports
dataset_py_path = os.path.join(teleprism_src_path, 'dataset', 'dataset.py')
with open(dataset_py_path, 'r') as f:
    dataset_source = f.read()

dataset_source = dataset_source.replace('from .load import process_dataset', '# from .load import process_dataset')
dataset_source = dataset_source.replace('from .qa_templates import QA_templates', '# from .qa_templates import QA_templates')
dataset_source = dataset_source.replace('from .base import TimeseriesData, TaskDataset', '# from .base import TimeseriesData, TaskDataset')
dataset_source = dataset_source.replace('from datasets import load_dataset', '# from datasets import load_dataset')

# Create and execute the modified module
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
dataset_module = type('module', (), {})()
dataset_module.__dict__.update({
    'np': np,
    'random': random,
    'logging': logging,
    'StandardScaler': StandardScaler,
    'Optional': Optional,
    'List': List,
    'load_dataset': hf_load_dataset,
    'process_dataset': load_module.process_dataset,
    'QA_templates': qa_module.QA_templates,
    'TimeseriesData': base_module.TimeseriesData,
    'TaskDataset': base_module.TaskDataset,
})
exec(dataset_source, dataset_module.__dict__)
teleprism.dataset = dataset_module

# Get PretrainingDataset class
PretrainingDataset = dataset_module.PretrainingDataset


def serialize_array(arr: np.ndarray) -> str:
    """Serialize numpy array to base64 string for parquet storage."""
    return base64.b64encode(pickle.dumps(arr)).decode('ascii')


def create_prompt_message(question: str) -> str:
    """
    Create prompt in the exact same format as the SFT dataloader
    (dataloader.py lines 314-315).
    """
    return (
        f"<|im_start|>user\n"
        f"<|begin_of_TS|><|end_of_TS|>{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )



def create_extra_info(sample, category: str) -> Dict[str, Any]:
    """
    Create extra_info dict with metadata and serialized timeseries.
    
    This dict will be passed to the reward function.
    """
    return {
        'question_category': category,
        'timeseries': serialize_array(sample.timeseries),  # [C, T]
        'timestamp': serialize_array(np.array(sample.timestamp)),
        'anomaly_type': int(sample.anomaly_type) if hasattr(sample, 'anomaly_type') else 0,
        'sequence_length': sample.timeseries.shape[1] if sample.timeseries is not None else 0,
        'description': sample.description,
        'cot': sample.answers if isinstance(sample.answers, list) else [sample.answers],
    }


def load_and_process_telecom_data(split: str, use_thinking: bool = True) -> List[Any]:
    """
    Load TelecomTS dataset using existing TelePrism pipeline.
    
    This leverages your existing dataset.py which handles:
    - Loading from HuggingFace
    - Applying thinking traces
    - Balancing categories
    """
    print(f"Loading {split} split using TelePrism PretrainingDataset...")
    
    # Use your existing dataset class which handles all preprocessing
    dataset = PretrainingDataset(
        seq_len_channel=128,
        data_split=split,
        scale=False,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        downsampling_type="interpolate",
        pad_mode="constant",
        KPI_list=[
            "RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS",
            "UL_NPRB", "UL_SNR", "TX_Bytes", "RX_Bytes",
            "Estimated_UL_Buffer", "PRBs_DL_Current", "PRBs_UL_Current",
            "PRB_Utilization_DL", "PRB_Utilization_UL",
            "UL_Protocol", "UL_NumberOfPackets",
            "DL_Protocol", "DL_NumberOfPackets"
        ],
        descr_pretrain=False,
        use_thinking=use_thinking,
        task_list=[
            "anomaly_detection", "root_cause", "anomaly_bounds",
            "zone", "activity", "cong", "motion"
        ]
    )
    
    print(f"Loaded {len(dataset)} samples after balancing")
    return dataset.data


def convert_to_verl_format(
    samples: List[Any],
    split: str,
    use_thinking: bool = True
) -> pd.DataFrame:
    """
    Convert TelecomTS samples to VERL parquet format.
    
    Expected columns:
    - prompt: List[Dict] with role/content
    - target: List[str] with ground truth answers
    - data_source: str (always 'telecom_qa')
    - extra_info: Dict with metadata
    """
    data_rows = []
    
    for sample in tqdm(samples, desc=f"Converting {split} samples"):
        # Append valid anomaly list to root_cause questions
        question = sample.questions
        if sample.question_category == "root_cause":
            question += f" The valid anomalies are {', '.join(anomaly_list)}"

        # Create prompt (with TS placeholder)
        prompt = create_prompt_message(question)

        # Create target - just the parsed answer (no thinking tags for GRPO)
        target = sample.parsed_answer.strip()

        # Create extra_info with serialized timeseries
        extra_info = create_extra_info(sample, sample.question_category)

        row = {
            'prompt': prompt,
            'target': [target],  # VERL expects list format
            'data_source': f'telecom_qa/{sample.question_category}',
            'extra_info': extra_info,
            'reward_model': {
                'ground_truth': target  # VERL's naive reward manager expects this
            }
        }

        data_rows.append(row)

    return pd.DataFrame(data_rows)


def save_parquet(df: pd.DataFrame, output_path: str):
    """Save DataFrame to parquet with appropriate compression."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, compression='snappy')
    print(f"Saved {len(df)} samples to {output_path}")


def print_example(df: pd.DataFrame, num_examples: int = 2):
    """Print example rows for verification."""
    print("\n" + "="*80)
    print("EXAMPLE SAMPLES")
    print("="*80)
    
    for idx in range(min(num_examples, len(df))):
        row = df.iloc[idx]
        print(f"\n--- Sample {idx + 1} ---")
        print(f"Prompt: {row['prompt']}")
        print(f"Target: {row['target'][0][:200]}...")  # First 200 chars
        print(f"Data source: {row['data_source']}")
        print(f"Category: {row['extra_info']['question_category']}")
        print(f"TS shape: {pickle.loads(base64.b64decode(row['extra_info']['timeseries'])).shape}")
        print(f"Anomaly type: {row['extra_info']['anomaly_type']}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess TelecomTS for VERL GRPO')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/grpo/data',
                        help='Output directory for parquet files')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='train',
                        help='Dataset split to process')
    parser.add_argument('--use_thinking', action='store_true', default=True,
                        help='Include thinking traces in targets')
    parser.add_argument('--show_examples', action='store_true', default=True,
                        help='Print example samples after conversion')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"TelecomTS → VERL Parquet Conversion")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print(f"Thinking traces: {args.use_thinking}")
    print("="*80)
    
    # Load and process data using existing pipeline
    samples = load_and_process_telecom_data(args.split, args.use_thinking)

    # Convert to VERL format
    df = convert_to_verl_format(samples, args.split, args.use_thinking)
    
    # Save to parquet
    output_filename = f"telecom_{args.split}_grpo.parquet"
    output_path = os.path.join(args.output_dir, output_filename)
    save_parquet(df, output_path)
    
    # Print examples
    if args.show_examples:
        print_example(df)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total samples: {len(df)}")
    
    # Category distribution
    categories = {}
    for _, row in df.iterrows():
        cat = row['extra_info']['question_category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # File size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nFile size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("Conversion complete!")
    print("="*80)
    print(f"\nNext steps:")
    print(f" Process the other split if needed")
    print(f" Update launcher script paths:")
    print(f"   TRAIN_FILE='{output_path}'")
    print(f" Implement reward function in launchers/telecom_qa/reward_telecom.py")


if __name__ == '__main__':
    main()
