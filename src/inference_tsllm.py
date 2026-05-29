import json
import os
from pathlib import Path
import yaml
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
import deepspeed

from tqdm import tqdm
from typing import Tuple
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig

from types import SimpleNamespace
from teleprism.encoders import Autoformer, FEDformer, Informer, NonStationary_Transformer, TimesNet
from teleprism.encoders.teleencoder.teleEncoder import TeleEncoder
from teleprism.utils.utils import AlignLayer
from teleprism.models.ts_llm.tsllm import FullModel
from teleprism.dataset.dataloader import get_dataloader, get_eval_dataloader
from teleprism.encoders.wrappers import (
    ChronosEncoder, build_chronos_encoder,
    mantis_get_embeddings, toto_get_embeddings, tslib_get_embeddings,
    TSLIB_MODELS,
)

from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
from toto.model.backbone import TotoBackbone
from mantis.architecture import Mantis8M, MantisV1

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------------------------
# Utility: Plot a batch of input time-series
# ------------------------------------------------------------------------------

def plot_time_series(ts_tensor: torch.Tensor, kpi_names: list, idx: int, store_dir: Path) -> None:
    """
    Plots the time-series in ts_tensor and saves it to disk.

    Args:
        ts_tensor (Tensor): shape [C, T]
        kpi_names (list): List of KPI/channel names
        rank (int): GPU rank (used for filename)
    """
    ts_np = ts_tensor.detach().cpu().float().transpose(0, 1).numpy()  # [T, C]
    seq_len = ts_np.shape[0]

    fig, axs = plt.subplots(nrows=7, ncols=3, figsize=(18, 14), constrained_layout=True)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(kpi_names):
            ax.plot(np.arange(seq_len), ts_np[:, i])
            ax.set_title(kpi_names[i])
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
        else:
            ax.axis("off")

    plt.suptitle("Input Time Series (18 KPIs)", fontsize=16)
    plt.savefig(store_dir / f"predictions_{idx}.png", dpi=300)

# ------------------------------------------------------------------------------
# Load model, tokenizer, engine, and dataloader from checkpoint
# ------------------------------------------------------------------------------

def load_model(args: argparse.Namespace, ds_config: dict, load_data: bool = True, eval: bool = False) -> Tuple[torch.nn.Module, AutoTokenizer, torch.utils.data.DataLoader, torch.device, torch.dtype]:
    """
    Initializes DeepSpeed engine, model, tokenizer, and dataloader.

    Returns:
        model (nn.Module)
        tokenizer (AutoTokenizer)
        dataloader (DataLoader)
        device (torch.device)
        dtype (torch.dtype)
    """
    deepspeed.init_distributed(dist_backend="nccl", init_method="env://")

    # Determine precision
    dtype = torch.bfloat16 if ds_config["bf16"]["enabled"] else (
            torch.float16 if ds_config["fp16"]["enabled"] else torch.float32)

    # Setup device info
    args.rank = dist.get_rank()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.device = torch.device(f"cuda:{args.local_rank}")

    # LLM base model
    hfdsc = HfDeepSpeedConfig(ds_config)
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model, trust_remote_code=True, torch_dtype=dtype
        )

    if args.use_lora:
    # Apply LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)



    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|begin_of_TS|>", "<|end_of_TS|>", "<|CLS|>",
                "<|activity|>", "</activity|>",
                "<|zone|>", "</zone|>",
                "<|root_cause|>", "</root_cause|>",
                "<|mean|>", "</mean|>",
                "<|variance|>", "</variance|>",
                "<|trends|>", "</trends|>",
                "<|periodicity|>", "</periodicity|>",
                "<|cong|>", "</cong|>",
                "<|mobility|>", "</mobility|>",
                "<|anomaly_detection|>", "</anomaly_detection|>",
                "<|anomaly_bounds|>", "</anomaly_bounds|>",
                "<|anomaly_length|>", "</anomaly_length|>",
            ]
        }
    )

    model.resize_token_embeddings(len(tokenizer))

    if ds_config["activation_checkpointing"]["partition_activations"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if 'llama' in args.llm_model.lower():
        tokenizer.pad_token = '<|finetune_right_pad_id|>'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Time-series encoder + alignment layer
    if args.model_name == "toto":
        if args.toto_model["use_pretrained"]:
            ts_encoder = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(args.device)
            ts_encoder.compile()
            ts_encoder = TotoForecaster(ts_encoder.model)
            ts_encoder = ts_encoder.model
            embed_dim = 768 * 2
        else:
            sc = args.toto_model["scratch"]
            ts_encoder = TotoBackbone(
                patch_size=sc["patch_size"],
                stride=sc["stride"],
                embed_dim=sc["embed_dim"],
                num_layers=sc["num_layers"],
                num_heads=sc["num_heads"],
                mlp_hidden_dim=sc["mlp_hidden_dim"],
                dropout=0.1,
                spacewise_every_n_layers=sc["spacewise_every_n_layers"],
                scaler_cls="<class 'model.scaler.CausalPatchStdMeanScaler'>",
                output_distribution_classes=["<class 'model.distribution.MixtureOfStudentTsOutput'>"],
                output_distribution_kwargs={"k_components": sc["k_components"]},
                spacewise_first=False,
                use_memory_efficient_attention=False,
            ).to(args.device)
            n_patches = (sc["seq_len"] - sc["patch_size"]) // sc["stride"] + 1
            embed_dim = n_patches * sc["embed_dim"]

        ts_encoder.get_embeddings = toto_get_embeddings.__get__(
            ts_encoder, TotoBackbone
        )

        for param in ts_encoder.parameters():
            param.requires_grad = args.toto_model["train"]

    elif args.model_name == "teleencoder":
        arch = args.teleencoder_model["architecture"]
        ctx  = args.teleencoder_model["context"]
        exp  = args.teleencoder_model["experts"]
        rtr  = args.teleencoder_model["router"]
        patch_size   = arch["patch"]["size"]
        patch_stride = arch["patch"]["stride"]
        seq_len      = arch["seq_len"]
        d_model      = arch["d_model"]
        num_patches  = 1 + (seq_len - patch_size) // patch_stride
        embed_dim    = num_patches * d_model
        ts_encoder = TeleEncoder(
            num_channels=len(args.data["KPI_list"]),
            seq_len=seq_len,
            d_model=d_model,
            num_heads=arch["num_heads"],
            num_layers=arch["num_layers"],
            d_ff=arch["d_ff"],
            num_experts=exp["num_experts"],
            top_k=exp["top_k"],
            K=ctx["K"],
            patch_size=patch_size,
            patch_stride=patch_stride,
            dropout=arch["dropout"],
            router_type=rtr["type"],
            ctx_weight=rtr["ctx_weight"],
            router_temperature=rtr["temperature"],
            attn_mode=arch["attn_mode"],
            scale_encoder=ctx["scale_encoder"],
            d_node=ctx["graph"]["d_node"],
            n_gnn_layers=ctx["graph"]["n_gnn_layers"],
            gnn_alpha_init=ctx["graph"]["gnn_alpha_init"],
            gnn_tau_init=ctx["graph"]["gnn_tau_init"],
            sparse_inference=exp["sparse_inference"],
            router_init_std=exp["router_init_std"],
            expert_init_gain=exp["expert_init_gain"],
            router_jitter_std=exp["router_jitter_std"],
            dense_routing_warmup_epochs=exp["dense_routing_warmup_epochs"],
        ).to(args.device, dtype=dtype)
        if args.model_status["pretrained"]:
            ckpt = torch.load(args.model_status["path_to_checkpoint"])
            ts_encoder.load_state_dict(ckpt)

    elif args.model_name == "mantis":
        if args.mantis_model["use_pretrained"]:
            ts_encoder = Mantis8M.from_pretrained("paris-noah/Mantis-8M").to(args.device).to(dtype)
            ts_encoder.get_embeddings = mantis_get_embeddings.__get__(ts_encoder, Mantis8M)
        else:
            sc = args.mantis_model["scratch"]
            ts_encoder = MantisV1(
                hidden_dim=sc["hidden_dim"],
                transf_depth=sc["transf_depth"],
                transf_num_heads=sc["transf_num_heads"],
                transf_mlp_dim=sc["transf_mlp_dim"],
                transf_dim_head=sc["transf_dim_head"],
                device=str(args.device),
            ).to(dtype)
            ts_encoder.get_embeddings = mantis_get_embeddings.__get__(ts_encoder, MantisV1)

        embed_dim = ts_encoder.hidden_dim

        for param in ts_encoder.parameters():
            param.requires_grad = args.mantis_model["train"]

    elif args.model_name == "chronos":
        ts_encoder, embed_dim = build_chronos_encoder(
            args.chronos_model, dtype=dtype, device=args.device
        )

        for param in ts_encoder.parameters():
            param.requires_grad = args.chronos_model["train"]

    elif args.model_name in TSLIB_MODELS:
        model_map = {
            "autoformer": (Autoformer, "autoformer_model"),
            "fedformer": (FEDformer, "fedformer_model"),
            "informer": (Informer, "informer_model"),
            "nonstationary_transformer": (NonStationary_Transformer, "nonstationary_transformer_model"),
            "timesnet": (TimesNet, "timesnet_model"),
        }
        ModelClass, cfg_key = model_map[args.model_name]
        cfg = getattr(args, cfg_key)
        num_channels = len(args.data["KPI_list"])
        ns = SimpleNamespace(
            seq_len=args.seq_len_channel,
            label_len=0,
            pred_len=0,
            enc_in=num_channels,
            dec_in=num_channels,
            **cfg,
        )
        embed_dim = cfg["d_model"]
        ts_encoder = ModelClass.Model(ns).to(args.device, dtype=dtype)
        ts_encoder.get_embeddings = tslib_get_embeddings.__get__(ts_encoder, type(ts_encoder))

        for param in ts_encoder.parameters():
            param.requires_grad = cfg.get("train", False)

    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    align_layer = AlignLayer(embed_dim, model.config.hidden_size).to(args.device, dtype=dtype)

    # Optionally use classification head
    if args.classification["use_head"]:
        task = args.classification["task"]
        if task == 'root_cause':
            num_classes = 11
        elif task == 'anomaly_detection':
            num_classes = 2
        elif task == 'zone':
            num_classes = 4
        elif task == 'activity':
            num_classes = 3
        elif task in ["cong", "motion"]:
            num_classes = 2
        else:
            raise ValueError(f"Unknown classification task: {task}")

        head = torch.nn.Linear(model.config.hidden_size, num_classes).to(args.device, dtype=dtype)
    else:
        head = None
    

    if not args.train_llm:
        for param in model.parameters():
            param.requires_grad = False

    # Full hybrid model
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        full_model = FullModel(ts_encoder, align_layer, model, head)


    # Remove scheduler from ds_config if it exists
    ds_config.pop("scheduler", None)

    # Initialize DeepSpeed engine
    full_model, *_ = deepspeed.initialize(
        model=full_model,
        model_parameters=filter(lambda p: p.requires_grad, full_model.parameters()),
        config_params=ds_config,
    )

    full_model.load_checkpoint(
        args.checkpoint_dir,
        tag=args.tag,
        load_module_strict=True,
        load_optimizer_states=False,  # Don't load optimizer state
        load_lr_scheduler_states=False,  # Don't load scheduler state
        load_module_only=True  # Only load model weights
    )

    if not load_data:
        return full_model, tokenizer, args.device, dtype
    
    # DataLoader
    if not eval:
        dataloader = get_dataloader(args, tokenizer, "test")
    else:
        dataloader = get_eval_dataloader(args, tokenizer, "test")
    return full_model, tokenizer, dataloader, args.device, dtype


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


# ------------------------------------------------------------------------------
# Main Inference Function
# ------------------------------------------------------------------------------

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
    parser.add_argument("--pin_memory", type=bool, default=True)

    deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Merge args with defaults
    with open("configs/train_tsllm.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**{**cfg, **vars(args)})

    # Initialize model + loader
    model, tokenizer, dataloader, device, dtype = load_model(args, ds_config)

    # Inference loop
    print("Running inference...")
    for step, batch in enumerate(tqdm(dataloader)):
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        ts = batch.timeseries.to(device, dtype=dtype)
        labels = batch.labels.to(device)

        # Strip input_ids to remove answer part for generation
        strip = torch.tensor(
            tokenizer.encode("<|im_start|>assistant", add_special_tokens=False),
            device=input_ids.device
        )

        # Find the last token index that matches the strip
        match = (input_ids.unfold(1, strip.size(0), 1) == strip).all(dim=2)
        last_token_idx = torch.tensor(
            [m.nonzero(as_tuple=True)[0][-1].item() + strip.size(0) for m in match],
            device=input_ids.device
        )

        input_ids_cut = input_ids[:, :last_token_idx.max()]
        attention_mask_cut = attention_mask[:, :last_token_idx.max()]

        row_range = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)  # [1, L]
        keep_mask = row_range < last_token_idx.unsqueeze(1)
        pad_id = tokenizer.pad_token_id
        input_ids = input_ids.masked_fill(~keep_mask, pad_id)
        attention_mask = attention_mask & keep_mask

        text_inputs = {
            "input_ids": input_ids_cut,
            "attention_mask": attention_mask_cut,
        }

        model.eval()
        with torch.no_grad():
            ts = scale_kpi_units_(ts, args.data["KPI_list"])

            scaled_params = None

            output = model.generate(
                ts=ts,
                text_inputs=text_inputs,
                scaled_params=scaled_params,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                kwargs={"model_name": args.model_name}
            )

        # Print results
        for idx, (sample_input_ids, sample_labels, sample_output) in enumerate(zip(input_ids, labels, output)):
            prompt = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
            label_ids_clean = sample_labels[sample_labels != -100]
            true_answer_full = tokenizer.decode(label_ids_clean, skip_special_tokens=True)
            true_answer_full = true_answer_full.split("<|im_end|>")[0]

            print("========= prompt =========")
            print(prompt)
            print("========= true answer =========")
            print(true_answer_full)
            print("========= model output =========")
            print(sample_output)

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
