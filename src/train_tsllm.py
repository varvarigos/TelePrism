import math
import os
import yaml
import json
import torch
import random
import argparse
import deepspeed
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import safetensors.torch as safetorch
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace
from teleprism.utils.utils import AlignLayer
from teleprism.encoders.teleencoder.teleEncoder import TeleEncoder
from teleprism.models.ts_llm.tsllm import FullModel
from teleprism.dataset.dataloader import get_dataloader
from teleprism.encoders import Autoformer, FEDformer, Informer, NonStationary_Transformer, TimesNet
from teleprism.encoders.wrappers import (
    ChronosEncoder, build_chronos_encoder,
    mantis_get_embeddings, toto_get_embeddings, tslib_get_embeddings,
    TSLIB_MODELS,
)

from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
from toto.model.backbone import TotoBackbone
from mantis.architecture import Mantis8M, MantisV1


SEED = 42
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def cleanup() -> None:
    """Cleans up the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Destroyed process group")


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

@torch.no_grad()
def evaluate(val_loader, full_model, args, tokenizer, ds_config, epoch, evaluator_name="eval"):
    """
    Runs evaluation to compute average validation loss
    in distributed DeepSpeed training.

    Returns:
        avg_loss (float) on global rank 0, otherwise 0.
    """

    # Switch to eval mode
    full_model.eval()

    dtype = (
        torch.bfloat16 if ds_config["bf16"]["enabled"]
        else torch.float16 if ds_config["fp16"]["enabled"]
        else torch.float32
    )

    loss_total = torch.zeros((), device=args.device)
    total_samples = torch.zeros((), device=args.device)

    if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
        val_loader.sampler.set_epoch(epoch)

    max_eval_batches = getattr(args, "max_eval_batches", None)
    for batch_idx, batch in enumerate(
        tqdm(val_loader, desc=f"Running {evaluator_name}...", total=len(val_loader))
    ):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break
        with torch.amp.autocast(device_type=args.device.type, dtype=dtype, enabled=True):
            # Preprocess time series the same way as training
            ts = batch.timeseries
            ts = scale_kpi_units_(ts, args.data["KPI_list"])

            scaled_params = None

            # Forward pass
            outputs = full_model(
                ts=ts,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                tokenizer=tokenizer,
                labels=batch.labels,
                scaled_params=scaled_params,
                kwargs={"model_name": args.model_name},
            )

            loss = outputs.loss.detach()

        # Accumulate
        loss_total += loss * batch.input_ids.size(0)
        total_samples += batch.input_ids.size(0)

    avg_loss = loss_total / total_samples

    full_model.train()
    return avg_loss


def train(args: dict, ds_config: dict) -> None:
    """
    Main training loop using DeepSpeed and LoRA fine-tuning.

    Args:
        args (dict): Training arguments.
        ds_config (dict): DeepSpeed configuration dictionary.
    """
    try:
        # Initialize distributed backend
        deepspeed.init_distributed(dist_backend="nccl", init_method="env://")

        args = argparse.Namespace(**args)
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = dist.get_rank()
        args.device = torch.device(f"cuda:{args.local_rank}")
        set_seed(SEED)

        # Determine precision
        dtype = (
            torch.bfloat16
            if ds_config["bf16"]["enabled"]
            else torch.float16 if ds_config["fp16"]["enabled"] else torch.float32
        )

        # Create model base
        hfdsc = HfDeepSpeedConfig(ds_config)
        monitor = MonitorMaster(DeepSpeedConfig(ds_config).monitor_config)

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

        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model,
            trust_remote_code=True,
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|begin_of_TS|>", "<|end_of_TS|>",
                ]
            }
        )

        model.resize_token_embeddings(len(tokenizer))

        # For activation checkpointing
        if ds_config["activation_checkpointing"]["partition_activations"]:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        if 'llama' in args.llm_model.lower():
            tokenizer.pad_token = '<|finetune_right_pad_id|>'

        # Time-series encoder and alignment layer
        if args.model_name == "toto":
            if args.toto_model["use_pretrained"]:
                ts_encoder = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(args.device)
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

            if args.pretrained_encoder["pretrained"]:
                state_dict = safetorch.load_file(
                    os.path.join(args.pretrained_encoder["path_to_TS"], "model.safetensors")
                )
                backbone_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
                ts_encoder.load_state_dict(backbone_state_dict)

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

            if args.pretrained_encoder["pretrained"]:
                state_dict = safetorch.load_file(
                    os.path.join(args.pretrained_encoder["path_to_TS"], "model.safetensors")
                )
                ts_encoder.load_state_dict(state_dict)

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

            if args.pretrained_encoder["pretrained"]:
                state_dict = safetorch.load_file(
                    os.path.join(args.pretrained_encoder["path_to_TS"], "model.safetensors")
                )
                ts_encoder.load_state_dict(state_dict)

        elif args.model_name == "chronos":
            ts_encoder, embed_dim = build_chronos_encoder(
                args.chronos_model, dtype=dtype, device=args.device
            )

            for param in ts_encoder.parameters():
                param.requires_grad = args.chronos_model["train"]

            if args.pretrained_encoder["pretrained"]:
                state_dict = safetorch.load_file(
                    os.path.join(args.pretrained_encoder["path_to_TS"], "model.safetensors")
                )
                ts_encoder.load_state_dict(state_dict)

        elif args.model_name in TSLIB_MODELS:
            # ----------------------------------------------------------------
            # Time-Series-Library encoders: Autoformer, FEDformer, Informer,
            # NonStationary_Transformer, TimesNet
            # All share the same instantiation pattern via a SimpleNamespace cfg.
            # ----------------------------------------------------------------
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
                param.requires_grad = cfg.get("train", True)

            if args.pretrained_encoder["pretrained"]:
                state_dict = safetorch.load_file(
                    os.path.join(args.pretrained_encoder["path_to_TS"], "model.safetensors")
                )
                ts_encoder.load_state_dict(state_dict)

        else:
            raise ValueError(f"Unknown model name: {args.model_name}")

        align_layer = AlignLayer(embed_dim, model.config.hidden_size).to(
            args.device, dtype=dtype
        )
        if args.pretrained_encoder["pretrained"] == True:
            if '.safetensors' in args.pretrained_encoder["path_to_align"]:
                ckpt_align = safetorch.load_file(args.pretrained_encoder["path_to_align"])
            else:
                ckpt_align = torch.load(args.pretrained_encoder["path_to_align"], weights_only=False)
            align_layer.load_state_dict(ckpt_align)

        if not args.train_llm:
            for param in model.parameters():
                param.requires_grad = False


        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            full_model = FullModel(ts_encoder, align_layer, model)


        print(
            f"Total trainable parameters: {sum(p.numel() for p in full_model.parameters() if p.requires_grad)}"
        )

        # Build dataloaders
        dataloader = get_dataloader(args, tokenizer, "train")
        val_loader = get_dataloader(args, tokenizer, "test")

        # Scheduler steps
        if ds_config["scheduler"]["type"] == "WarmupCosineLR":
            dataset_size = len(dataloader.dataset)  # total samples
            micro_batch = int(ds_config["train_micro_batch_size_per_gpu"])
            gas = int(ds_config.get("gradient_accumulation_steps", 1))
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            # samples consumed per optimizer step across all GPUs
            global_batch_per_step = micro_batch * gas * world_size

            steps_per_epoch = math.ceil(dataset_size / global_batch_per_step)
            total_steps = steps_per_epoch * args.epochs  # optimizer steps

            ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
            ds_config["scheduler"]["params"]["warmup_num_steps"] = int(
                0.01 * total_steps
            )

        # DeepSpeed engine init
        full_model, *_ = deepspeed.initialize(
            model=full_model,
            model_parameters=filter(lambda p: p.requires_grad, full_model.parameters()),
            config_params=ds_config,
        )

        # Resume trainable weights from a checkpoint (consolidated pytorch_model.pt
        # or a legacy DeepSpeed shard dir). GatheredParameters(modifier_rank=0)
        # sets each param correctly under Zero-3. Frozen base weights are skipped
        # (identical to the HF download already loaded by from_pretrained()).
        if args.deepspeed_pretrained["status"] == True:
            ckpt_dir = args.deepspeed_pretrained["checkpoint"]
            tag = args.deepspeed_pretrained["tag"]
            consolidated_pt = os.path.join(ckpt_dir, tag, "pytorch_model.pt")
            if os.path.isfile(consolidated_pt):
                if args.rank == 0:
                    print(f"[resume] loading consolidated fp32 checkpoint {consolidated_pt}", flush=True)
                state_dict = torch.load(consolidated_pt, map_location="cpu", weights_only=True)
            else:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                if args.rank == 0:
                    print(f"[resume] consolidating legacy Zero shard checkpoint {ckpt_dir} (tag={tag})", flush=True)
                state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=tag)
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                          for k, v in state_dict.items()}
            loaded = 0
            missing = []
            skipped_frozen = 0
            for name, param in full_model.module.named_parameters():
                if not param.requires_grad:
                    skipped_frozen += 1
                    continue
                if name not in state_dict:
                    missing.append(name)
                    continue
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if args.rank == 0:
                        param.data.copy_(state_dict[name].to(param.dtype).to(param.device))
                loaded += 1
            if args.rank == 0:
                print(
                    f"[resume] trainable_loaded={loaded} missing_trainable={len(missing)} "
                    f"frozen_skipped={skipped_frozen}",
                    flush=True,
                )
                if missing:
                    print(f"  WARNING: first 5 missing trainable: {missing[:5]}")
            del state_dict

        # Training loop
        micro_loss_accum = 0.0

        full_model.train()
        for epoch in range(args.epochs):
            dataloader.sampler.set_epoch(epoch)
            for micro_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                with torch.amp.autocast(device_type=args.device.type, dtype=dtype, enabled=True):
                    ts = batch.timeseries
                    ts = scale_kpi_units_(ts, args.data["KPI_list"])

                    scaled_params = None

                    outputs = full_model(
                        ts=ts,
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        tokenizer=tokenizer,
                        labels=batch.labels,
                        scaled_params=scaled_params,
                        kwargs={"model_name": args.model_name},
                    )
                    loss = outputs.loss

                full_model.backward(loss)

                # Log gradient norms at optimizer step
                if full_model.is_gradient_accumulation_boundary():
                    ts_enc_sq = torch.zeros((), device=args.device, dtype=dtype)
                    llm_sq = torch.zeros((), device=args.device, dtype=dtype)

                    for name, p in full_model.named_parameters():
                        full_grad = deepspeed.utils.safe_get_full_grad(p)
                        if full_grad is None or not p.requires_grad:
                            continue
                        g2 = (full_grad.detach().float() / gas).pow(2).sum()
                        if "ts_encoder" in name or "align_layer" in name:
                            ts_enc_sq += g2
                        else:
                            llm_sq += g2

                    if dist.is_initialized():
                        dist.all_reduce(ts_enc_sq, op=dist.ReduceOp.SUM)
                        dist.all_reduce(llm_sq, op=dist.ReduceOp.SUM)

                    ts_enc_gnorm = torch.sqrt((ts_enc_sq) / world_size)
                    llm_gnorm = torch.sqrt((llm_sq) / world_size)

                    monitor.write_events(
                        [
                            (
                                "train/ts_enc_gnorm",
                                float(ts_enc_gnorm.item()),
                                full_model.global_samples,
                            ),
                            (
                                "train/llm_gnorm",
                                float(llm_gnorm.item()),
                                full_model.global_samples,
                            ),
                        ]
                    )

                full_model.step()

                # Accumulate scalar loss for this GAS window
                micro_loss_accum += float(loss.detach().item())

                # Log loss and lr at optimizer step
                if full_model.is_gradient_accumulation_boundary():
                    # Avg over micro-batches in this window
                    avg_micro_loss = micro_loss_accum / gas

                    # DP-reduce across ranks
                    loss_tensor = torch.tensor(avg_micro_loss, device=args.device)
                    if dist.is_initialized():
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        loss_tensor /= world_size

                    step_id = int(full_model.global_steps)

                    lr = float(full_model.optimizer.param_groups[0]["lr"])

                    print("LEARNING RATE", lr)

                    monitor.write_events(
                        [
                            (
                                "train/loss",
                                float(loss_tensor.item()),
                                full_model.global_samples,
                            ),
                            ("train/lr", lr, full_model.global_samples),
                        ]
                    )

                    # Reset accumulator for next window
                    micro_loss_accum = 0.0

                    if args.rank == 0:
                        tqdm.write(
                            f"Epoch {epoch}, OptStep {step_id}, Loss: {loss_tensor.item():.4f}"
                        )

                # Debug short-circuit
                if args.max_steps_per_epoch is not None and (micro_idx + 1) >= args.max_steps_per_epoch:
                    if args.rank == 0:
                        tqdm.write(f"[debug] max_steps_per_epoch={args.max_steps_per_epoch} reached, breaking inner loop")
                    break


            # ----------------------------------------------------------
            # END-OF-EPOCH BOUNDARY (eval + save).
            # Instrumented heavily because this is where overnight runs
            # have died. Each step prints a marker on rank 0 so a future
            # crash log will pinpoint exactly which step blew up.
            # ----------------------------------------------------------
            if args.rank == 0:
                print(f"[boundary] epoch {epoch} inner loop done — starting eval", flush=True)

            # Evaluation
            if epoch % args.eval_interval == 0:
                if args.model_name == "teleencoder":
                    full_model.module.ts_encoder.force_dense_eval = True
                eval_loss = evaluate(
                    val_loader=val_loader,
                    full_model=full_model,
                    args=args,
                    tokenizer=tokenizer,
                    ds_config=ds_config,
                    epoch=epoch,
                    evaluator_name="eval"
                )
                if args.model_name == "teleencoder":
                    full_model.module.ts_encoder.force_dense_eval = False

                if args.rank == 0:
                    print(f"[boundary] epoch {epoch} eval done — all_reduce eval_loss", flush=True)

                eval_loss_tensor = torch.tensor(eval_loss.item(), device=args.device)
                if dist.is_initialized():
                    dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                    eval_loss_tensor /= world_size

                monitor.write_events(
                    [
                        (
                            "eval/loss",
                            float(eval_loss_tensor.item()),
                            full_model.global_samples,
                        ),
                    ]
                )
                if args.rank == 0:
                    print(f"[boundary] epoch {epoch} eval_loss={float(eval_loss_tensor.item()):.4f} logged", flush=True)

            # Save checkpoint — LOCAL-ONLY save path (replaces the
            # full_model.save_checkpoint + GatheredParameters chain).
            #
            # Why: full_model.save_checkpoint() and the all-params
            # GatheredParameters trigger big multi-rank NCCL collectives
            # that have been the source of every overnight SIGSEGV.
            # safe_get_full_fp32_param is a local-only API that returns
            # the rank's slice of the fp32 master without any NCCL op.
            # Only TRAINABLE params are saved (~425M, ~1.7 GB); frozen
            # Qwen base weights are identical to the HF download and do
            # not need re-saving.
            if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
                from deepspeed.utils import safe_get_full_fp32_param
                save_dir = os.path.join(args.output_model, f"epoch-{epoch}-{micro_idx}")
                if args.rank == 0:
                    print(f"[boundary] epoch {epoch} entering save → {save_dir}", flush=True)
                    os.makedirs(save_dir, exist_ok=True)
                consolidated = {}
                saved_n = skipped_frozen = skipped_none = 0
                for name, param in full_model.module.named_parameters():
                    if not param.requires_grad:
                        skipped_frozen += 1
                        continue
                    full_param = safe_get_full_fp32_param(param)
                    if full_param is None:
                        skipped_none += 1
                        continue
                    if args.rank == 0:
                        consolidated[name] = full_param.detach().to("cpu").clone()
                        saved_n += 1
                if args.rank == 0:
                    out_path = os.path.join(save_dir, "pytorch_model.pt")
                    torch.save(consolidated, out_path)
                    print(
                        f"[boundary] epoch {epoch} saved {saved_n} trainable params to {out_path} "
                        f"(skipped_frozen={skipped_frozen} skipped_none={skipped_none})",
                        flush=True,
                    )
                del consolidated
                if args.rank == 0:
                    print(f"[boundary] epoch {epoch} save complete — moving to next epoch", flush=True)

    finally:
        cleanup()


def main() -> None:
    with open("configs/ds_conf.json", "r") as f:
        ds_config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--output_model", type=str, default="/data/andreas/ChronoSense_Experiments_LATEST/checkpoints/cold_start"
    )
    parser.add_argument(
        "--batch_size", type=int, default=ds_config["train_micro_batch_size_per_gpu"]
    )
    parser.add_argument("--epochs", type=int, default=60)
    # Debug knobs (default None = no break) for testing the epoch boundary
    # without waiting hours per epoch. Set both to small ints (e.g. 5) to
    # smoke-test save/eval/load across many epoch boundaries quickly.
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, break the inner training loop after N micro-batches per epoch.")
    parser.add_argument("--max_eval_batches", type=int, default=None,
                        help="If set, only run N eval batches per epoch.")
    parser.add_argument("--project_name", type=str, default="TelePrism")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seq_len_channel", type=int, default=128)
    parser.add_argument("--upsampling_type", type=str, default="pad")
    parser.add_argument("--downsampling_type", type=str, default="average")
    parser.add_argument("--pad_mode", type=str, default="constant")
    parser.add_argument("--upsampling_pad_direction", type=str, default="forward")
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--grad_clip", type=float, default=ds_config["gradient_clipping"]
    )

    deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    with open("configs/train_tsllm.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    args_dict = {**cfg, **vars(args)}
    train(args_dict, ds_config)


if __name__ == "__main__":
    main()
