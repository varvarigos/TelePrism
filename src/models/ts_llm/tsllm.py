import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Dict

import torch
import torch.nn as nn

from toto.data.util.dataset import MaskedTimeseries

class FullModel(nn.Module):
    """
    FullModel integrates a time-series encoder with a language model by aligning
    the time-series embeddings and concatenating them with the LLM text embeddings.

    Args:
        ts_encoder (nn.Module): Time-series encoder module producing latent embeddings.
        align_layer (nn.Module): A linear layer to project TS embeddings to match LLM input size.
        base_model (PreTrainedModel): A language model supporting `generate()` and `forward()` with `inputs_embeds`.
    """

    def __init__(
        self,
        ts_encoder: nn.Module,
        align_layer: nn.Module,
        base_model: PreTrainedModel,
        pool_method: str = "AR"
    ):
        super().__init__()
        self.ts_encoder = ts_encoder
        self.align_layer = align_layer

        self.base_model = base_model
        self.pool_method = pool_method

    def _pool_tokens(self, tensor: torch.Tensor, method: str = "AR") -> torch.Tensor:
        if method == "AR":
            return tensor
        elif method == "mean":
            return nn.AdaptiveAvgPool1d(1)(tensor.transpose(1, 2)).transpose(1, 2)
        elif method == "max":
            return nn.AdaptiveMaxPool1d(1)(tensor.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def forward(
        self,
        ts: torch.Tensor,                        # [B, C, T]
        input_ids: torch.Tensor,                 # [B, L]
        attention_mask: torch.Tensor,            # [B, L]
        tokenizer: PreTrainedTokenizer,
        labels: Optional[torch.Tensor] = None,   # [B, L]
        scaled_params: Optional[tuple] = None,   # (mean, std) each [B, C]
        kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the combined time-series and language model.

        Args:
            ts: Time-series tensor [B, C, N, P].
            input_ids: Token IDs for text [B, L].
            attention_mask: Attention mask for text [B, L].
            labels: Labels for language modeling loss [B, L].

        Returns:
            ModelOutput with loss and logits.
        """
        device = next(self.base_model.parameters()).device
        dtype = next(self.base_model.parameters()).dtype

        ts = ts.to(device=device, dtype=dtype)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Encode and align time-series
        if self.ts_encoder is not None:
            if kwargs['model_name'] == "toto":
                timestamp_seconds = torch.zeros(ts.size(0), ts.size(1), ts.size(2), device=ts.device)
                time_interval_seconds = torch.full((ts.size(1),), 0.1, device=ts.device)
                inputs = MaskedTimeseries(
                    series=ts,
                    padding_mask=torch.full_like(ts, True, dtype=torch.bool),
                    id_mask=torch.zeros_like(ts),
                    timestamp_seconds=timestamp_seconds,
                    time_interval_seconds=time_interval_seconds,
                )   # torch.Size([B, C, T])


                ts_embed = self.ts_encoder.get_embeddings(
                    inputs.series,
                    inputs.padding_mask,
                    inputs.id_mask
                )   # [B, C, N, P]


                ts_embed = ts_embed.contiguous().view(
                    ts_embed.size(0),
                    ts_embed.size(1),
                    ts_embed.size(2) * ts_embed.size(3)
                ) # [B, C, N * P]


            elif kwargs['model_name'] == "teleencoder":
                ts_embed, _ = self.ts_encoder(ts)        # [B, C, S, d_model]
                ts_embed = ts_embed.contiguous().view(
                    ts_embed.size(0),
                    ts_embed.size(1),
                    ts_embed.size(2) * ts_embed.size(3)
                )                                        # [B, C, S*d_model]

            elif kwargs['model_name'] == "mantis":
                ts_embed = self.ts_encoder.get_embeddings(ts)  # [B, C, num_patches * hidden_dim]

            elif kwargs['model_name'] == "chronos":
                ts_embed = self.ts_encoder.get_embeddings(ts)  # [B, C, d_model]

            elif kwargs['model_name'] in (
                "autoformer", "fedformer", "informer", "nonstationary_transformer", "timesnet"
            ):
                ts_embed = self.ts_encoder.get_embeddings(ts)  # [B, T, d_model]

            else:
                ts_embed = self.ts_encoder(ts)           # [B, L_ts, D_ts]

            aligned_embed = self.align_layer(ts_embed)   # [B, L_ts, D_lm]

        else:
            aligned_embed = ts.unsqueeze(1)              # Fallback


        # Get LLM embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids).to(device, dtype) # [B, L, D]

        # Get token IDs for the special tokens
        ts_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_TS|>")
        ts_end_id = tokenizer.convert_tokens_to_ids("<|end_of_TS|>")



        # Locate start and end token positions
        ts_start_idx = (input_ids == ts_start_id) # [B, L]
        ts_end_idx = (input_ids == ts_end_id)     # [B, L]



        # All have the same length prefix, so the ts_start_idx for each B, L is the same
        ts_start_idx = ts_start_idx.nonzero(as_tuple=False)[0, 1].int() # [B]
        ts_end_idx = ts_end_idx.nonzero(as_tuple=False)[0, 1].int()     # [B]


        # Split into prefix, aligned TS embed, and suffix
        prefix = inputs_embeds[:, :ts_start_idx + 1, :]  # includes up to <|begin_of_TS|>
        suffix = inputs_embeds[:, ts_end_idx:, :]        # includes <|end_of_TS|> and beyond


        # Pool TS embeddings if necessary
        aligned_embed = self._pool_tokens(aligned_embed, method=self.pool_method)


        # Concatenate prefix, aligned TS embeddings, and suffix
        inputs_embeds = torch.cat((prefix, aligned_embed, suffix), dim=1)

        # Extend attention mask
        attention_mask = torch.cat((
            attention_mask[:, :ts_start_idx + 1],
            torch.ones(
                (attention_mask.size(0), aligned_embed.size(1)),
                device=device, dtype=torch.long
            ),
            attention_mask[:, ts_end_idx:]
        ), dim=1)

        # Extend labels with TS embeddings
        if labels is not None:
            ts_labels = torch.full(
                (input_ids.size(0), aligned_embed.size(1)),
                -100,
                dtype=labels.dtype,
                device=device
            )

            labels = torch.cat(
                (labels[:, :ts_start_idx + 1], ts_labels, labels[:, ts_end_idx:]),
                dim=1
            )

        try:
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels if labels is not None else None,
            )
        except RuntimeError as e:
            print("Error during base_model forward:", e)

        return outputs


    def generate(
        self,
        ts: torch.Tensor,                          # [C, T]
        text_inputs: Dict[str, torch.Tensor],      # {"input_ids": [L]}
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        dtype: torch.dtype,
        scaled_params: Optional[tuple] = None,   # (mean, std) each [B, C]
        kwargs: Optional[Dict[str, torch.Tensor]] = None,
        **gen_kwargs
    ) -> str:
        """
        Generate output sequence from time-series and prompt using the base model.

        Args:
            ts: Time-series input [C, T].
            text_inputs: Dict with 'input_ids' for text prompt.
            tokenizer: Tokenizer for decoding output.
            device: Device to run inference on.
            dtype: Precision (e.g., torch.bfloat16).
            **gen_kwargs: Additional generation parameters.

        Returns:
            Decoded string from the generated output.
        """

        self.eval()
        self.base_model.eval()
        self.ts_encoder.eval()
        self.align_layer.eval()

        with torch.no_grad():
            ts = ts.to(device=device, dtype=dtype)  # Adjusted for Batch Inference

            # Encode time-series
            if self.ts_encoder is not None:
                # Under Zero-3, gather the encoder's params for its forward: some
                # encoders access weights functionally (e.g. MultiheadAttention's
                # out_proj), whose per-module gather hooks don't fire during
                # generate(). No-op when params aren't partitioned.
                with deepspeed.zero.GatheredParameters(
                    list(self.ts_encoder.parameters()), modifier_rank=None
                ):
                    if kwargs['model_name'] == "toto":
                        timestamp_seconds = torch.zeros(ts.size(0), ts.size(1), ts.size(2), device=ts.device)
                        time_interval_seconds = torch.full((ts.size(1),), 0.1, device=ts.device)
                        inputs = MaskedTimeseries(
                            series=ts,
                            padding_mask=torch.full_like(ts, True, dtype=torch.bool),
                            id_mask=torch.zeros_like(ts),
                            timestamp_seconds=timestamp_seconds,
                            time_interval_seconds=time_interval_seconds,
                        )   # torch.Size([B, C, T])

                        ts_embed = self.ts_encoder.get_embeddings(
                            inputs.series,
                            inputs.padding_mask,
                            inputs.id_mask
                        )   # [B, C, N, P]
                        ts_embed = ts_embed.contiguous().view(
                            ts_embed.size(0),
                            ts_embed.size(1),
                            ts_embed.size(2) * ts_embed.size(3)
                        ) # [B, C, N * P]

                    elif kwargs['model_name'] == "teleencoder":
                        ts_embed, _ = self.ts_encoder(ts)        # [B, C, S, d_model]
                        ts_embed = ts_embed.contiguous().view(
                            ts_embed.size(0),
                            ts_embed.size(1),
                            ts_embed.size(2) * ts_embed.size(3)
                        )                                        # [B, C, S*d_model]

                    elif kwargs['model_name'] == "mantis":
                        ts_embed = self.ts_encoder.get_embeddings(ts)  # [B, C, num_patches * hidden_dim]

                    elif kwargs['model_name'] == "chronos":
                        ts_embed = self.ts_encoder.get_embeddings(ts)  # [B, C, d_model]

                    elif kwargs['model_name'] in (
                        "autoformer", "fedformer", "informer", "nonstationary_transformer", "timesnet"
                    ):
                        ts_embed = self.ts_encoder.get_embeddings(ts)  # [B, T, d_model]

                    else:
                        ts_embed = self.ts_encoder(ts)           # [B, L_ts, D_ts]

                aligned_embed = self.align_layer(ts_embed)   # [B, L_ts, D_lm]

            else:
                aligned_embed = ts.unsqueeze(1)              # Fallback



            # Tokenize input
            input_ids = text_inputs["input_ids"].to(device) # Removed unsqueeze for batch inference


            attention_mask = text_inputs["attention_mask"].to(device)

            ## Switch right to left ##
            
            pad_token_id = tokenizer.pad_token_id
            B, L = input_ids.shape
            seq_lens = attention_mask.sum(dim=1)  # [B]

            # Compute shift for each sequence
            shift = (L - seq_lens).unsqueeze(1)  # [B,1]


            # Create an index grid
            arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

            # Mask: True where we keep original tokens
            left_mask = arange >= shift

            # Create a new tensor filled with pad_token_id
            left_input_ids = torch.full_like(input_ids, pad_token_id)

            # Shifted indices: for valid tokens, map to original positions
            source_idx = (arange - shift).clamp(min=0)

            # Use torch.where to place valid tokens
            left_input_ids = torch.where(left_mask, input_ids[torch.arange(B).unsqueeze(1), source_idx], left_input_ids)

            input_ids = left_input_ids


            attention_mask = left_mask.to(input_ids.dtype)


            # Get LLM embeddings
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids).to(device, dtype) # [B, L, D]


            # Get token IDs for the special tokens
            ts_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_TS|>")
            ts_end_id = tokenizer.convert_tokens_to_ids("<|end_of_TS|>")

            # Locate start and end token positions
            ts_start_idx = (input_ids == ts_start_id) # [B, L]
            ts_end_idx = (input_ids == ts_end_id)     # [B, L]

            # All have the same length prefix, so the ts_start_idx for each B, L is the same
            ts_start_idx = ts_start_idx.nonzero(as_tuple=False)[0, 1].int() # [B]
            ts_end_idx = ts_end_idx.nonzero(as_tuple=False)[0, 1].int()     # [B]

            # Split into prefix, aligned TS embed, and suffix
            prefix = inputs_embeds[:, :ts_start_idx + 1, :]  # includes up to <|begin_of_TS|>
            suffix = inputs_embeds[:, ts_end_idx:, :]        # includes <|end_of_TS|> and beyond


            # Pool TS embeddings if necessary
            aligned_embed = self._pool_tokens(aligned_embed, method=self.pool_method)


            # Concatenate prefix, aligned TS embeddings, and suffix
            inputs_embeds = torch.cat((prefix, aligned_embed, suffix), dim=1)
   
            attention_mask = torch.cat((
            attention_mask[:, :ts_start_idx + 1],
            torch.ones(
                (attention_mask.size(0), aligned_embed.size(1)),
                device=device, dtype=torch.long
            ),
            attention_mask[:, ts_end_idx:]
            ), dim=1)

            # Generate output sequence using the base model's generate function
            generated_ids = self.base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            return [tokenizer.decode(generated_id, skip_special_tokens=False) for generated_id in generated_ids]
