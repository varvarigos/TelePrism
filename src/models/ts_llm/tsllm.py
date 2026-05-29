import torch
import torch.nn as nn
import torch.nn.functional as F
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
        head: Optional[nn.Module] = None,
        pool_method: str = "AR"
    ):
        super().__init__()
        self.ts_encoder = ts_encoder
        self.align_layer = align_layer

        self.base_model = base_model
        self.pool_method = pool_method

        self.use_cls_head = False if head is None else True
        if self.use_cls_head:
            self.head = head

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
        gt_class: Optional[torch.Tensor] = None, # [B] classification labels if head is not None
        scaled_params: Optional[tuple] = None,   # (mean, std) each [B, C]
        weight_boost: Optional[int] = None,    # weight boost for loss
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
                output_hidden_states=True if self.use_cls_head else False,
            )
        except RuntimeError as e:
            print("Error during base_model forward:", e)

        # If classification head is used, compute classification logits and loss
        if self.use_cls_head:
            # Extract CLS token hidden states
            cls_token_id = tokenizer.convert_tokens_to_ids('<|CLS|>')
            cls_positions = (input_ids == cls_token_id).nonzero(as_tuple=False)  # [B, 2] (batch_idx, seq_idx)
            cls_hidden_states = outputs.hidden_states[-1][
                cls_positions[:, 0], cls_positions[:, 1], :
            ]  # [B, H]
            logits = self.head(cls_hidden_states)  # [B, num_classes]
            outputs.loss = F.cross_entropy(logits, gt_class)

            preds = torch.argmax(logits, dim=1)
            correct = (preds == gt_class).sum().item()
            outputs.accuracy = torch.tensor(correct / gt_class.size(0), device=device)

        else:
            if weight_boost is not None:
                # Apply weight boosting to the loss
                outputs.loss = self._boost_loss(outputs, labels, tokenizer, weight_boost, device)

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

            if self.use_cls_head:
                # If classification head is used, compute classification logits
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Extract CLS token hidden states
                cls_token_id = tokenizer.convert_tokens_to_ids('<|CLS|>')
                cls_positions = (input_ids == cls_token_id).nonzero(as_tuple=False)  # [B, 2] (batch_idx, seq_idx)
                cls_hidden_states = outputs.hidden_states[-1][
                    cls_positions[:, 0], cls_positions[:, 1], : 
                ]  # [B, H]
                logits = self.head(cls_hidden_states)  # [B, num_classes]
                outputs.logits = logits

                return outputs

            else:
                # Generate output sequence using the base model's generate function
                generated_ids = self.base_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

                return [tokenizer.decode(generated_id, skip_special_tokens=False) for generated_id in generated_ids]


    def _boost_loss(
        self,
        outputs,
        labels,
        tokenizer,
        weight_boost: float,
        device: torch.device,
    ):
        """
        Applies loss re-weighting and MSE hybrid loss between special tokens.

        Args:
            outputs: Model outputs containing logits [B, L, V].
            labels: Ground-truth token IDs [B, L].
            tokenizer: Tokenizer to identify special tokens.
            weight_boost: Scalar weight multiplier for emphasized tokens.
        Returns:
            Weighted total loss (torch.Tensor)
        """
        # Emphasize loss between special tokens
        if labels is not None:
            special_tokens = {
                "<|activity|>": weight_boost, "<|zone|>": weight_boost, 
                "<|root_cause|>": weight_boost, "<|mean|>": weight_boost,
                "<|variance|>": weight_boost, "<|trends|>": weight_boost,
                "<|periodicity|>": weight_boost, "<|cong|>": weight_boost,
                "<|mobility|>": weight_boost, "<|anomaly_detection|>": weight_boost,
                "<|anomaly_bounds|>": weight_boost, "<|anomaly_length|>": weight_boost,
            }

            batch_size, seq_len, vocab_size = outputs.logits.size()

            # Compute per-token causal cross entropy loss (no reduction)
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for CE computation
            flat_logits = outputs.logits.view(-1, vocab_size).float()
            flat_labels = shift_labels.view(-1)

            ce_per_token = F.cross_entropy(
                flat_logits, flat_labels,
                reduction='none', ignore_index=-100
            ).view(batch_size, seq_len)

            # Create weighting mask initialized to 1 (no scaling)
            weight_mask = torch.ones_like(ce_per_token, dtype=torch.float, device=device)

            total_mse, mse_count = 0.0, 0
            loss_fn_mse = nn.MSELoss(reduction='none')
            # For each special token pair, locate spans and apply (weight - 1)
            for token, weight in special_tokens.items():
                # Get token IDs
                start_id = tokenizer.convert_tokens_to_ids(token)
                end_id = tokenizer.convert_tokens_to_ids(token.replace("<|", "</"))

                start_mask = (labels == start_id)
                end_mask = (labels == end_id)

                for b_idx in range(batch_size):
                    start_positions = torch.nonzero(start_mask[b_idx], as_tuple=False)
                    end_positions = torch.nonzero(end_mask[b_idx], as_tuple=False)
                    if start_positions.numel() == 0 or end_positions.numel() == 0:
                        continue

                    # Take first occurrence (you can adapt if multiple spans exist)
                    start_pos = start_positions[0, 0]
                    end_pos = end_positions[0, 0]
                    if end_pos <= start_pos + 1:
                        continue

                    # After right-shifting the labels, the positions shift by -1, so:
                    # start_pos-1 is the token predicting the start special token
                    # end_pos-1 is the token predicting the end special token
                    weight_mask[b_idx, start_pos : end_pos - 1] *= weight

            # Apply mask: multiply the per-token loss by its weight
            weighted_loss = (ce_per_token * weight_mask).sum() / weight_mask[shift_labels != -100].sum()
            if mse_count > 0:
                weighted_loss += total_mse / mse_count

            return weighted_loss
