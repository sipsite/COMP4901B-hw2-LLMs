"""Utility functions for student-implemented loss computations.

The training entry point expects a callable named `compute_loss_from_logits`.
Students should implement the function so that it takes model logits and
ground truth labels and returns a scalar loss tensor.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother

from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    labels: Optional[torch.Tensor],
    num_items_in_batch: int,
) -> torch.Tensor:
    """Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        ignore_index: Label id that should be ignored when computing the loss. The
            trainer passes HuggingFace's default ignore index (-100).

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.

    Students should implement this function by computing the cross-entropy loss
    from the raw logits. You may not call `torch.nn.CrossEntropyLoss`; instead,
    derive the loss explicitly using a log-softmax over the vocabulary dimension.
    """

    # raise NotImplementedError("Implement token-level cross-entropy using the logits.")
    logits = outputs.logits
    return cross_entropy_loss(logits, labels, num_items_in_batch=num_items_in_batch)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int,
) -> torch.Tensor:
    """
    Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        num_items_in_batch: Number of valid items in batch for normalization.

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.
    """
    # Causal language modeling: shift logits and labels
    # logits[:, i] predicts labels[:, i+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten the tensors
    # shift_logits: [batch_size * (seq_len-1), vocab_size]
    # shift_labels: [batch_size * (seq_len-1)]
    batch_size, seq_len_minus_1, vocab_size = shift_logits.shape
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    # Compute log softmax for numerical stability
    log_probs = F.log_softmax(flat_logits, dim=-1)

    # Create mask for valid tokens (not IGNORE_TOKEN_ID)
    mask = (flat_labels != IGNORE_TOKEN_ID).float()

    # Replace IGNORE_TOKEN_ID with 0 to avoid index errors in gather
    # These positions will be masked out anyway
    gather_labels = flat_labels.clone()
    gather_labels[flat_labels == IGNORE_TOKEN_ID] = 0

    # Gather the log probabilities of the correct tokens
    # log_probs[i, gather_labels[i]] is the log probability of the correct token at position i
    token_log_probs = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

    # Apply mask to get loss only for valid tokens
    masked_log_probs = token_log_probs * mask

    # Sum the negative log probabilities
    total_loss = -masked_log_probs.sum()

    # Normalize by num_items_in_batch to get mean loss
    if num_items_in_batch > 0:
        loss = total_loss / num_items_in_batch
    else:
        loss = total_loss

    return loss
