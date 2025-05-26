"""Module for calculating metrics

Classes:
    MetricsAccumulator: Accumulates and computes evaluation metrics for model predictions
"""
from typing import Optional, Tuple, Dict
import math
import torch
from torch.nn import functional as F

class MetricsAccumulator:
    """
    Accumulates and computes evaluation metrics. This class saves VRAM usage since the logits do
    not need to be stored in memory. 
    """
    def __init__(self):
        """
        Initializes the MetricsAccumulator with loss and token count to 0"""
        self.total_loss = 0.0
        self.total_tokens = 0

    def __call__(self, eval_preds: Optional[Tuple] = None, compute_result: bool = False) -> dict:
        """
        Computes the evaluation metrics

        Args:
            eval_preds (Optional[Tuple]): Tuple of logits and labels from the model
            compute_result (bool): Flag to indicate if final computation should be done

        Returns:
            dict: Dictionary containing evaluation loss and perplexity
        """
        if compute_result:
            if self.total_tokens == 0:
                return {"eval_loss": 0.0, "eval_perplexity": float("inf")}
            mean_loss = self.total_loss / self.total_tokens
            perplexity = float(math.exp(mean_loss))
            return {
                "eval_loss": mean_loss,
                "eval_perplexity": perplexity,
            }

        logits, labels = eval_preds

        if isinstance(logits, tuple):
            logits = logits[0]

        with torch.no_grad():
            logits = logits.detach().cpu()
            labels = labels.detach().cpu()

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            valid = shift_labels != -100
            shift_logits = shift_logits[valid]
            shift_labels = shift_labels[valid]

            loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
            num_tokens = shift_labels.numel()

            self.total_loss += loss.item()
            self.total_tokens += num_tokens

        return {}