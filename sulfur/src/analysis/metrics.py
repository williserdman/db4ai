from __future__ import annotations

from typing import Dict

import torch


def split_acc(
    pred: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> tuple[Dict[str, float], torch.Tensor]:
    """Return split accuracies and correctness mask."""
    corr = pred == y
    out = {
        "overall": corr.float().mean().item(),
        "train": corr[train_mask].float().mean().item(),
        "val": corr[val_mask].float().mean().item(),
        "test": corr[test_mask].float().mean().item(),
    }
    return out, corr


def eval_preds(
    pred: torch.Tensor,
    ground_truth: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    name: str,
) -> Dict[str, float | torch.Tensor | str]:
    """Evaluate predictions across splits.

    Returns a dictionary with split accuracies and the correctness mask.
    """
    corr = pred == ground_truth
    return {
        "name": name,
        "overall": corr.float().mean().item(),
        "train": corr[train_mask].float().mean().item(),
        "val": corr[val_mask].float().mean().item(),
        "test": corr[test_mask].float().mean().item(),
        "corr": corr,
    }
