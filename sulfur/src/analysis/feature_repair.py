from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch_geometric.data import Data
import torch.nn.functional as F

from .metrics import split_acc


@dataclass
class FeatureRepairConfig:
    max_epochs: int = 500
    lr: float = 0.1
    weight_decay: float = 0.0
    log_every: int = 25
    max_target_nodes: Optional[int] = None
    l2_to_original: float = 0.0
    use_entropy_weighting: bool = True


@dataclass
class FeatureRepairResult:
    history: Dict[str, list]
    best_state: Dict[str, torch.Tensor] | None
    metrics_base: Dict[str, float]
    correct_base: torch.Tensor
    target_mask: torch.Tensor


def _select_target_mask(
    incorrect_mask: torch.Tensor,
    entropy: torch.Tensor,
    max_target_nodes: Optional[int],
) -> torch.Tensor:
    target_mask = incorrect_mask.clone()
    target_indices = torch.where(target_mask)[0]
    if max_target_nodes is not None and target_indices.numel() > max_target_nodes:
        order = torch.argsort(entropy[target_indices], descending=True)
        target_indices = target_indices[order[:max_target_nodes]]
        target_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        target_mask[target_indices] = True
    if target_indices.numel() == 0:
        raise RuntimeError("No target nodes selected for repair.")
    return target_mask


def repair_features_overfit(
    model: torch.nn.Module,
    data: Data,
    config: FeatureRepairConfig,
) -> FeatureRepairResult:
    """Optimize a per-node feature delta for initially incorrect nodes.

    Returns history and best checkpoint (by validation accuracy).
    """
    device = next(model.parameters()).device
    model.eval()

    graph_data = data.to(device)
    x_original = graph_data.x.detach().clone()
    y = graph_data.y
    train_mask = graph_data.train_mask
    val_mask = graph_data.val_mask
    test_mask = graph_data.test_mask

    with torch.no_grad():
        logits_base, _ = model(graph_data)
        probs_base = torch.softmax(logits_base, dim=1)
        pred_base = logits_base.argmax(dim=1)

    metrics_base, correct_base = split_acc(pred_base, y, train_mask, val_mask, test_mask)
    incorrect_mask = ~correct_base

    eps = 1e-12
    entropy = -(probs_base.clamp_min(eps) * probs_base.clamp_min(eps).log()).sum(dim=1)

    target_mask = _select_target_mask(incorrect_mask, entropy, config.max_target_nodes)
    target_mask_f = target_mask.float().unsqueeze(1)

    delta_param = torch.nn.Parameter(torch.zeros_like(x_original))
    optimizer = torch.optim.Adam([delta_param], lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    history = {
        "epoch": [],
        "loss_total": [],
        "loss_target_ce": [],
        "loss_l2": [],
        "overall": [],
        "train": [],
        "val": [],
        "test": [],
        "fixed": [],
        "regressed": [],
        "net": [],
        "target_grad_mean": [],
        "target_grad_max": [],
        "zero_grad_target_rows": [],
        "target_delta_l2_mean": [],
        "target_delta_l2_max": [],
    }

    best_val = -1.0
    best_state = None

    for epoch in range(1, config.max_epochs + 1):
        optimizer.zero_grad()

        delta_masked = delta_param * target_mask_f
        x_repaired = x_original + delta_masked

        data_tmp = graph_data.clone()
        data_tmp.x = x_repaired

        logits, _ = model(data_tmp)
        logits_target = logits[target_mask]
        y_target = y[target_mask]

        if config.use_entropy_weighting:
            weights = entropy[target_mask].detach()
            weights = weights / (weights.mean() + 1e-12)
            per_node_ce = F.cross_entropy(logits_target, y_target, reduction="none")
            loss_target_ce = (weights * per_node_ce).mean()
        else:
            loss_target_ce = F.cross_entropy(logits_target, y_target)

        if config.l2_to_original > 0.0:
            loss_l2 = (delta_masked[target_mask]).pow(2).mean()
        else:
            loss_l2 = torch.tensor(0.0, device=device)

        loss_total = loss_target_ce + (config.l2_to_original * loss_l2)
        loss_total.backward()

        with torch.no_grad():
            grad = delta_param.grad
            target_grad = grad[target_mask]
            target_grad_norm = target_grad.norm(dim=1)
            zero_grad_rows = (target_grad_norm < 1e-12).sum().item()

        optimizer.step()
        scheduler.step()

        if epoch % config.log_every == 0 or epoch == 1 or epoch == config.max_epochs:
            with torch.no_grad():
                pred_now = logits.argmax(dim=1)
                m_now, corr_now = split_acc(pred_now, y, train_mask, val_mask, test_mask)

            fixed = ((~correct_base) & corr_now).sum().item()
            regressed = (correct_base & (~corr_now)).sum().item()
            net = fixed - regressed

            delta_now = (delta_param * target_mask_f)[target_mask]
            delta_l2 = delta_now.norm(dim=1)

            history["epoch"].append(epoch)
            history["loss_total"].append(loss_total.item())
            history["loss_target_ce"].append(loss_target_ce.item())
            history["loss_l2"].append(loss_l2.item())
            history["overall"].append(m_now["overall"])
            history["train"].append(m_now["train"])
            history["val"].append(m_now["val"])
            history["test"].append(m_now["test"])
            history["fixed"].append(fixed)
            history["regressed"].append(regressed)
            history["net"].append(net)
            history["target_grad_mean"].append(target_grad_norm.mean().item())
            history["target_grad_max"].append(target_grad_norm.max().item())
            history["zero_grad_target_rows"].append(int(zero_grad_rows))
            history["target_delta_l2_mean"].append(delta_l2.mean().item())
            history["target_delta_l2_max"].append(delta_l2.max().item())

            if m_now["val"] > best_val:
                best_val = m_now["val"]
                best_state = {
                    "epoch": epoch,
                    "delta": delta_param.detach().clone(),
                    "pred": pred_now.detach().clone(),
                    "metrics": m_now,
                }

    return FeatureRepairResult(
        history=history,
        best_state=best_state,
        metrics_base=metrics_base,
        correct_base=correct_base,
        target_mask=target_mask,
    )


def repair_features_all_nodes(
    model: torch.nn.Module,
    data: Data,
    config: FeatureRepairConfig,
    target_mask: torch.Tensor,
) -> FeatureRepairResult:
    """Optimize a per-node feature delta using all node labels."""
    device = next(model.parameters()).device
    model.eval()

    graph_data = data.to(device)
    x_original = graph_data.x.detach().clone()
    y = graph_data.y
    train_mask = graph_data.train_mask
    val_mask = graph_data.val_mask
    test_mask = graph_data.test_mask

    with torch.no_grad():
        logits_base, _ = model(graph_data)
        pred_base = logits_base.argmax(dim=1)

    metrics_base, correct_base = split_acc(pred_base, y, train_mask, val_mask, test_mask)

    target_mask_f = target_mask.float().unsqueeze(1)

    delta_param = torch.nn.Parameter(torch.zeros_like(x_original))
    optimizer = torch.optim.Adam([delta_param], lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    history = {
        "epoch": [],
        "loss_total": [],
        "loss_all_ce": [],
        "loss_l2": [],
        "overall": [],
        "fixed": [],
        "regressed": [],
        "net": [],
    }

    best_overall = -1.0
    best_state = None

    for epoch in range(1, config.max_epochs + 1):
        optimizer.zero_grad()

        delta_masked = delta_param * target_mask_f
        x_repaired = x_original + delta_masked

        data_tmp = graph_data.clone()
        data_tmp.x = x_repaired

        logits, _ = model(data_tmp)
        pred = logits.argmax(dim=1)

        loss_all_ce = F.cross_entropy(logits, y)
        if config.l2_to_original > 0.0:
            loss_l2 = (delta_masked[target_mask]).pow(2).mean()
        else:
            loss_l2 = torch.tensor(0.0, device=device)

        loss_total = loss_all_ce + (config.l2_to_original * loss_l2)
        loss_total.backward()

        optimizer.step()
        scheduler.step()

        if epoch % config.log_every == 0 or epoch == 1 or epoch == config.max_epochs:
            with torch.no_grad():
                metrics_now, corr_now = split_acc(pred, y, train_mask, val_mask, test_mask)

            fixed = ((~correct_base) & corr_now).sum().item()
            regressed = (correct_base & (~corr_now)).sum().item()
            net = fixed - regressed

            history["epoch"].append(epoch)
            history["loss_total"].append(loss_total.item())
            history["loss_all_ce"].append(loss_all_ce.item())
            history["loss_l2"].append(loss_l2.item())
            history["overall"].append(metrics_now["overall"])
            history["fixed"].append(fixed)
            history["regressed"].append(regressed)
            history["net"].append(net)

            if metrics_now["overall"] > best_overall:
                best_overall = metrics_now["overall"]
                best_state = {
                    "epoch": epoch,
                    "delta": delta_param.detach().clone(),
                    "pred": pred.detach().clone(),
                    "metrics": metrics_now,
                }

    return FeatureRepairResult(
        history=history,
        best_state=best_state,
        metrics_base=metrics_base,
        correct_base=correct_base,
        target_mask=target_mask,
    )


def summarize_drift(
    x_original: torch.Tensor,
    delta: torch.Tensor,
    correct_base: torch.Tensor,
    correct_best: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Summarize drift by outcome group."""
    fixed_mask = (~correct_base) & correct_best
    regressed_mask = correct_base & (~correct_best)
    still_wrong_mask = (~correct_base) & (~correct_best)
    always_correct_mask = correct_base & correct_best

    drift = delta
    drift_l2 = drift.norm(dim=1)
    cos_sim = F.cosine_similarity(x_original + delta, x_original, dim=1, eps=1e-12)
    drift_cosine = 1.0 - cos_sim

    def summarize_group(mask: torch.Tensor, name: str) -> Dict[str, float]:
        if mask.sum() == 0:
            return {
                "group": name,
                "count": 0.0,
                "l2_mean": float("nan"),
                "l2_median": float("nan"),
                "cos_mean": float("nan"),
                "cos_median": float("nan"),
            }
        return {
            "group": name,
            "count": float(mask.sum().item()),
            "l2_mean": float(drift_l2[mask].mean().item()),
            "l2_median": float(drift_l2[mask].median().item()),
            "cos_mean": float(drift_cosine[mask].mean().item()),
            "cos_median": float(drift_cosine[mask].median().item()),
        }

    non_target_mask = ~target_mask
    non_target_l2_max = drift_l2[non_target_mask].max().item() if non_target_mask.any() else 0.0

    return {
        "fixed": summarize_group(fixed_mask, "fixed"),
        "still_wrong": summarize_group(still_wrong_mask, "still_wrong"),
        "regressed": summarize_group(regressed_mask, "regressed"),
        "always_correct": summarize_group(always_correct_mask, "always_correct"),
        "non_target_l2_max": {"value": float(non_target_l2_max)},
    }
