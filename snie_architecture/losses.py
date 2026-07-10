from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SNIELossWeights:
    residual: float
    direct: float
    rank: float
    centering: float
    reliability: float
    scale: float
    evidence: float
    direct_unweighted: bool = False


def smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def snie_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    item_to_scale: torch.Tensor,
    item_bounds: torch.Tensor,
    weights: SNIELossWeights,
    phase: str,
    rank_reference: dict[str, torch.Tensor] | None,
    rank_total_band: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    residual = batch["residual"]
    y = batch["y"]
    scale_totals = batch["scale_totals"]

    loss_residual = smooth_l1(out["delta"], residual)
    direct_weight = None if weights.direct_unweighted else out["rho"].detach()
    loss_direct = smooth_l1(out["delta_direct"], residual, direct_weight)
    if phase == "full":
        loss_rank = same_total_rank_loss(
            out["delta"],
            y,
            scale_totals,
            item_to_scale,
            rank_total_band,
            rank_reference,
        )
    else:
        loss_rank = out["delta"].sum() * 0.0
    loss_centering = conditional_centering_loss(out["delta"], scale_totals, item_to_scale)
    target_reliability = torch.exp(
        -torch.abs(out["delta_direct"].detach() - residual) / item_bounds.unsqueeze(0).clamp_min(1e-6)
    )
    loss_reliability = F.mse_loss(out["rho"], target_reliability)
    loss_scale = smooth_l1(out["total_pred"], scale_totals)

    active_residual = loss_residual if phase == "full" else loss_residual * 0.0
    total = (
        weights.residual * active_residual
        + weights.direct * loss_direct
        + weights.rank * loss_rank
        + weights.centering * loss_centering
        + weights.reliability * loss_reliability
        + weights.scale * loss_scale
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "residual": float(loss_residual.detach().cpu()),
        "direct": float(loss_direct.detach().cpu()),
        "rank": float(loss_rank.detach().cpu()),
        "centering": float(loss_centering.detach().cpu()),
        "reliability": float(loss_reliability.detach().cpu()),
        "scale": float(loss_scale.detach().cpu()),
    }
    return total, logs


def evidence_token_loss(
    model,
    x: torch.Tensor,
    out: dict[str, torch.Tensor],
    graph_override: torch.Tensor | None,
    fraction: float,
    margin: float,
    max_items: int,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("evidence_token_loss expects token-like inputs with shape (batch, time, feature_dim)")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if max_items <= 0:
        return out["delta"].sum() * 0.0

    provenance = out["provenance"].detach()
    base_delta = out["delta"].detach()
    item_count = provenance.shape[1]
    chosen = torch.randperm(item_count, device=x.device)[: min(max_items, item_count)]
    losses = []
    for item_idx in chosen.tolist():
        scores = provenance[:, item_idx, :]
        kept = token_keep_drop(x, scores, fraction, mode="keep")
        dropped = token_keep_drop(x, scores, fraction, mode="drop")
        keep_delta = model(kept, graph_override=graph_override)["delta"][:, item_idx]
        drop_delta = model(dropped, graph_override=graph_override)["delta"][:, item_idx]
        target = base_delta[:, item_idx]
        losses.append((keep_delta - target).abs().mean() + F.relu(drop_delta.abs() - margin).mean())
    return torch.stack(losses).mean() if losses else out["delta"].sum() * 0.0


def token_keep_drop(x: torch.Tensor, token_scores: torch.Tensor, fraction: float, mode: str) -> torch.Tensor:
    batch, time, _ = x.shape
    score_time = token_scores.shape[-1]
    keep_count = max(1, int(round(fraction * score_time)))
    top = torch.topk(token_scores, k=min(keep_count, score_time), dim=-1).indices
    score_mask = torch.zeros(batch, score_time, device=x.device, dtype=torch.bool)
    score_mask.scatter_(-1, top, True)
    token_mask = torch.zeros(batch, time, device=x.device, dtype=torch.bool)
    for score_idx in range(score_time):
        start = int(round(score_idx * time / score_time))
        end = int(round((score_idx + 1) * time / score_time))
        token_mask[:, start:end] = score_mask[:, score_idx].unsqueeze(-1)
    token_mask = token_mask.unsqueeze(-1)
    std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    replacement = torch.randn_like(x) * std
    if mode == "keep":
        return torch.where(token_mask, x, replacement)
    if mode == "drop":
        return torch.where(token_mask, replacement, x)
    raise ValueError(f"Unknown mode: {mode}")


def same_total_rank_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    scale_totals: torch.Tensor,
    item_to_scale: torch.Tensor,
    total_band: float,
    reference: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    losses = []
    if reference:
        ref_pred = reference["pred"].to(pred.device).detach()
        ref_y = reference["y"].to(pred.device).detach()
        ref_scale_totals = reference["scale_totals"].to(pred.device).detach()
        all_pred = torch.cat([pred, ref_pred], dim=0)
        all_y = torch.cat([y, ref_y], dim=0)
        all_scale_totals = torch.cat([scale_totals, ref_scale_totals], dim=0)
    else:
        all_pred = pred
        all_y = y
        all_scale_totals = scale_totals

    anchor_count = pred.shape[0]
    for item_idx in range(pred.shape[1]):
        scale_idx = int(item_to_scale[item_idx].item())
        anchor_totals = scale_totals[:, scale_idx]
        compare_totals = all_scale_totals[:, scale_idx]
        for row_idx in range(anchor_count):
            total_mask = (compare_totals - anchor_totals[row_idx]).abs() <= total_band
            total_mask[row_idx] = False
            if not total_mask.any():
                continue
            target_diff = y[row_idx, item_idx] - all_y[:, item_idx]
            mask = total_mask & (target_diff != 0)
            if not mask.any():
                continue
            sign = torch.sign(target_diff[mask])
            pred_diff = pred[row_idx, item_idx] - all_pred[:, item_idx]
            losses.append(F.softplus(-sign * pred_diff[mask]).mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def conditional_centering_loss(
    pred: torch.Tensor,
    scale_totals: torch.Tensor,
    item_to_scale: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for item_idx in range(pred.shape[1]):
        scale_idx = int(item_to_scale[item_idx].item())
        totals = scale_totals[:, scale_idx]
        for total in torch.unique(totals):
            rows = torch.nonzero(totals == total, as_tuple=False).flatten()
            if rows.numel() >= 2:
                losses.append(pred[rows, item_idx].mean().pow(2))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0
