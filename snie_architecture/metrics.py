from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, rankdata


@dataclass
class MetricReport:
    summary: dict[str, float]
    per_item: dict[str, dict[str, float]]
    per_scale: dict[str, dict[str, float]]


def evaluate_predictions(
    y_true: np.ndarray,
    scale_totals: np.ndarray,
    item_to_scale: np.ndarray,
    score_pred: np.ndarray,
    item_pred: np.ndarray | None,
    total_pred: np.ndarray | None,
) -> MetricReport:
    item_to_scale = np.asarray(item_to_scale, dtype=np.int64)
    per_item: dict[str, dict[str, float]] = {}
    aucs: list[float] = []
    spears: list[float] = []
    kends: list[float] = []
    accs: list[float] = []

    for item_idx in range(y_true.shape[1]):
        scale_idx = int(item_to_scale[item_idx])
        auc = same_total_pair_auc(y_true[:, item_idx], score_pred[:, item_idx], scale_totals[:, scale_idx])
        spear, kend = same_total_rank_corr(y_true[:, item_idx], score_pred[:, item_idx], scale_totals[:, scale_idx])
        row = {"same_total_auc": auc, "spearman": spear, "kendall": kend}
        if item_pred is not None:
            rounded = np.rint(item_pred[:, item_idx])
            lo = np.nanmin(y_true[:, item_idx])
            hi = np.nanmax(y_true[:, item_idx])
            row["rounded_accuracy"] = float(np.mean(np.clip(rounded, lo, hi) == y_true[:, item_idx]))
            accs.append(row["rounded_accuracy"])
        per_item[f"item_{item_idx}"] = row
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(spear):
            spears.append(spear)
        if not np.isnan(kend):
            kends.append(kend)

    per_scale: dict[str, dict[str, float]] = {}
    for scale_idx in range(scale_totals.shape[1]):
        item_indices = np.where(item_to_scale == scale_idx)[0]
        per_scale[f"scale_{scale_idx}"] = {
            "same_total_auc": _nanmean(per_item[f"item_{idx}"]["same_total_auc"] for idx in item_indices),
            "rounded_accuracy": _nanmean(per_item[f"item_{idx}"].get("rounded_accuracy", np.nan) for idx in item_indices),
        }

    summary = {
        "macro_same_total_auc": _nanmean(aucs),
        "same_total_spearman": _nanmean(spears),
        "same_total_kendall": _nanmean(kends),
        "rounded_item_accuracy": _nanmean(accs),
    }
    if total_pred is not None:
        summary["total_mae"] = float(np.mean(np.abs(scale_totals.reshape(-1) - total_pred.reshape(-1))))
    elif item_pred is not None:
        pred_totals = np.zeros_like(scale_totals)
        for scale_idx in range(pred_totals.shape[1]):
            item_indices = np.where(item_to_scale == scale_idx)[0]
            pred_totals[:, scale_idx] = item_pred[:, item_indices].sum(axis=1)
        summary["total_mae"] = float(np.mean(np.abs(scale_totals.reshape(-1) - pred_totals.reshape(-1))))
    else:
        summary["total_mae"] = float("nan")
    return MetricReport(summary=summary, per_item=per_item, per_scale=per_scale)


def same_total_pair_auc(y: np.ndarray, score: np.ndarray, totals: np.ndarray) -> float:
    values: list[float] = []
    weights: list[int] = []
    for total in np.unique(totals):
        idx = np.where(totals == total)[0]
        if idx.size < 2:
            continue
        yy = y[idx]
        ss = score[idx]
        levels = np.unique(yy)
        if levels.size < 2:
            continue
        for low_pos, low in enumerate(levels[:-1]):
            for high in levels[low_pos + 1 :]:
                mask = (yy == low) | (yy == high)
                labels = (yy[mask] == high).astype(np.int64)
                auc = _binary_auc(labels, ss[mask])
                if not np.isnan(auc):
                    n_low = int(np.sum(yy == low))
                    n_high = int(np.sum(yy == high))
                    values.append(auc)
                    weights.append(n_low * n_high)
    return float(np.average(values, weights=weights)) if values else float("nan")


def same_total_rank_corr(y: np.ndarray, score: np.ndarray, totals: np.ndarray) -> tuple[float, float]:
    y_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for total in np.unique(totals):
        idx = np.where(totals == total)[0]
        if idx.size < 3 or np.unique(y[idx]).size < 2:
            continue
        y_parts.append(y[idx] - y[idx].mean())
        score_parts.append(score[idx] - score[idx].mean())
    if not y_parts:
        return float("nan"), float("nan")
    yy = np.concatenate(y_parts)
    ss = np.concatenate(score_parts)
    if np.std(yy) < 1e-8 or np.std(ss) < 1e-8:
        return float("nan"), float("nan")
    spearman = _pearson(rankdata(yy), rankdata(ss))
    kendall = kendalltau(yy, ss).statistic
    return float(spearman), float(kendall)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    pos_rank_sum = float(ranks[positives].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - a.mean()
    bb = b - b.mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def _nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")
