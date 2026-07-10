from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import SplitIndices, TensorBundle, make_tensor_loader
from .losses import SNIELossWeights, evidence_token_loss, snie_loss
from .metrics import MetricReport, evaluate_predictions


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    warmup_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    num_workers: int
    device: str
    monitor: str
    rank_queue_size: int
    rank_total_band: float
    graph_edge_dropout: float
    evidence_fraction: float
    evidence_margin: float
    evidence_max_items: int
    random_state: int | None = None
    restore_best: bool = True
    save_arrays: bool = False
    save_provenance: bool = False


@dataclass
class TrainingResult:
    summary: dict[str, float]
    per_item: dict[str, dict[str, float]]
    per_scale: dict[str, dict[str, float]]
    history: list[dict[str, float]]
    arrays: dict[str, np.ndarray]


def train_snie_model(
    model: torch.nn.Module,
    bundle: TensorBundle,
    splits: SplitIndices,
    config: TrainingConfig,
    loss_weights: SNIELossWeights,
    artifact_dir: str | Path | None = None,
) -> TrainingResult:
    _validate_training_config(config)
    set_random_state(config.random_state)
    device = resolve_device(config.device)
    model = model.to(device)

    train_loader = make_tensor_loader(
        bundle,
        splits.train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        random_state=config.random_state,
        pin_memory=device.type == "cuda",
        drop_last=len(splits.train) >= config.batch_size,
    )
    validation_loader = make_tensor_loader(
        bundle,
        splits.validation,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        random_state=config.random_state,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = make_tensor_loader(
        bundle,
        splits.test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        random_state=config.random_state,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rank_queue = SameTotalRankQueue(config.rank_queue_size) if config.rank_queue_size > 0 else None
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(config.max_epochs):
        phase = "warmup" if epoch < config.warmup_epochs else "full"
        train_log = train_one_epoch(model, train_loader, optimizer, bundle, config, loss_weights, device, phase, rank_queue)
        validation_arrays = collect_snie_outputs(model, validation_loader, bundle, device, save_provenance=False)
        validation_report = make_report(validation_arrays, bundle)
        monitor_value = float(validation_report.summary.get(config.monitor, float("nan")))
        row = {"epoch": float(epoch + 1), **train_log, f"validation_{config.monitor}": monitor_value}
        history.append(row)
        score = monitor_value if np.isfinite(monitor_value) else -float("inf")
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if config.restore_best and best_state is not None:
        model.load_state_dict(best_state)

    arrays = collect_snie_outputs(model, test_loader, bundle, device, save_provenance=config.save_provenance)
    report = make_report(arrays, bundle)
    result = TrainingResult(
        summary=report.summary,
        per_item=report.per_item,
        per_scale=report.per_scale,
        history=history,
        arrays=arrays,
    )
    if artifact_dir is not None:
        write_training_result(result, artifact_dir, config, loss_weights)
    return result


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    bundle: TensorBundle,
    config: TrainingConfig,
    loss_weights: SNIELossWeights,
    device: torch.device,
    phase: str,
    rank_queue: "SameTotalRankQueue | None",
) -> dict[str, float]:
    model.train()
    logs: dict[str, list[float]] = {}
    item_to_scale = bundle.item_to_scale.to(device)
    item_bounds = bundle.item_bounds.to(device)

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        graph_override = sample_training_graph(model, bundle, config.graph_edge_dropout, device) if phase == "full" else None
        out = model(batch["x"], graph_override=graph_override)
        loss, log = snie_loss(
            out,
            batch,
            item_to_scale,
            item_bounds,
            weights=loss_weights,
            phase=phase,
            rank_reference=rank_queue.reference(device) if rank_queue is not None else None,
            rank_total_band=config.rank_total_band,
        )
        if phase == "full" and loss_weights.evidence > 0:
            evi = evidence_token_loss(
                model,
                batch["x"],
                out,
                graph_override=graph_override,
                fraction=config.evidence_fraction,
                margin=config.evidence_margin,
                max_items=config.evidence_max_items,
            )
            loss = loss + loss_weights.evidence * evi
            log["evidence"] = float(evi.detach().cpu())
            log["loss"] = float(loss.detach().cpu())
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if rank_queue is not None:
            rank_queue.enqueue(out["delta"].detach().cpu(), batch["y"].detach().cpu(), batch["scale_totals"].detach().cpu())
        for key, value in log.items():
            logs.setdefault(key, []).append(float(value))
    return {f"train_{key}": float(np.mean(values)) for key, values in logs.items()}


def collect_snie_outputs(
    model: torch.nn.Module,
    loader,
    bundle: TensorBundle,
    device: torch.device,
    save_provenance: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "y": [],
        "residual": [],
        "scale_totals": [],
        "score_pred": [],
        "item_pred": [],
        "total_pred": [],
        "rho": [],
        "gamma": [],
    }
    provenance: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch["x"])
            delta = out["delta"].cpu().numpy()
            total_pred = out["total_pred"].cpu().numpy()
            if bundle.conditional_mean is not None:
                item_pred = bundle.conditional_mean.predict(total_pred) + delta
            else:
                item_pred = delta
            chunks["y"].append(batch["y"].cpu().numpy())
            chunks["residual"].append(batch["residual"].cpu().numpy())
            chunks["scale_totals"].append(batch["scale_totals"].cpu().numpy())
            chunks["score_pred"].append(delta)
            chunks["item_pred"].append(item_pred)
            chunks["total_pred"].append(total_pred)
            chunks["rho"].append(out["rho"].cpu().numpy())
            chunks["gamma"].append(out["gamma"].cpu().numpy())
            if save_provenance:
                provenance.append(out["provenance"].cpu().numpy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items() if values}
    if provenance:
        arrays["provenance"] = np.concatenate(provenance, axis=0)
    return arrays


def make_report(arrays: dict[str, np.ndarray], bundle: TensorBundle) -> MetricReport:
    return evaluate_predictions(
        y_true=arrays["y"],
        scale_totals=arrays["scale_totals"],
        item_to_scale=bundle.item_to_scale.cpu().numpy(),
        score_pred=arrays["score_pred"],
        item_pred=arrays["item_pred"],
        total_pred=arrays["total_pred"],
    )


class SameTotalRankQueue:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.pred: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.scale_totals: torch.Tensor | None = None

    def reference(self, device: torch.device) -> dict[str, torch.Tensor] | None:
        if self.pred is None or self.y is None or self.scale_totals is None:
            return None
        return {
            "pred": self.pred.to(device, non_blocking=True),
            "y": self.y.to(device, non_blocking=True),
            "scale_totals": self.scale_totals.to(device, non_blocking=True),
        }

    def enqueue(self, pred: torch.Tensor, y: torch.Tensor, scale_totals: torch.Tensor) -> None:
        if self.max_size <= 0:
            return
        if self.pred is None:
            self.pred = pred[-self.max_size :].clone()
            self.y = y[-self.max_size :].clone()
            self.scale_totals = scale_totals[-self.max_size :].clone()
            return
        self.pred = torch.cat([self.pred, pred], dim=0)[-self.max_size :].clone()
        self.y = torch.cat([self.y, y], dim=0)[-self.max_size :].clone()
        self.scale_totals = torch.cat([self.scale_totals, scale_totals], dim=0)[-self.max_size :].clone()


def sample_training_graph(
    model: torch.nn.Module,
    bundle: TensorBundle,
    edge_dropout: float,
    device: torch.device,
) -> torch.Tensor | None:
    if edge_dropout <= 0 or not hasattr(model, "graph"):
        return None
    graph = model.graph
    edge_mask = torch.triu(graph.abs() > 0, diagonal=1)
    if not edge_mask.any():
        return graph
    if bundle.graph_stability is not None:
        stability = bundle.graph_stability.to(device=device, dtype=graph.dtype)
        drop_probability = (edge_dropout * (1.0 - stability)).clamp(0.0, 1.0)
    else:
        drop_probability = torch.full_like(graph, edge_dropout).clamp(0.0, 1.0)
    sampled = (torch.rand_like(graph) > drop_probability).to(graph.dtype)
    keep = torch.ones_like(graph)
    keep[edge_mask] = sampled[edge_mask]
    keep = torch.triu(keep, diagonal=1)
    keep = keep + keep.T
    return graph * keep


def write_training_result(
    result: TrainingResult,
    artifact_dir: str | Path,
    config: TrainingConfig,
    loss_weights: SNIELossWeights,
) -> None:
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": _json_ready(result.summary),
                "per_item": _json_ready(result.per_item),
                "per_scale": _json_ready(result.per_scale),
                "config": asdict(config),
                "loss_weights": asdict(loss_weights),
            },
            f,
            indent=2,
        )
    with (path / "history.json").open("w", encoding="utf-8") as f:
        json.dump(_json_ready(result.history), f, indent=2)
    _write_table(path / "per_item.csv", result.per_item)
    _write_table(path / "per_scale.csv", result.per_scale)
    if config.save_arrays:
        arrays = {
            key: value
            for key, value in result.arrays.items()
            if config.save_provenance or key != "provenance"
        }
        np.savez_compressed(path / "arrays.npz", **arrays)


def _write_table(path: Path, rows: dict[str, dict[str, float]]) -> None:
    keys = sorted({metric for row in rows.values() for metric in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", *keys])
        writer.writeheader()
        for name, row in rows.items():
            writer.writerow({"name": name, **row})


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def set_random_state(random_state: int | None) -> None:
    if random_state is None:
        return
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _validate_training_config(config: TrainingConfig) -> None:
    if config.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if config.warmup_epochs < 0 or config.warmup_epochs > config.max_epochs:
        raise ValueError("warmup_epochs must be in [0, max_epochs]")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.rank_queue_size < 0:
        raise ValueError("rank_queue_size must be non-negative")
    if config.rank_total_band < 0:
        raise ValueError("rank_total_band must be non-negative")
    if not 0 <= config.graph_edge_dropout <= 1:
        raise ValueError("graph_edge_dropout must be in [0, 1]")
    if not 0 < config.evidence_fraction <= 1:
        raise ValueError("evidence_fraction must be in (0, 1]")
    if config.evidence_margin < 0:
        raise ValueError("evidence_margin must be non-negative")
