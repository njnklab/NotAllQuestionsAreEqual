from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .graph import validate_graph


class ConditionalMeanPredictor(Protocol):
    """Optional residual-to-score adapter used at evaluation time."""

    def predict(self, scale_totals: np.ndarray) -> np.ndarray:
        ...


@dataclass
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass
class TensorBundle:
    """Data already prepared by the caller.

    This public release deliberately does not include file readers, label
    mappings, subject identifiers, or split definitions tied to a private data
    source. Callers pass tensors that are already anonymized and aligned.
    """

    x: torch.Tensor
    y: torch.Tensor
    scale_totals: torch.Tensor
    residual: torch.Tensor
    item_to_scale: torch.Tensor
    item_bounds: torch.Tensor
    graph: torch.Tensor
    conditional_mean: ConditionalMeanPredictor | None = None
    graph_stability: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.x = torch.as_tensor(self.x).float()
        self.y = torch.as_tensor(self.y).float()
        self.scale_totals = torch.as_tensor(self.scale_totals).float()
        self.residual = torch.as_tensor(self.residual).float()
        self.item_to_scale = torch.as_tensor(self.item_to_scale).long()
        self.item_bounds = torch.as_tensor(self.item_bounds).float()
        self.graph = validate_graph(torch.as_tensor(self.graph).float(), self.num_items)
        if self.graph_stability is not None:
            self.graph_stability = validate_graph(torch.as_tensor(self.graph_stability).float(), self.num_items)
        self._validate_shapes()

    @property
    def num_examples(self) -> int:
        return int(self.y.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.y.shape[1])

    @property
    def num_scales(self) -> int:
        return int(self.scale_totals.shape[1])

    def _validate_shapes(self) -> None:
        if self.x.shape[0] != self.num_examples:
            raise ValueError("x and y must have the same leading dimension")
        expected_items = (self.num_examples, self.num_items)
        if tuple(self.residual.shape) != expected_items:
            raise ValueError(f"residual shape must be {expected_items}")
        if self.scale_totals.ndim != 2 or self.scale_totals.shape[0] != self.num_examples:
            raise ValueError("scale_totals must have shape (examples, scales)")
        if tuple(self.item_to_scale.shape) != (self.num_items,):
            raise ValueError("item_to_scale must have shape (items,)")
        if tuple(self.item_bounds.shape) != (self.num_items,):
            raise ValueError("item_bounds must have shape (items,)")
        if not torch.all(self.item_bounds > 0):
            raise ValueError("item_bounds must be positive")
        if int(self.item_to_scale.max().item()) >= self.num_scales or int(self.item_to_scale.min().item()) < 0:
            raise ValueError("item_to_scale contains an invalid scale index")


class TensorItemDataset(Dataset):
    def __init__(self, bundle: TensorBundle, indices: np.ndarray) -> None:
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw_idx = int(self.indices[idx])
        return {
            "x": self.bundle.x[raw_idx],
            "y": self.bundle.y[raw_idx],
            "residual": self.bundle.residual[raw_idx],
            "scale_totals": self.bundle.scale_totals[raw_idx],
            "index": torch.tensor(raw_idx, dtype=torch.long),
        }


def make_tensor_loader(
    bundle: TensorBundle,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    random_state: int | None,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = None
    if random_state is not None:
        generator = torch.Generator()
        generator.manual_seed(int(random_state))
    return DataLoader(
        TensorItemDataset(bundle, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )


def make_random_split(
    num_examples: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_state: int | None,
) -> SplitIndices:
    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=np.float64)
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if np.any(fractions < 0) or not np.isclose(float(fractions.sum()), 1.0):
        raise ValueError("split fractions must be non-negative and sum to 1")
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(num_examples)
    train_end = int(round(train_fraction * num_examples))
    validation_end = train_end + int(round(validation_fraction * num_examples))
    return SplitIndices(
        train=np.sort(indices[:train_end]),
        validation=np.sort(indices[train_end:validation_end]),
        test=np.sort(indices[validation_end:]),
    )
