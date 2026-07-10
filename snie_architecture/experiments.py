from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .data import SplitIndices, TensorBundle
from .losses import SNIELossWeights
from .model import SNIEArchitectureConfig, SNIEModel
from .trainer import TrainingConfig, TrainingResult, set_random_state, train_snie_model


EncoderFactory = Callable[[int | None], torch.nn.Module]
ModelFactory = Callable[[int | None], torch.nn.Module]


@dataclass(frozen=True)
class ArchitectureVariant:
    """Named architecture edit for public ablation experiments."""

    name: str
    updates: Mapping[str, object]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    build_model: ModelFactory
    training: TrainingConfig
    loss_weights: SNIELossWeights


@dataclass
class ExperimentResult:
    rows: list[dict[str, float | str]]
    runs: dict[str, TrainingResult]


def make_snie_model_factory(
    base_config: SNIEArchitectureConfig,
    encoder_factory: EncoderFactory,
    encoder_dim: int,
    bundle: TensorBundle,
    variant: ArchitectureVariant | None,
) -> ModelFactory:
    """Create a model factory for one architecture variant."""

    updates = dict(variant.updates) if variant is not None else {}

    def build(random_state: int | None) -> torch.nn.Module:
        set_random_state(random_state)
        config = replace(base_config, **updates)
        encoder = encoder_factory(random_state)
        return SNIEModel(
            config=config,
            encoder=encoder,
            encoder_dim=encoder_dim,
            graph=bundle.graph,
            item_bounds=bundle.item_bounds,
        )

    return build


def make_variant_specs(
    base_name: str,
    base_config: SNIEArchitectureConfig,
    encoder_factory: EncoderFactory,
    encoder_dim: int,
    bundle: TensorBundle,
    training: TrainingConfig,
    base_loss_weights: SNIELossWeights,
    variants: Sequence[ArchitectureVariant],
) -> list[ExperimentSpec]:
    specs = [
        ExperimentSpec(
            name=base_name,
            build_model=make_snie_model_factory(base_config, encoder_factory, encoder_dim, bundle, None),
            training=training,
            loss_weights=base_loss_weights,
        )
    ]
    for variant in variants:
        specs.append(
            ExperimentSpec(
                name=variant.name,
                build_model=make_snie_model_factory(base_config, encoder_factory, encoder_dim, bundle, variant),
                training=training,
                loss_weights=base_loss_weights,
            )
        )
    return specs


def run_experiment_matrix(
    specs: Sequence[ExperimentSpec],
    bundle: TensorBundle,
    splits: SplitIndices,
    random_states: Sequence[int | None],
    artifact_root: str | Path | None,
) -> ExperimentResult:
    """Run a matrix of method variants and random states.

    Artifacts, when enabled, are written under neutral replicate folders and do
    not include input examples, identifiers, model weights, or caller data paths.
    """

    rows: list[dict[str, float | str]] = []
    runs: dict[str, TrainingResult] = {}
    root = Path(artifact_root) if artifact_root is not None else None

    for spec in specs:
        for replicate_idx, random_state in enumerate(random_states):
            training = replace(spec.training, random_state=random_state)
            model = spec.build_model(random_state)
            artifact_dir = None
            run_key = f"{spec.name}/replicate_{replicate_idx}"
            if root is not None:
                artifact_dir = root / spec.name / f"replicate_{replicate_idx}"
            result = train_snie_model(model, bundle, splits, training, spec.loss_weights, artifact_dir=artifact_dir)
            runs[run_key] = result
            rows.append(
                {
                    "variant": spec.name,
                    "replicate": float(replicate_idx),
                    **{key: float(value) for key, value in result.summary.items()},
                }
            )

    if root is not None:
        write_experiment_summary(rows, root / "summary.csv")
    return ExperimentResult(rows=rows, runs=runs)


def write_experiment_summary(rows: Sequence[dict[str, float | str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    leading = [key for key in ("variant", "replicate") if key in keys]
    fieldnames = leading + [key for key in keys if key not in leading]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
