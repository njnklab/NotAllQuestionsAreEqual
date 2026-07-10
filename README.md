# SNIE Public Architecture Release

This folder contains a sanitized release of the core SNIE network architecture.

Included:

- Graph-conditioned item-query network architecture
- Encoder interface and a generic feature-token encoder
- Signed adjacency normalization utilities
- Tensor-only training loop for already anonymized inputs
- Method losses, metrics, and experiment-matrix orchestration

Excluded:

- Private or licensed data sources
- Subject identifiers, labels, audio paths, and split files
- Generated artifacts from private runs
- Data-source-specific configuration and item names
- Tuned run parameters and private experiment presets

The public model is intentionally data-agnostic. Callers must provide:

- An encoder that returns token-level and pooled representations
- A signed item graph
- Per-item residual bounds or another public bounding policy
- Any training objective and hyperparameters used in their own setting
- Split indices and anonymized tensors for inputs, item targets, residual targets, and scale totals

## Files

- `snie_architecture/model.py`: SNIE forward architecture and output maps
- `snie_architecture/encoders.py`: generic encoder contract and feature-token encoder
- `snie_architecture/graph.py`: signed graph validation and normalization
- `snie_architecture/data.py`: tensor bundle, tensor loader, and generic split helper
- `snie_architecture/conditioning.py`: data-agnostic conditional mean table for residual targets
- `snie_architecture/losses.py`: SNIE residual, direct, rank, centering, reliability, scale, and evidence losses
- `snie_architecture/metrics.py`: same-total item ranking metrics and aggregate reports
- `snie_architecture/trainer.py`: training, validation, test, and optional neutral artifact writing
- `snie_architecture/experiments.py`: architecture variant and replicate experiment orchestration

No module in this folder reads raw files, defines private label schemas, embeds data paths, or saves model weights.

## Minimal Shape Contract

- Input to the public example encoder: `(batch, time, feature_dim)`
- Encoder output consumed by SNIE: token states `(batch, time, encoder_dim)` and pooled states `(batch, encoder_dim)`
- Graph input: signed item adjacency `(num_items, num_items)`
- Item bound input: positive vector `(num_items,)`
- Main outputs: item residuals `(batch, num_items)`, scale outputs `(batch, num_scales)`, token evidence maps `(batch, num_items, time)`, and graph flow maps `(batch, num_items, num_items)`

## Training Entry Point

Use `TensorBundle` for anonymized tensors and `train_snie_model(...)` for the loop. All optimization settings and loss weights are required from the caller; the release does not ship private presets or tuned defaults.

## Experiment Entry Point

Use `ArchitectureVariant`, `make_variant_specs(...)`, and `run_experiment_matrix(...)` to run ablations or repeated initializations. The artifact writer only emits neutral metric/history tables and optional arrays requested by the caller.
