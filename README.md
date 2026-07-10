# SNIE Public Architecture Release

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
