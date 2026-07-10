from __future__ import annotations

import torch


def normalize_signed_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Degree-normalize a signed item graph and bound its spectral norm."""

    graph = adjacency.float().clone()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    graph.fill_diagonal_(0.0)
    degree = graph.abs().sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = degree[mask].rsqrt()
    graph = inv_sqrt[:, None] * graph * inv_sqrt[None, :]

    if graph.numel() > 0:
        norm = torch.linalg.matrix_norm(graph, ord=2)
        if torch.isfinite(norm) and norm > 1:
            graph = graph / norm
    graph.fill_diagonal_(0.0)
    return graph


def validate_graph(graph: torch.Tensor, num_items: int) -> torch.Tensor:
    """Return a float graph tensor after basic architectural validation."""

    graph = graph.float()
    expected = (num_items, num_items)
    if tuple(graph.shape) != expected:
        raise ValueError(f"graph shape must be {expected}, got {tuple(graph.shape)}")
    return graph
