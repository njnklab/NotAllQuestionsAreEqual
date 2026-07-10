from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoders import TokenEncoder
from .graph import validate_graph


@dataclass(frozen=True)
class SNIEArchitectureConfig:
    """Architecture switches and dimensions for the public SNIE network.

    Numerical values are intentionally required from the caller so this release
    does not publish experiment-specific hyperparameters.
    """

    num_items: int
    num_scales: int
    hidden_dim: int
    propagation_steps: int
    graph_query_steps: int
    top_fraction: float
    attention_temperature: float
    spectral_radius: float
    dropout: float
    use_graph_query: bool = True
    use_subject_adaptive_query: bool = True
    use_reliability_gate: bool = True
    use_spectral_bound: bool = True
    use_skip_connection: bool = True
    detach_scale_head: bool = True


class SNIEModel(nn.Module):
    """Structured Network for Item Evidence architecture.

    The model builds item-specific queries, attends to token evidence, predicts
    bounded item residuals, refines item states through a signed graph, and
    returns provenance maps that link graph flow back to token-level evidence.
    """

    def __init__(
        self,
        config: SNIEArchitectureConfig,
        encoder: TokenEncoder,
        encoder_dim: int,
        graph: torch.Tensor,
        item_bounds: torch.Tensor,
    ) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.encoder = encoder

        hidden = config.hidden_dim
        items = config.num_items
        self.token_proj = nn.Linear(encoder_dim, hidden)
        self.pool_proj = nn.Linear(encoder_dim, hidden)

        self.item_embed = nn.Parameter(torch.empty(items, hidden))
        self.subject_to_query = nn.Linear(hidden, hidden)
        self.query_self = nn.Linear(hidden, hidden)
        self.query_neighbor = nn.Linear(hidden, hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.query_activation = nn.GELU()

        self.to_query = nn.Linear(hidden, hidden, bias=False)
        self.to_key = nn.Linear(hidden, hidden, bias=False)
        self.to_value = nn.Linear(hidden, hidden, bias=False)

        self.direct_weight = nn.Parameter(torch.empty(items, hidden))
        self.direct_bias = nn.Parameter(torch.zeros(items))
        self.final_weight = nn.Parameter(torch.empty(items, hidden))
        self.final_bias = nn.Parameter(torch.zeros(items))

        self.reliability = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.message_norm = nn.LayerNorm(hidden)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.num_scales),
        )

        self.register_buffer("graph", validate_graph(graph, items))
        self.register_buffer("item_bounds", self._validate_bounds(item_bounds, items))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embed)
        nn.init.xavier_uniform_(self.direct_weight)
        nn.init.xavier_uniform_(self.final_weight)

    def forward(
        self,
        x: torch.Tensor,
        graph_override: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_features, pooled_features = self.encoder(x)
        tokens = self.token_proj(token_features)
        pooled = self.pool_proj(pooled_features)

        graph = self.graph if graph_override is None else validate_graph(graph_override.to(x.device), self.config.num_items)
        if edge_mask is not None:
            graph = graph * validate_graph(edge_mask.to(x.device), self.config.num_items)

        queries = self._queries(pooled, graph)
        attention, item_states = self._attend(queries, tokens)
        direct_delta = self._bounded_item_head(item_states, self.direct_weight, self.direct_bias)
        reliability = torch.sigmoid(self.reliability(item_states)).squeeze(-1)

        refined_states, normalized_flow, flow_strength = self._propagate(item_states, pooled, reliability, graph)
        final_delta = self._bounded_item_head(refined_states, self.final_weight, self.final_bias)

        scale_input = pooled.detach() if self.config.detach_scale_head else pooled
        total_pred = self.scale_head(scale_input)
        provenance = self._provenance(attention, flow_strength)
        return {
            "delta": final_delta,
            "delta_direct": direct_delta,
            "rho": reliability,
            "z": item_states,
            "z_refined": refined_states,
            "alpha": attention,
            "gamma": flow_strength,
            "r_bar": normalized_flow,
            "total_pred": total_pred,
            "provenance": provenance,
        }

    def _queries(self, pooled: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        batch = pooled.shape[0]
        query = self.item_embed.unsqueeze(0).expand(batch, -1, -1)
        if self.config.use_subject_adaptive_query:
            query = query + self.subject_to_query(pooled).unsqueeze(1)
        if not self.config.use_graph_query or self.config.graph_query_steps <= 0:
            return query

        for _ in range(self.config.graph_query_steps):
            own = self.query_self(query)
            neighbor = torch.einsum("ij,bjh->bih", graph, self.query_neighbor(query))
            query = self.query_norm(query + self.query_activation(own + neighbor))
        return query

    def _attend(self, queries: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.to_query(queries)
        key = self.to_key(tokens)
        value = self.to_value(tokens)
        scores = torch.einsum("bih,bth->bit", query, key) / query.shape[-1] ** 0.5
        scores = scores / max(self.config.attention_temperature, 1e-6)

        token_count = scores.shape[-1]
        top_count = max(1, int(round(self.config.top_fraction * token_count)))
        top_idx = torch.topk(scores, k=min(top_count, token_count), dim=-1).indices
        hard_mask = torch.zeros_like(scores, dtype=torch.bool)
        hard_mask.scatter_(-1, top_idx, True)

        soft_mask = torch.softmax(scores, dim=-1)
        straight_through_mask = hard_mask.to(scores.dtype) + soft_mask - soft_mask.detach()
        stable_scores = scores - scores.max(dim=-1, keepdim=True).values
        weights = stable_scores.exp() * straight_through_mask
        attention = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        item_states = torch.einsum("bit,bth->bih", attention, value)
        return attention, item_states

    def _bounded_item_head(self, z: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bih,ih->bi", z, weight) + bias
        return self.item_bounds.unsqueeze(0) * torch.tanh(logits)

    def _propagate(
        self,
        z: torch.Tensor,
        pooled: torch.Tensor,
        reliability: torch.Tensor,
        graph: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, items, hidden = z.shape
        if self.config.propagation_steps <= 0:
            flow_strength = torch.zeros(batch, items, items, device=z.device, dtype=z.dtype)
            eye = torch.eye(items, device=z.device, dtype=z.dtype).unsqueeze(0)
            flow_strength = flow_strength + eye * z.norm(dim=-1).unsqueeze(1)
            empty_flow = torch.zeros(batch, items, items, device=z.device, dtype=z.dtype)
            return z, empty_flow, flow_strength

        subject = pooled[:, None, None, :].expand(batch, items, items, hidden)
        target = z[:, :, None, :].expand(batch, items, items, hidden)
        source = z[:, None, :, :].expand(batch, items, items, hidden)
        edge = graph[None, :, :, None].expand(batch, items, items, 1)

        if self.config.use_reliability_gate:
            gate = torch.sigmoid(self.gate(torch.cat([subject, target, source, edge], dim=-1))).squeeze(-1)
            source_reliability = reliability[:, None, :]
        else:
            gate = torch.ones(batch, items, items, device=z.device, dtype=z.dtype)
            source_reliability = torch.ones(batch, 1, items, device=z.device, dtype=z.dtype)

        flow = gate * source_reliability * graph.unsqueeze(0)
        if self.config.use_spectral_bound:
            norm = torch.linalg.matrix_norm(flow, ord=2, dim=(-2, -1)).clamp_min(1.0)
            normalized_flow = self.config.spectral_radius * flow / norm[:, None, None]
        else:
            normalized_flow = flow

        refined = z
        flow_strength = torch.zeros(batch, items, items, device=z.device, dtype=z.dtype)
        for _ in range(self.config.propagation_steps):
            messages = self.message(refined)
            update = torch.einsum("bij,bjh->bih", normalized_flow, messages)
            flow_strength = flow_strength + normalized_flow.abs() * messages.norm(dim=-1)[:, None, :]
            refined = self.message_norm(refined + update) if self.config.use_skip_connection else self.message_norm(update)

        eye = torch.eye(items, device=z.device, dtype=z.dtype).unsqueeze(0)
        flow_strength = flow_strength + eye * z.norm(dim=-1).unsqueeze(1)
        return refined, normalized_flow, flow_strength

    @staticmethod
    def _provenance(attention: torch.Tensor, flow_strength: torch.Tensor) -> torch.Tensor:
        item_weights = flow_strength / flow_strength.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bij,bjt->bit", item_weights, attention)

    @staticmethod
    def _validate_bounds(item_bounds: torch.Tensor, num_items: int) -> torch.Tensor:
        bounds = item_bounds.float()
        if tuple(bounds.shape) != (num_items,):
            raise ValueError(f"item_bounds shape must be ({num_items},), got {tuple(bounds.shape)}")
        if not torch.all(bounds > 0):
            raise ValueError("item_bounds must be positive")
        return bounds

    @staticmethod
    def _validate_config(config: SNIEArchitectureConfig) -> None:
        if config.num_items <= 0:
            raise ValueError("num_items must be positive")
        if config.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if config.propagation_steps < 0:
            raise ValueError("propagation_steps must be non-negative")
        if config.graph_query_steps < 0:
            raise ValueError("graph_query_steps must be non-negative")
        if not 0 < config.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        if config.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if config.spectral_radius <= 0:
            raise ValueError("spectral_radius must be positive")
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
