from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConditionalMeanTable:
    """Generic scale-total conditional mean table for residual targets."""

    item_to_scale: np.ndarray
    smoothing: float
    global_mean: np.ndarray | None = None
    tables: list[dict[int, float]] | None = None

    def fit(self, y: np.ndarray, scale_totals: np.ndarray, train_indices: np.ndarray) -> "ConditionalMeanTable":
        train_indices = np.asarray(train_indices, dtype=np.int64)
        if train_indices.size == 0:
            raise ValueError("train_indices must not be empty")
        self.global_mean = y[train_indices].mean(axis=0)
        self.tables = self._fit_tables(y, scale_totals, train_indices, self.global_mean)
        return self

    def residuals(self, y: np.ndarray, scale_totals: np.ndarray) -> np.ndarray:
        return np.asarray(y - self.predict(scale_totals), dtype=np.float32)

    def predict(self, scale_totals: np.ndarray) -> np.ndarray:
        if self.tables is None or self.global_mean is None:
            raise RuntimeError("ConditionalMeanTable must be fitted before calling predict")
        out = np.zeros((scale_totals.shape[0], len(self.item_to_scale)), dtype=np.float32)
        for item_idx, scale_idx in enumerate(self.item_to_scale):
            table = self.tables[item_idx]
            known_totals = np.asarray(sorted(table.keys()), dtype=np.int64)
            for row_idx, total in enumerate(scale_totals[:, int(scale_idx)]):
                key = int(round(float(total)))
                if key in table:
                    out[row_idx, item_idx] = table[key]
                elif known_totals.size:
                    nearest = int(known_totals[np.argmin(np.abs(known_totals - key))])
                    out[row_idx, item_idx] = table[nearest]
                else:
                    out[row_idx, item_idx] = float(self.global_mean[item_idx])
        return out

    def item_bounds(self, y: np.ndarray, residuals: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        raw_span = np.maximum(1.0, y[train_indices].max(axis=0) - y[train_indices].min(axis=0))
        residual_span = np.maximum(np.abs(residuals[train_indices]).max(axis=0), raw_span / 2.0)
        return np.asarray(residual_span + np.finfo(np.float32).eps, dtype=np.float32)

    def _fit_tables(
        self,
        y: np.ndarray,
        scale_totals: np.ndarray,
        indices: np.ndarray,
        global_mean: np.ndarray,
    ) -> list[dict[int, float]]:
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        tables: list[dict[int, float]] = []
        for item_idx, scale_idx in enumerate(self.item_to_scale):
            buckets: dict[int, list[float]] = {}
            for row_idx in indices:
                total = int(round(float(scale_totals[row_idx, int(scale_idx)])))
                buckets.setdefault(total, []).append(float(y[row_idx, item_idx]))
            item_table: dict[int, float] = {}
            for total, values in buckets.items():
                count = len(values)
                mean = float(np.mean(values))
                item_table[total] = float((count * mean + self.smoothing * global_mean[item_idx]) / (count + self.smoothing))
            tables.append(item_table)
        return tables
