"""Numerical accuracy, stability, and conditioning metrics."""

from __future__ import annotations

import numpy as np


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(b) - np.asarray(A) @ np.asarray(x), ord=2))


def relative_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    denom = np.linalg.norm(x_true, ord=2)
    if np.isclose(denom, 0.0):
        return float(np.linalg.norm(x_hat - x_true, ord=2))
    return float(np.linalg.norm(x_hat - x_true, ord=2) / denom)


def condition_number(A: np.ndarray) -> float:
    return float(np.linalg.cond(A))


def is_diagonally_dominant(A: np.ndarray) -> bool:
    A = np.asarray(A)
    diagonal = np.abs(np.diag(A))
    off_diagonal = np.sum(np.abs(A), axis=1) - diagonal
    return bool(np.all(diagonal >= off_diagonal))
