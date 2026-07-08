"""Iterative linear-system solvers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class IterativeResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_history: list[float]


def _validate_inputs(A: np.ndarray, b: np.ndarray, x0: np.ndarray | None):
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).reshape(-1)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if b.shape[0] != n:
        raise ValueError("b must have the same number of rows as A")
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise np.linalg.LinAlgError("A has a zero diagonal entry")
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float, copy=True).reshape(-1)
    if x.shape[0] != n:
        raise ValueError("x0 must have the same length as b")
    return A, b, x


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> IterativeResult:
    """Solve Ax=b using Jacobi iteration."""
    A, b, x = _validate_inputs(A, b, x0)
    D = np.diag(A)
    R = A - np.diagflat(D)
    residual_history = []

    for iteration in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        residual = np.linalg.norm(b - A @ x_new, ord=2)
        residual_history.append(float(residual))

        if residual < tol:
            return IterativeResult(x_new, True, iteration, residual_history)
        x = x_new

    return IterativeResult(x, False, max_iter, residual_history)


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> IterativeResult:
    """Solve Ax=b using Gauss-Seidel iteration."""
    A, b, x = _validate_inputs(A, b, x0)
    n = A.shape[0]
    residual_history = []

    for iteration in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            left = A[i, :i] @ x_new[:i]
            right = A[i, i + 1 :] @ x[i + 1 :]
            x_new[i] = (b[i] - left - right) / A[i, i]

        residual = np.linalg.norm(b - A @ x_new, ord=2)
        residual_history.append(float(residual))
        if residual < tol:
            return IterativeResult(x_new, True, iteration, residual_history)
        x = x_new

    return IterativeResult(x, False, max_iter, residual_history)
