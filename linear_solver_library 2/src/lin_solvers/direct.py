"""Direct linear-system solvers."""

from __future__ import annotations

import numpy as np


def gaussian_elimination_partial_pivot(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A:
        Square coefficient matrix.
    b:
        Right-hand-side vector.

    Returns
    -------
    x:
        Solution vector.
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).reshape(-1)

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if b.shape[0] != n:
        raise ValueError("b must have the same number of rows as A")

    # Forward elimination with partial pivoting.
    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        if np.isclose(A[pivot_row, k], 0.0):
            raise np.linalg.LinAlgError("Matrix is singular or nearly singular")

        if pivot_row != k:
            A[[k, pivot_row], :] = A[[pivot_row, k], :]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        for i in range(k + 1, n):
            multiplier = A[i, k] / A[k, k]
            A[i, k:] -= multiplier * A[k, k:]
            b[i] -= multiplier * b[k]

    if np.isclose(A[-1, -1], 0.0):
        raise np.linalg.LinAlgError("Matrix is singular or nearly singular")

    # Back substitution.
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]

    return x
