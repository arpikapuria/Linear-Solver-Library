"""Run numerical solver benchmarks."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from lin_solvers.direct import gaussian_elimination_partial_pivot
from lin_solvers.iterative import jacobi, gauss_seidel
from lin_solvers.metrics import condition_number, relative_error, residual_norm, is_diagonally_dominant
from lin_solvers.plotting import plot_benchmark_table, plot_convergence

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_spd_diagonally_dominant_system(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M.T @ M
    A += n * np.eye(n)  # improves conditioning and diagonal dominance behavior
    x_true = rng.normal(size=n)
    b = A @ x_true
    return A, b, x_true


def time_solver(name, solve_fn, A, b, x_true):
    start = time.perf_counter()
    result = solve_fn(A, b)
    elapsed = time.perf_counter() - start

    if hasattr(result, "x"):
        x_hat = result.x
        converged = result.converged
        iterations = result.iterations
    else:
        x_hat = result
        converged = True
        iterations = np.nan

    return {
        "method": name,
        "n": A.shape[0],
        "time_seconds": elapsed,
        "residual_norm": residual_norm(A, x_hat, b),
        "relative_error": relative_error(x_hat, x_true),
        "condition_number": condition_number(A),
        "diagonally_dominant": is_diagonally_dominant(A),
        "converged": converged,
        "iterations": iterations,
    }


def main():
    rows = []
    sizes = [10, 25, 50, 100]

    for n in sizes:
        A, b, x_true = make_spd_diagonally_dominant_system(n, seed=42 + n)
        rows.append(time_solver("Gaussian elimination partial pivot", gaussian_elimination_partial_pivot, A, b, x_true))
        rows.append(time_solver("Jacobi", lambda A, b: jacobi(A, b, tol=1e-8, max_iter=20_000), A, b, x_true))
        rows.append(time_solver("Gauss-Seidel", lambda A, b: gauss_seidel(A, b, tol=1e-8, max_iter=20_000), A, b, x_true))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "solver_benchmark_results.csv", index=False)
    plot_benchmark_table(df, OUTPUT_DIR / "solver_runtime_benchmark.png")

    # Detailed convergence plot for one representative system.
    A, b, _ = make_spd_diagonally_dominant_system(50, seed=123)
    jacobi_result = jacobi(A, b, tol=1e-10, max_iter=20_000)
    gs_result = gauss_seidel(A, b, tol=1e-10, max_iter=20_000)
    plot_convergence(
        {
            "Jacobi": jacobi_result.residual_history,
            "Gauss-Seidel": gs_result.residual_history,
        },
        OUTPUT_DIR / "iterative_convergence.png",
    )

    print("Benchmark complete. Results saved to outputs/")
    print(df)


if __name__ == "__main__":
    main()
