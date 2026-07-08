"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_convergence(histories: dict[str, list[float]], out_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.semilogy(range(1, len(history) + 1), history, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Residual norm ||b - Ax||₂")
    plt.title("Iterative Solver Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_benchmark_table(df: pd.DataFrame, out_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    for method in df["method"].unique():
        subset = df[df["method"] == method]
        plt.plot(subset["n"], subset["time_seconds"], marker="o", label=method)
    plt.xlabel("Matrix size n")
    plt.ylabel("Runtime seconds")
    plt.title("Solver Runtime Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
