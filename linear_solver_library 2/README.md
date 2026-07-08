# Numerical Linear-System Solver Library

A Python project implementing and benchmarking direct and iterative methods for solving linear systems:

- Gaussian elimination with partial pivoting
- Jacobi iteration
- Gauss-Seidel iteration
- Convergence diagnostics
- Numerical stability and conditioning analysis
- Covariance-system solves for mean-variance portfolio optimization

## Quick Start

```bash
cd linear_solver_library
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_benchmarks.py
python run_portfolio_demo.py
```

Outputs are saved in the `outputs/` folder.

## Project Structure

```text
linear_solver_library/
  run_benchmarks.py          # benchmark direct/iterative solvers
  run_portfolio_demo.py      # covariance solve for portfolio optimization
  requirements.txt
  src/lin_solvers/
    direct.py                # Gaussian elimination with partial pivoting
    iterative.py             # Jacobi and Gauss-Seidel
    metrics.py               # residuals, errors, condition numbers
    portfolio.py             # mean-variance optimization utilities
    plotting.py              # plots for convergence and benchmark results
```

## Notes

Iterative solvers are not guaranteed to converge for every matrix. They work best on diagonally dominant or symmetric positive definite systems. The benchmark includes controlled systems so the convergence behavior is easy to study.
