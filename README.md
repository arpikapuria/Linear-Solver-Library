# Numerical Linear System Solver Library

## Overview

This project is a Python library for solving systems of linear equations using numerical methods. It implements direct and iterative solvers, tracks convergence behavior, and benchmarks performance across different matrix sizes.

The project connects linear algebra concepts with practical Python programming and numerical computing.

## Project Goals

- Implement Gaussian elimination with partial pivoting
- Implement Jacobi iteration
- Implement Gauss-Seidel iteration
- Compare accuracy and convergence behavior
- Track residual norms and relative errors
- Analyze condition numbers
- Benchmark solver performance
- Apply linear system solving to a portfolio optimization example

## Methods Implemented

### Gaussian Elimination

A direct method that transforms a system into upper triangular form and solves using back substitution.

### Jacobi Method

An iterative method that updates each variable using values from the previous iteration.

### Gauss-Seidel Method

An iterative method that updates variables using the newest available values during each iteration.

## Tools and Libraries

- Python
- NumPy
- pandas
- matplotlib

## Skills Demonstrated

- Numerical linear algebra
- Algorithm implementation
- Iterative methods
- Performance benchmarking
- Error analysis
- Data visualization
- Python package organization

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
