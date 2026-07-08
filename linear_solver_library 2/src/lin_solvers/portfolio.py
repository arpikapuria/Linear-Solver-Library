"""Mean-variance portfolio optimization utilities."""

from __future__ import annotations

import numpy as np

from .direct import gaussian_elimination_partial_pivot


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio weights.

    Formula: w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)
    """
    cov = np.asarray(cov, dtype=float)
    ones = np.ones(cov.shape[0])
    inv_ones = gaussian_elimination_partial_pivot(cov, ones)
    return inv_ones / (ones @ inv_ones)


def tangency_portfolio_weights(cov: np.ndarray, expected_returns: np.ndarray, risk_free_rate: float = 0.0) -> np.ndarray:
    """Tangency portfolio weights.

    Formula: w = Sigma^{-1}(mu-rf*1) / 1' Sigma^{-1}(mu-rf*1)
    """
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    excess = mu - risk_free_rate
    raw = gaussian_elimination_partial_pivot(cov, excess)
    return raw / raw.sum()


def portfolio_stats(weights: np.ndarray, expected_returns: np.ndarray, cov: np.ndarray, risk_free_rate: float = 0.0) -> dict[str, float]:
    weights = np.asarray(weights)
    mu = np.asarray(expected_returns)
    cov = np.asarray(cov)
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = float((ret - risk_free_rate) / vol) if vol > 0 else np.nan
    return {"expected_return": ret, "volatility": vol, "sharpe": sharpe}
