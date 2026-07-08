"""Portfolio optimization demo using covariance-system solves."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from lin_solvers.portfolio import min_variance_weights, tangency_portfolio_weights, portfolio_stats
from lin_solvers.metrics import condition_number

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def simulate_asset_returns(n_assets: int = 8, n_days: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Create a realistic positive definite covariance matrix through factor structure.
    factor_loadings = rng.normal(0.6, 0.25, size=(n_assets, 3))
    factor_cov = np.diag([0.00025, 0.00016, 0.00010])
    idiosyncratic = np.diag(rng.uniform(0.00005, 0.00018, size=n_assets))
    cov = factor_loadings @ factor_cov @ factor_loadings.T + idiosyncratic

    daily_mu = rng.uniform(0.00015, 0.00065, size=n_assets)
    returns = rng.multivariate_normal(daily_mu, cov, size=n_days)
    tickers = [f"Asset_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=tickers)


def main():
    returns = simulate_asset_returns()
    expected_returns = returns.mean().to_numpy() * 252
    cov = returns.cov().to_numpy() * 252

    mv_weights = min_variance_weights(cov)
    tan_weights = tangency_portfolio_weights(cov, expected_returns, risk_free_rate=0.02)

    rows = []
    for name, weights in [
        ("Global minimum variance", mv_weights),
        ("Tangency portfolio", tan_weights),
    ]:
        stats = portfolio_stats(weights, expected_returns, cov, risk_free_rate=0.02)
        for ticker, weight in zip(returns.columns, weights):
            rows.append({"portfolio": name, "asset": ticker, "weight": weight})
        print(f"\n{name}")
        print(f"Expected return: {stats['expected_return']:.2%}")
        print(f"Volatility:       {stats['volatility']:.2%}")
        print(f"Sharpe:           {stats['sharpe']:.3f}")

    weights_df = pd.DataFrame(rows)
    weights_df.to_csv(OUTPUT_DIR / "portfolio_weights.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "covariance_condition_number", "value": condition_number(cov)},
            {"metric": "num_assets", "value": len(returns.columns)},
            {"metric": "num_days", "value": len(returns)},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "portfolio_covariance_summary.csv", index=False)

    print("\nPortfolio demo complete. Results saved to outputs/")


if __name__ == "__main__":
    main()
