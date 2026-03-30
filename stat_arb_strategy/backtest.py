"""
backtest.py
-----------
Runs the rolling Ridge / z-score stat arb strategy on train and test splits
for every pair in candidate_pairs and returns a summary results table.

Usage
-----
    from stat_arb_strategy.backtest import run_backtest
    results_df, pnl_curves = run_backtest()
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm.auto import tqdm

from stat_arb import rolling_ridge_zscore
from data_tickers import candidate_pairs


# ---------------------------------------------------------------------------
# Strategy execution on a single pre-split series pair
# ---------------------------------------------------------------------------

def run_strategy(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    adf_threshold: float = 0.50,
) -> dict:
    """
    Run the z-score mean-reversion strategy on s1/s2 and return performance metrics.

    Returns a dict with keys:
        position, daily_pnl, cum_pnl, z_scores, spread, adf_pvals,
        total_pnl, sharpe, n_trades, pct_in_regime
    """
    z_scores, spread, adf_pvals = rolling_ridge_zscore(s1, s2, window=window)

    # Align to common non-null index
    idx = (z_scores.dropna().index
           .intersection(spread.dropna().index)
           .intersection(adf_pvals.dropna().index))

    z   = z_scores.loc[idx]
    sp  = spread.loc[idx]
    ap  = adf_pvals.loc[idx]

    position = pd.Series(0, index=z.index, dtype=float)
    cp = 0
    for i in range(1, len(z)):
        zv = z.iloc[i]
        in_regime = ap.iloc[i] < adf_threshold

        if cp == 0:
            if in_regime and zv >  entry_threshold: cp = -1
            elif in_regime and zv < -entry_threshold: cp =  1
        elif cp ==  1 and zv >= -exit_threshold: cp = 0
        elif cp == -1 and zv <=  exit_threshold: cp = 0

        if not in_regime:
            cp = 0

        position.iloc[i] = cp

    daily_pnl = (position.shift(1) * sp.diff()).dropna()
    cum_pnl   = daily_pnl.cumsum()
    std       = daily_pnl.std()
    sharpe    = daily_pnl.mean() / std * np.sqrt(252) if std > 0 else np.nan
    n_trades  = int((position.diff().abs() > 0).sum())

    return {
        'position':      position,
        'daily_pnl':     daily_pnl,
        'cum_pnl':       cum_pnl,
        'z_scores':      z,
        'spread':        sp,
        'adf_pvals':     ap,
        'total_pnl':     round(cum_pnl.iloc[-1], 2) if len(cum_pnl) else np.nan,
        'sharpe':        round(sharpe, 2),
        'n_trades':      n_trades,
        'pct_in_regime': round((ap < adf_threshold).mean() * 100, 1),
    }


# ---------------------------------------------------------------------------
# Full backtest across all candidate pairs
# ---------------------------------------------------------------------------

def run_backtest(
    start: str = '2023-01-15',
    end: str = '2026-03-12',
    test_size: float = 0.2,
    window: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    adf_threshold: float = 0.50,
    coint_pval_cutoff: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    """
    Download data, screen for cointegration, then backtest the strategy on
    all passing pairs across train and test splits.

    Returns
    -------
    results_df  : pd.DataFrame  Summary table (one row per pair per split)
    pnl_curves  : dict          {'{name} Test': cum_pnl Series, ...}
    """
    # --- Download all tickers in one call ---
    all_tickers = list({t for pair in candidate_pairs for t in pair[:2]})
    print("Downloading data...")
    raw = yf.download(
        tickers=all_tickers, start=start, end=end, auto_adjust=True
    )['Close'].interpolate(method='linear')

    results    = []
    pnl_curves = {}

    for t1, t2, name in tqdm(candidate_pairs, desc='Backtesting pairs'):
        if t1 not in raw.columns or t2 not in raw.columns:
            print(f"  Skipping {name} — missing data")
            continue

        common = raw[[t1, t2]].dropna()
        if len(common) < window + 50:
            print(f"  Skipping {name} — too few rows ({len(common)})")
            continue

        s1 = common[t1]
        s2 = common[t2]

        s1_train, s1_test, s2_train, s2_test = train_test_split(
            s1, s2, test_size=test_size, shuffle=False
        )

        # Cointegration screen on TRAINING data only — avoids test-period contamination
        resid = OLS(s2_train, add_constant(s1_train)).fit().resid
        pval  = adfuller(resid, autolag='AIC')[1]
        if pval >= coint_pval_cutoff:
            print(f"  Skipping {name} — not cointegrated on train data (p={pval:.3f})")
            continue

        print(f"  Running {name} (train ADF p={pval:.3f})")

        for split_label, x, y in [('Train', s1_train, s2_train), ('Test', s1_test, s2_test)]:
            if len(x) < window + 10:
                continue

            res = run_strategy(x, y, window, entry_threshold, exit_threshold, adf_threshold)

            results.append({
                'Pair':          name,
                'Split':         split_label,
                'Total PnL ($)': res['total_pnl'],
                'Sharpe':        res['sharpe'],
                'Trades':        res['n_trades'],
                '% In Regime':   res['pct_in_regime'],
            })

            if split_label == 'Test':
                pnl_curves[name] = res['cum_pnl']

    results_df = (
        pd.DataFrame(results)
        .sort_values(['Split', 'Sharpe'], ascending=[True, False])
        .reset_index(drop=True)
    )

    print("\n=== Backtest Results ===")
    print(results_df.to_string(index=False))

    return results_df, pnl_curves
