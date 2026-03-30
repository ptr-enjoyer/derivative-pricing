"""
signals.py
----------
Generates live trading signals for all cointegrated pairs using the most
recent window of market data. Run this daily to get your trade signals.

Usage
-----
    python -m stat_arb_strategy.signals

    or from a script / notebook:

    from stat_arb_strategy.signals import generate_signals
    signals_df = generate_signals()
    print(signals_df)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from stat_arb import rolling_ridge_zscore
from data_tickers import candidate_pairs


# ---------------------------------------------------------------------------
# Core signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    lookback_days: int = 400,
    window: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    adf_threshold: float = 0.50,
    coint_pval_cutoff: float = 0.10,
) -> pd.DataFrame:
    """
    For every cointegrated pair, fit the rolling Ridge model on the most
    recent `window` days and return today's trading signal.

    Parameters
    ----------
    lookback_days    : int    How many calendar days of history to download
                              (needs to be >> window to fit the rolling model)
    window           : int    Rolling window length (must match backtest)
    entry_threshold  : float  Z-score level to enter a trade
    exit_threshold   : float  Z-score level to exit a trade
    adf_threshold    : float  Max ADF p-value to allow trading
    coint_pval_cutoff: float  Max full-history ADF p-value to include a pair

    Returns
    -------
    pd.DataFrame with columns:
        Pair, Ticker1, Ticker2,
        Z-Score, Spread, ADF p-val, In Regime,
        Signal, Action,
        Price1, Price2, Date
    """
    end   = date.today()
    start = end - timedelta(days=lookback_days)

    all_tickers = list({t for pair in candidate_pairs for t in pair[:2]})
    print(f"Downloading data {start} → {end} ...")
    raw = yf.download(
        tickers=all_tickers,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
    )['Close'].interpolate(method='linear')

    rows = []

    for t1, t2, name in candidate_pairs:
        if t1 not in raw.columns or t2 not in raw.columns:
            continue

        common = raw[[t1, t2]].dropna()
        if len(common) < window + 10:
            print(f"  {name}: skipping — only {len(common)} rows")
            continue

        # Cointegration screen on history excluding the most recent live window —
        # avoids recent regime breaks (e.g. tariff shocks) invalidating the screen
        screen_data = common.iloc[:-window]
        resid = OLS(screen_data[t2], add_constant(screen_data[t1])).fit().resid
        pval  = adfuller(resid, autolag='AIC')[1]
        if pval >= coint_pval_cutoff:
            print(f"  {name}: not cointegrated on history (p={pval:.3f}), skipping")
            continue

        s1 = common[t1]
        s2 = common[t2]

        # Run rolling model — only need the last row's output
        z_scores, spread, adf_pvals = rolling_ridge_zscore(s1, s2, window=window)

        # Get the most recent valid values
        last_z   = z_scores.dropna().iloc[-1]   if z_scores.dropna().shape[0]   > 0 else np.nan
        last_sp  = spread.dropna().iloc[-1]     if spread.dropna().shape[0]     > 0 else np.nan
        last_adf = adf_pvals.dropna().iloc[-1]  if adf_pvals.dropna().shape[0]  > 0 else np.nan
        last_date = z_scores.dropna().index[-1] if z_scores.dropna().shape[0]   > 0 else None

        in_regime = (not np.isnan(last_adf)) and (last_adf < adf_threshold)

        # Determine signal
        if np.isnan(last_z) or not in_regime:
            signal = 0
            action = 'FLAT — no regime'
        elif last_z > entry_threshold:
            signal = -1
            action = f'SHORT spread  (short {t2}, long {t1})'
        elif last_z < -entry_threshold:
            signal = 1
            action = f'LONG spread   (long {t2}, short {t1})'
        elif abs(last_z) < exit_threshold:
            signal = 0
            action = 'FLAT — z-score near zero'
        else:
            signal = 0
            action = 'HOLD / no new entry'

        rows.append({
            'Pair':      name,
            'Ticker1':   t1,
            'Ticker2':   t2,
            'Z-Score':   round(last_z, 3),
            'Spread':    round(last_sp, 4),
            'ADF p-val': round(last_adf, 4),
            'In Regime': in_regime,
            'Signal':    signal,
            'Action':    action,
            'Price1':    round(s1.iloc[-1], 4),
            'Price2':    round(s2.iloc[-1], 4),
            'Date':      last_date,
        })

    signals_df = pd.DataFrame(rows)

    # Sort: active signals first, then by abs z-score descending
    if not signals_df.empty:
        signals_df['_abs_z'] = signals_df['Z-Score'].abs()
        signals_df = (
            signals_df
            .sort_values(['Signal', '_abs_z'], ascending=[False, False])
            .drop(columns='_abs_z')
            .reset_index(drop=True)
        )

    return signals_df


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_signals(signals_df: pd.DataFrame):
    """Print a clean signal summary to the terminal."""
    if signals_df.empty:
        print("No signals generated.")
        return

    active = signals_df[signals_df['Signal'] != 0]
    flat   = signals_df[signals_df['Signal'] == 0]

    print(f"\n{'='*70}")
    print(f"  STAT ARB SIGNALS  —  {signals_df['Date'].max().date()}")
    print(f"{'='*70}")

    if active.empty:
        print("  No active signals today.")
    else:
        print(f"\n  ACTIVE SIGNALS ({len(active)} pairs)\n")
        for _, row in active.iterrows():
            direction = "LONG " if row['Signal'] == 1 else "SHORT"
            print(
                f"  [{direction}]  {row['Pair']:<28} "
                f"z={row['Z-Score']:+.2f}  ADF={row['ADF p-val']:.3f}  "
                f"|  {row['Action']}"
            )

    print(f"\n  FLAT ({len(flat)} pairs)\n")
    for _, row in flat.iterrows():
        print(
            f"  [FLAT ]  {row['Pair']:<28} "
            f"z={row['Z-Score']:+.2f}  ADF={row['ADF p-val']:.3f}  "
            f"|  {row['Action']}"
        )

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    signals_df = generate_signals()
    print_signals(signals_df)
    print(signals_df.to_string(index=False))



