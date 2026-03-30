import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller


def load_pair(ticker1: str, ticker2: str, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """
    Download, align and validate two price series for use in a stat arb strategy.

    Parameters
    ----------
    ticker1 : str  e.g. 'CL=F'
    ticker2 : str  e.g. 'BZ=F'
    start   : str  e.g. '2023-01-01'
    end     : str  e.g. '2026-03-01'

    Returns
    -------
    s1, s2 : pd.Series
        Close price series aligned to a common date index, linearly interpolated.

    Raises
    ------
    ValueError if fewer than 100 common trading days are found.
    """
    raw = yf.download(
        tickers=[ticker1, ticker2],
        start=start,
        end=end,
        auto_adjust=True,
    )['Close']

    # Interpolate over missing days (e.g. one market closed, other open)
    raw = raw.interpolate(method='linear')

    # Align to common dates only
    common_dates = raw[ticker1].dropna().index.intersection(raw[ticker2].dropna().index)

    if len(common_dates) < 100:
        raise ValueError(
            f"Only {len(common_dates)} common trading days found for "
            f"{ticker1}/{ticker2} — need at least 100."
        )

    s1 = raw.loc[common_dates, ticker1].sort_index()
    s2 = raw.loc[common_dates, ticker2].sort_index()

    # Sanity check: confirm index is identical
    assert (s1.index == s2.index).all(), "Index mismatch after alignment"

    # Report cointegration status
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    resid = OLS(s2, add_constant(s1)).fit().resid
    pval = adfuller(resid, autolag='AIC')[1]
    coint = 'YES' if pval < 0.05 else 'NO'
    print(f"{ticker1}/{ticker2} | {len(s1)} common days | ADF p={pval:.4f} | Cointegrated: {coint}")

    return s1, s2


def rolling_ridge_zscore(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 60,
    n_folds: int = 3,
    fold_size: int = 10,
    lambdas: np.ndarray = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    For each rolling window, use 3-fold cross-validation to select the best
    Ridge lambda, fit beta/alpha on the window, then compute the spread z-score
    and ADF p-value at that timestep.

    Parameters
    ----------
    s1      : pd.Series  Endogenous series (X / predictor), e.g. WTI
    s2      : pd.Series  Exogenous series  (Y / target),   e.g. Brent
    window  : int        Rolling window length in days (default 60)
    n_folds : int        Number of sequential CV folds  (default 3)
    fold_size : int      Number of test observations per fold (default 10)
    lambdas : np.ndarray Ridge regularisation values to search over.
                         Defaults to np.logspace(-4, 4, 50)

    Returns
    -------
    z_scores  : pd.Series  Rolling z-score of the spread
    spread    : pd.Series  Raw spread value at each timestep
    adf_pvals : pd.Series  ADF p-value of the spread within each window
    """
    if lambdas is None:
        lambdas = np.logspace(-4, 4, 50)

    z_scores  = pd.Series(index=s1.index, dtype=float)
    spread    = pd.Series(index=s1.index, dtype=float)
    adf_pvals = pd.Series(index=s1.index, dtype=float)

    for t in range(window, len(s1)):
        endog = s1.iloc[t - window:t].values.reshape(-1, 1)
        exog  = s2.iloc[t - window:t].values

        # --- 3-fold sequential CV to pick best lambda ---
        best_beta, best_alpha, best_mse = None, None, np.inf

        for lam in lambdas:
            end    = window // 2
            splits = 0
            fold_mses = []

            while splits < n_folds:
                X_tr = endog[:end];        y_tr = exog[:end]
                X_te = endog[end:end + fold_size]
                y_te = exog[end:end + fold_size]

                if len(X_te) == 0:
                    break

                m = Ridge(lam).fit(X_tr, y_tr)
                fold_mses.append(mean_squared_error(y_te, m.predict(X_te)))
                end    += fold_size
                splits += 1

            if not fold_mses:
                continue

            avg_mse = np.mean(fold_mses)
            if avg_mse < best_mse:
                best_mse   = avg_mse
                best_beta  = m.coef_[0]
                best_alpha = m.intercept_

        if best_beta is None:
            continue

        # --- Compute spread, z-score and ADF on this window ---
        spread_w = exog - best_alpha - best_beta * endog.flatten()
        mu, std  = spread_w.mean(), spread_w.std()

        z_scores.iloc[t]  = (spread_w[-1] - mu) / std if std > 0 else np.nan
        spread.iloc[t]    = spread_w[-1]
        adf_pvals.iloc[t] = adfuller(spread_w, autolag='AIC')[1]

    return z_scores, spread, adf_pvals
