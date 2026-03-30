import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Raw price series
# ---------------------------------------------------------------------------

def plot_prices(s1: pd.Series, s2: pd.Series, label1: str = 'Series 1', label2: str = 'Series 2'):
    """Plot the two raw close price series on separate axes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(s1, color='steelblue')
    axes[0].set_title(f'{label1} Close Price')
    axes[0].set_ylabel('Price ($)')

    axes[1].plot(s2, color='darkorange')
    axes[1].set_title(f'{label2} Close Price')
    axes[1].set_ylabel('Price ($)')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Scatter — price relationship
# ---------------------------------------------------------------------------

def plot_scatter(s1: pd.Series, s2: pd.Series, label1: str = 'X', label2: str = 'Y'):
    """Scatter plot of s2 vs s1 to visualise the linear relationship."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s1, s2, marker='x', alpha=0.5, s=15)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(f'{label2} vs {label1}')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. OLS residuals (static spread)
# ---------------------------------------------------------------------------

def plot_residuals(residuals: pd.Series, title: str = 'OLS Residuals (Spread)'):
    """Plot OLS residuals as a time series and histogram."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(residuals.index, residuals, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title(title)
    axes[0].set_ylabel('Residual ($)')

    axes[1].hist(residuals, bins=40, edgecolor='black', color='steelblue')
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Residuals')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Rolling z-score
# ---------------------------------------------------------------------------

def plot_zscore(
    z_scores: pd.Series,
    entry_threshold: float = 2.0,
    adf_pvals: pd.Series = None,
    adf_threshold: float = 0.50,
    title: str = 'Rolling Z-Score',
):
    """
    Plot the rolling z-score with entry threshold lines.
    Optionally shade periods where the ADF regime filter is inactive.
    """
    z = z_scores.dropna()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(z, label='Z-Score', color='steelblue')
    ax.axhline( entry_threshold, color='red',   linestyle='--', label=f'+{entry_threshold}')
    ax.axhline(-entry_threshold, color='green', linestyle='--', label=f'-{entry_threshold}')
    ax.axhline(0, color='black', linewidth=0.5)

    if adf_pvals is not None:
        ap = adf_pvals.reindex(z.index)
        ax.fill_between(
            z.index, z.min(), z.max(),
            where=(ap >= adf_threshold),
            alpha=0.15, color='grey', label='No regime (ADF)'
        )

    ax.set_title(title)
    ax.set_ylabel('Z-Score')
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Full strategy dashboard (z-score / ADF / position / PnL)
# ---------------------------------------------------------------------------

def plot_strategy(
    z_scores: pd.Series,
    position: pd.Series,
    cum_pnl: pd.Series,
    adf_pvals: pd.Series = None,
    entry_threshold: float = 2.0,
    adf_threshold: float = 0.50,
    title: str = 'Strategy',
):
    """4-panel dashboard: z-score, ADF p-value, position, cumulative PnL."""
    n_panels = 4 if adf_pvals is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)

    z = z_scores.dropna()

    # Panel 0 — Z-score
    axes[0].plot(z, label='Z-Score', color='steelblue')
    axes[0].axhline( entry_threshold, color='red',   linestyle='--', label=f'+{entry_threshold}')
    axes[0].axhline(-entry_threshold, color='green', linestyle='--', label=f'-{entry_threshold}')
    axes[0].axhline(0, color='black', linewidth=0.5)
    if adf_pvals is not None:
        ap = adf_pvals.reindex(z.index)
        axes[0].fill_between(
            z.index, z.min(), z.max(),
            where=(ap >= adf_threshold),
            alpha=0.15, color='grey', label='No regime'
        )
    axes[0].set_title(f'Z-Score — {title}')
    axes[0].legend(loc='upper right')

    panel = 1
    if adf_pvals is not None:
        axes[panel].plot(adf_pvals.dropna(), color='steelblue', label='ADF p-value')
        axes[panel].axhline(adf_threshold, color='red', linestyle='--', label=f'threshold={adf_threshold}')
        axes[panel].set_title('ADF p-value (regime filter)')
        axes[panel].legend(loc='upper right')
        panel += 1

    # Position
    axes[panel].plot(position, drawstyle='steps-post', color='purple', label='Position')
    axes[panel].set_title('Position  (+1 = Long spread, -1 = Short spread, 0 = Flat)')
    axes[panel].set_yticks([-1, 0, 1])
    axes[panel].legend(loc='upper right')
    panel += 1

    # Cumulative PnL
    axes[panel].plot(cum_pnl, color='darkorange', label='Cumulative PnL ($)')
    axes[panel].axhline(0, color='black', linewidth=0.5)
    axes[panel].set_title('Cumulative PnL')
    axes[panel].legend(loc='upper right')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Multi-pair results leaderboard (DataFrame summary)
# ---------------------------------------------------------------------------

def build_results_table(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of per-pair result dicts into a formatted summary DataFrame.

    Each dict should contain:
        Pair, Split, Total PnL, Sharpe, Trades, % In Regime
    """
    df = pd.DataFrame(results)
    df = df.sort_values(['Split', 'Sharpe'], ascending=[True, False])
    df = df.reset_index(drop=True)
    return df


def plot_multi_pair_pnl(
    pnl_dict: dict[str, pd.Series],
    title: str = 'Cumulative PnL by Pair (Test)',
):
    """
    Overlay cumulative PnL curves for multiple pairs on one chart.

    Parameters
    ----------
    pnl_dict : dict  {pair_name: cumulative_pnl_series}
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, cum_pnl in pnl_dict.items():
        ax.plot(cum_pnl, label=name)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel('Cumulative PnL ($)')
    ax.legend(loc='upper left', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_sharpe_bar(results_df: pd.DataFrame, split: str = 'Test'):
    """Bar chart of Sharpe ratios for all pairs in a given split."""
    df = results_df[results_df['Split'] == split].sort_values('Sharpe', ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    colors = ['green' if s > 0 else 'red' for s in df['Sharpe']]
    ax.barh(df['Pair'], df['Sharpe'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'Sharpe Ratio by Pair — {split}')
    ax.set_xlabel('Sharpe Ratio')
    plt.tight_layout()
    plt.show()
