import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_sentiment(sentiment_path: str) -> pd.DataFrame:
    """Load the Fear/Greed sentiment CSV and normalize the date field."""
    sentiment = pd.read_csv(sentiment_path)
    sentiment.columns = sentiment.columns.str.strip()

    if 'date' not in sentiment.columns:
        raise ValueError('Sentiment file must contain a `date` column.')

    sentiment['date'] = pd.to_datetime(sentiment['date'], errors='coerce').dt.normalize()
    missing_dates = sentiment['date'].isna().sum()
    if missing_dates:
        print(f'WARNING: {missing_dates} sentiment rows have invalid dates and will be ignored.')
        sentiment = sentiment[sentiment['date'].notna()].copy()

    sentiment['classification'] = sentiment['classification'].astype(str).str.strip()
    sentiment['sentiment_group'] = np.where(
        sentiment['classification'].str.contains('Fear', case=False, na=False),
        'Fear',
        np.where(
            sentiment['classification'].str.contains('Greed', case=False, na=False),
            'Greed',
            'Neutral',
        ),
    )

    return sentiment


def find_column(df: pd.DataFrame, aliases: List[str], required: bool = True) -> str:
    """Resolve a column name from multiple aliases for safe loading."""
    columns = {col.strip().lower(): col for col in df.columns}
    for alias in aliases:
        if alias.strip().lower() in columns:
            return columns[alias.strip().lower()]

    if required:
        raise ValueError(f'None of the expected columns were found: {aliases}')
    return ''


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a column to numeric and preserve invalid values as NaN."""
    return pd.to_numeric(series, errors='coerce')


def load_trades(trades_path: str) -> pd.DataFrame:
    """Load the trade dataset, normalize timestamps, and handle missing critical fields."""
    trades = pd.read_csv(trades_path)
    trades.columns = trades.columns.str.strip()

    # Standardize known names from the provided schema.
    column_map = {
        find_column(trades, ['Account', 'account']): 'Account',
        find_column(trades, ['Execution Price', 'execution price', 'Price']): 'Execution Price',
        find_column(trades, ['Closed PnL', 'closedPnL', 'closed pnl', 'PnL']): 'Closed PnL',
        find_column(trades, ['Side', 'side']): 'Side',
        find_column(trades, ['Timestamp IST', 'time', 'timestamp_ist']): 'Timestamp IST',
        find_column(trades, ['Timestamp', 'timestamp'], required=False): 'Timestamp',
    }

    trades = trades.rename(columns={k: v for k, v in column_map.items() if k})

    # If leverage is present, detect it. Otherwise initialize an empty placeholder.
    leverage_col = ''
    for alias in ['leverage', 'Leverage']:
        if alias in trades.columns:
            leverage_col = alias
            break

    if leverage_col:
        trades = trades.rename(columns={leverage_col: 'leverage'})
    else:
        trades['leverage'] = np.nan
        print('INFO: leverage column not found in trade data; placeholder column created.')

    # Parse numeric values consistently.
    trades['Execution Price'] = safe_numeric_series(trades['Execution Price'])
    trades['Closed PnL'] = safe_numeric_series(trades['Closed PnL'])
    trades['leverage'] = safe_numeric_series(trades['leverage'])

    # Parse the trade timestamp using the human-readable IST column first.
    trades['trade_datetime'] = pd.to_datetime(
        trades.get('Timestamp IST', ''),
        format='%d-%m-%Y %H:%M',
        errors='coerce',
    )

    # Fall back to epoch timestamp if IST parsing fails.
    if trades['trade_datetime'].isna().all() and 'Timestamp' in trades.columns:
        trades['trade_datetime'] = pd.to_datetime(trades['Timestamp'], unit='ms', errors='coerce')

    missing_datetimes = trades['trade_datetime'].isna().sum()
    if missing_datetimes:
        print(f'WARNING: {missing_datetimes} trade rows have invalid timestamps.')

    trades['trade_date'] = trades['trade_datetime'].dt.normalize()

    # Drop rows with critical missing values.
    critical = ['Execution Price', 'Closed PnL']
    pre_drop = len(trades)
    trades = trades.dropna(subset=critical).copy()
    dropped = pre_drop - len(trades)
    if dropped:
        print(f'INFO: Dropped {dropped} trade rows missing required numeric fields.')

    # Normalize trade side to Long/Short labels for analysis.
    trades['trade_direction'] = np.where(
        trades['Side'].astype(str).str.upper() == 'BUY',
        'Long',
        np.where(trades['Side'].astype(str).str.upper() == 'SELL', 'Short', trades['Side'].astype(str)),
    )

    return trades


def merge_datasets(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Merge trade data with daily sentiment on normalized date."""
    merged = trades.merge(
        sentiment[['date', 'classification', 'sentiment_group']],
        left_on='trade_date',
        right_on='date',
        how='left',
        validate='m:1',
    )

    missing_classification = merged['classification'].isna().sum()
    if missing_classification:
        print(f'CHECKPOINT 2: {missing_classification} active trade rows have no matching sentiment row.')
        merged = merged[merged['classification'].notna()].copy()
        print(f'CHECKPOINT 2: Dropped unmapped trades. Remaining merged rows: {len(merged)}.')
    else:
        print('CHECKPOINT 2: All active trading days were successfully mapped to sentiment values.')

    print('Merged dataframe shape:', merged.shape)
    print(merged.head(10).to_string(index=False))
    return merged


def plot_fear_vs_greed_pnl(merged: pd.DataFrame) -> None:
    """Plot average closed PnL for Fear vs Greed regimes."""
    grouped = (
        merged[merged['sentiment_group'].isin(['Fear', 'Greed'])]
        .groupby('sentiment_group')['Closed PnL']
        .mean()
        .reindex(['Fear', 'Greed'])
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grouped.index, grouped.values, color=['#1f77b4', '#ff7f0e'])
    plt.title('Average Closed PnL: Fear vs Greed')
    plt.ylabel('Average Closed PnL (USD)')
    plt.xlabel('Sentiment Regime')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_fear_vs_greed_pnl.png')
    plt.close()


def plot_avg_leverage_by_sentiment(merged: pd.DataFrame) -> None:
    """Plot average leverage by full sentiment classification."""
    if merged['leverage'].notna().sum() == 0:
        print('INFO: No leverage values available to plot. Plot will be skipped.')
        return

    leverage_stats = merged.groupby('classification')['leverage'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(leverage_stats.index, leverage_stats.values, color='#2ca02c')
    plt.title('Average Leverage by Sentiment Classification')
    plt.ylabel('Average Leverage')
    plt.xlabel('Sentiment Classification')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_avg_leverage_by_sentiment.png')
    plt.close()


def plot_win_rate_by_side_and_sentiment(merged: pd.DataFrame) -> None:
    """Plot trade win rate by side and sentiment group."""
    win_rate = (
        merged.groupby(['trade_direction', 'sentiment_group'])['Closed PnL']
        .apply(lambda series: (series > 0).mean() * 100)
        .unstack(fill_value=0)
    )

    ax = win_rate.plot(
        kind='bar',
        figsize=(10, 6),
        rot=0,
        cmap='tab10',
    )
    ax.set_title('Win Rate (%) by Trade Side and Sentiment Group')
    ax.set_ylabel('Win Rate (%)')
    ax.set_xlabel('Trade Direction')
    ax.legend(title='Sentiment Group')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_win_rate_by_side_and_sentiment.png')
    plt.close()


def print_eda_summaries(merged: pd.DataFrame) -> None:
    """Print summary statistics for the EDA visuals so results are available in text form."""
    print('\nEDA Summary:')
    fear_greed_stats = (
        merged[merged['sentiment_group'].isin(['Fear', 'Greed'])]
        .groupby('sentiment_group')['Closed PnL']
        .agg(['mean', 'median', 'count'])
        .rename(columns={'mean': 'avg_closed_pnl', 'median': 'median_closed_pnl', 'count': 'trade_count'})
    )
    print('\nAverage closed PnL by Fear/Greed:')
    print(fear_greed_stats.to_string())

    if merged['leverage'].notna().sum() > 0:
        leverage_stats = merged.groupby('classification')['leverage'].mean().sort_values(ascending=False)
        print('\nAverage leverage by sentiment classification:')
        print(leverage_stats.to_string())
    else:
        print('\nAverage leverage by sentiment classification: NO LEVERAGE DATA AVAILABLE')

    win_rate = (
        merged.groupby(['trade_direction', 'sentiment_group'])['Closed PnL']
        .apply(lambda series: (series > 0).mean() * 100)
        .unstack(fill_value=np.nan)
    )
    print('\nWin rate (%) by trade side and sentiment group:')
    print(win_rate.to_string())


def run_eda(merged: pd.DataFrame) -> None:
    """Run the exploratory analysis and save the plots."""
    print('\nCHECKPOINT 3: Generating EDA visualizations.')
    plot_fear_vs_greed_pnl(merged)
    plot_avg_leverage_by_sentiment(merged)
    plot_win_rate_by_side_and_sentiment(merged)
    print_eda_summaries(merged)
    print('CHECKPOINT 3: Plots saved as PNG files with titles and labels.')


def strategy_analysis(merged: pd.DataFrame) -> None:
    """Extract account-level patterns and build concrete trading strategies."""
    account_pnl = merged.groupby('Account')['Closed PnL'].sum().sort_values(ascending=False)
    top_accounts = account_pnl.head(5)
    print('\nTop accounts by total Closed PnL:')
    print(top_accounts.to_string())

    top_accounts_df = merged[merged['Account'].isin(top_accounts.index)].copy()
    extreme_mask = top_accounts_df['classification'].isin(['Extreme Fear', 'Extreme Greed'])
    extreme_summary = (
        top_accounts_df[extreme_mask]
        .groupby(['classification', 'trade_direction'])
        .agg(
            avg_leverage=('leverage', 'mean'),
            win_rate_pct=('Closed PnL', lambda x: (x > 0).mean() * 100),
            average_closed_pnl=('Closed PnL', 'mean'),
            trade_count=('Closed PnL', 'count'),
        )
        .reset_index()
    )

    print('\nTop-account behavior during Extreme Fear/Extreme Greed:')
    print(extreme_summary.to_string(index=False))

    overall_extreme = merged[merged['classification'].isin(['Extreme Fear', 'Extreme Greed'])]
    overall_summary = (
        overall_extreme
        .groupby(['classification', 'trade_direction'])
        .agg(
            overall_avg_leverage=('leverage', 'mean'),
            overall_win_rate_pct=('Closed PnL', lambda x: (x > 0).mean() * 100),
            overall_trade_count=('Closed PnL', 'count'),
        )
        .reset_index()
    )

    print('\nOverall trader behavior during Extreme Fear/Extreme Greed:')
    print(overall_summary.to_string(index=False))

    # Build data-backed strategy recommendations.
    strategies = []

    if overall_extreme.empty:
        print('WARNING: No Extreme Fear/Extreme Greed rows available after merge; strategy recommendations are limited.')
    else:
        fear_stats = overall_summary[overall_summary['classification'] == 'Extreme Fear']
        greed_stats = overall_summary[overall_summary['classification'] == 'Extreme Greed']

        if not fear_stats.empty:
            fear_long_win = fear_stats.loc[fear_stats['trade_direction'] == 'Long', 'overall_win_rate_pct'].squeeze() if 'Long' in fear_stats['trade_direction'].values else np.nan
            fear_short_win = fear_stats.loc[fear_stats['trade_direction'] == 'Short', 'overall_win_rate_pct'].squeeze() if 'Short' in fear_stats['trade_direction'].values else np.nan
            fear_leverage = fear_stats['overall_avg_leverage'].mean() if fear_stats['overall_avg_leverage'].notna().any() else np.nan
            if fear_long_win >= fear_short_win:
                strategies.append(
                    f'Implement a mean-reversion long bias during Extreme Fear when the average long win rate is {fear_long_win:.1f}%.'
                )
            else:
                strategies.append(
                    f'Under Extreme Fear, prioritize tactical short coverage and size reduction because shorts outperform longs with a win rate of {fear_short_win:.1f}%.'
                )
            if not np.isnan(fear_leverage):
                strategies.append(
                    f'Cap leverage near {fear_leverage:.2f}x in Extreme Fear to align with the average market behavior and reduce drawdown risk.'
                )

        if not greed_stats.empty:
            greed_long_win = greed_stats.loc[greed_stats['trade_direction'] == 'Long', 'overall_win_rate_pct'].squeeze() if 'Long' in greed_stats['trade_direction'].values else np.nan
            greed_short_win = greed_stats.loc[greed_stats['trade_direction'] == 'Short', 'overall_win_rate_pct'].squeeze() if 'Short' in greed_stats['trade_direction'].values else np.nan
            if greed_short_win >= greed_long_win:
                strategies.append(
                    f'In Extreme Greed, look for short or hedge opportunities when short win rate ({greed_short_win:.1f}%) is equal to or higher than long win rate ({greed_long_win:.1f}%).'
                )
            else:
                strategies.append(
                    f'When Extreme Greed persists, use profit-taking and partial de-risking because long positions have stronger win performance ({greed_long_win:.1f}%).'
                )

    strategies = strategies[:3]
    print('\nCHECKPOINT 4: Strategy recommendations:')
    for idx, strategy in enumerate(strategies, 1):
        print(f'{idx}. {strategy}')


def main() -> None:
    base_path = os.path.dirname(os.path.abspath(__file__))
    sentiment_path = os.path.join(base_path, 'fear_greed_index.csv')
    trades_path = os.path.join(base_path, 'historical_data.csv')

    print('CHECKPOINT 1: Loading datasets and normalizing dates.')
    sentiment_df = load_sentiment(sentiment_path)
    trades_df = load_trades(trades_path)

    print(f'Sentiment date range: {sentiment_df["date"].min().date()} to {sentiment_df["date"].max().date()}')
    print(f'Trade date range: {trades_df["trade_date"].min().date()} to {trades_df["trade_date"].max().date()}')

    if trades_df['trade_date'].isna().any():
        raise ValueError('Some trades failed to convert to a daily date. Please verify the `Timestamp IST` format in the trade CSV.')

    merged_df = merge_datasets(trades_df, sentiment_df)
    run_eda(merged_df)
    strategy_analysis(merged_df)


if __name__ == '__main__':
    main()
