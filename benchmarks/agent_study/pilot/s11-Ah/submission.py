import pandas as pd
import numpy as np


def build_positions(data_dir: str) -> "pandas.DataFrame":
    """
    Build a cross-sectional equity strategy using momentum and LLM scores.

    Strategy:
    - Uses recent price momentum (20-day returns) as the primary signal
    - Combines with LLM sentiment scores for additional alpha
    - Ranks stocks and constructs a dollar-neutral long/short portfolio
    - Rebalances monthly to control turnover
    """

    # Load data
    close = pd.read_csv(f'{data_dir}/close_quoted.csv', index_col=0)
    volume = pd.read_csv(f'{data_dir}/volume.csv', index_col=0)
    llm = pd.read_csv(f'{data_dir}/llm_score.csv', index_col=0)
    listings = pd.read_csv(f'{data_dir}/listings.csv')
    corp_actions = pd.read_csv(f'{data_dir}/corporate_actions.csv')

    # Convert to datetime
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    llm.index = pd.to_datetime(llm.index)
    listings['listing_date'] = pd.to_datetime(listings['listing_date'])
    listings['delisting_date'] = pd.to_datetime(listings['delisting_date'])
    corp_actions['date'] = pd.to_datetime(corp_actions['date'])

    # Handle corporate actions (splits)
    close_adj = close.copy()
    for _, row in corp_actions.iterrows():
        ticker = row['ticker']
        date = row['date']
        ratio = row['ratio']
        if ticker in close_adj.columns:
            mask = close_adj.index < date
            if mask.any():
                close_adj.loc[mask, ticker] = close_adj.loc[mask, ticker] / ratio

    # Compute returns
    returns = close_adj.pct_change()

    # Momentum signal (20-day returns)
    momentum = returns.rolling(20).sum()

    # Volume signal (for filtering)
    vol_20d = volume.rolling(20).mean()

    # LLM signal
    llm_signal = llm.fillna(0)

    # Combine signals
    # Normalize signals using rolling statistics
    lookback = 60

    momentum_norm = momentum.copy()
    llm_norm = llm_signal.copy()
    vol_norm = vol_20d.copy()

    for col in momentum.columns:
        # Normalize momentum
        mom_mean = momentum[col].rolling(lookback).mean()
        mom_std = momentum[col].rolling(lookback).std()
        momentum_norm[col] = (momentum[col] - mom_mean) / (mom_std + 1e-6)

        # Normalize LLM
        llm_mean = llm_signal[col].rolling(lookback).mean()
        llm_std = llm_signal[col].rolling(lookback).std()
        llm_norm[col] = (llm_signal[col] - llm_mean) / (llm_std + 1e-6)

        # Normalize volume (for filtering - not in ranking)
        vol_mean = vol_20d[col].rolling(lookback).mean()
        vol_std = vol_20d[col].rolling(lookback).std()
        vol_norm[col] = (vol_20d[col] - vol_mean) / (vol_std + 1e-6)

    # Combine signals - momentum is primary signal, LLM provides secondary alpha
    combined_signal = 0.7 * momentum_norm + 0.3 * llm_norm
    combined_signal = combined_signal.fillna(0)

    # Create listing/delisting mask
    listing_mask = pd.DataFrame(True, index=close.index, columns=close.columns)

    for ticker in close.columns:
        ticker_listings = listings[listings['ticker'] == ticker]
        if len(ticker_listings) > 0:
            listing_info = ticker_listings.iloc[0]
            list_date = listing_info['listing_date']
            delist_date = listing_info['delisting_date']

            if pd.notna(list_date):
                listing_mask.loc[close.index < list_date, ticker] = False
            if pd.notna(delist_date):
                listing_mask.loc[close.index >= delist_date, ticker] = False

    # Apply listing mask to combined signal
    combined_signal = combined_signal * listing_mask

    # Convert signal to portfolio weights
    # Rebalance monthly to reduce turnover
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # Identify rebalance dates (first trading day of each month)
    rebalance_days = []
    prev_month = None
    for date in close.index:
        month = date.to_period('M')
        if month != prev_month:
            rebalance_days.append(date)
            prev_month = month

    # Compute weights only on rebalance days
    rebalance_weights = {}
    for rebalance_date in rebalance_days:
        idx = close.index.get_loc(rebalance_date)
        if idx < lookback:
            rebalance_weights[rebalance_date] = pd.Series(0.0, index=close.columns)
            continue

        signal_row = combined_signal.loc[rebalance_date].copy()

        # Get valid stocks
        valid_mask = listing_mask.loc[rebalance_date]
        valid_signals = signal_row[valid_mask].copy()

        n_stocks = (valid_mask).sum()
        if n_stocks < 10:
            rebalance_weights[rebalance_date] = pd.Series(0.0, index=close.columns)
            continue

        # Top and bottom 20%
        n_top = max(2, n_stocks // 5)
        n_bottom = max(2, n_stocks // 5)

        # Get top and bottom performers
        sorted_signals = valid_signals.dropna().sort_values(ascending=False)

        if len(sorted_signals) < n_top + n_bottom:
            rebalance_weights[rebalance_date] = pd.Series(0.0, index=close.columns)
            continue

        w = pd.Series(0.0, index=close.columns)
        long_tickers = sorted_signals.head(n_top).index
        short_tickers = sorted_signals.tail(n_bottom).index

        w[long_tickers] = 1.0 / n_top
        w[short_tickers] = -1.0 / n_bottom

        rebalance_weights[rebalance_date] = w

    # Forward-fill weights between rebalance dates
    for i, date in enumerate(close.index):
        # Find the most recent rebalance date on or before this date
        recent_rebalances = [d for d in rebalance_days if d <= date]
        if recent_rebalances:
            recent_date = recent_rebalances[-1]
            if recent_date in rebalance_weights:
                weights.loc[date] = rebalance_weights[recent_date]

    return weights


if __name__ == '__main__':
    # Test the function
    weights = build_positions('data')
    print(f"Portfolio weights shape: {weights.shape}")
    print(f"Date range: {weights.index.min()} to {weights.index.max()}")
    print(f"\nSum of absolute weights per row stats:")
    print((weights.abs().sum(axis=1)).describe())
