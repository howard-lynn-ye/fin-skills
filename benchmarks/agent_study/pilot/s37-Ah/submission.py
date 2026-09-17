import pandas as pd
import numpy as np
from pathlib import Path


def build_positions(data_dir: str) -> "pandas.DataFrame":
    """Return dates x tickers portfolio weights.

    Row t holds the weights you are IN over session t: they may use information
    available strictly before the close of session t. Weights should be roughly
    dollar-neutral and sum of absolute values <= 1 per row. Missing = 0.
    """
    data_dir = Path(data_dir)

    # Load data
    close = pd.read_csv(data_dir / "close_quoted.csv", index_col=0)
    close.index = pd.to_datetime(close.index)

    corporate_actions = pd.read_csv(data_dir / "corporate_actions.csv")
    corporate_actions['date'] = pd.to_datetime(corporate_actions['date'])

    listings = pd.read_csv(data_dir / "listings.csv")
    listings['listing_date'] = pd.to_datetime(listings['listing_date'])
    listings['delisting_date'] = pd.to_datetime(listings['delisting_date'])

    # Adjust prices for splits
    for _, action in corporate_actions.iterrows():
        if action['kind'] == 'split':
            ticker = action['ticker']
            date = action['date']
            ratio = action['ratio']

            if ticker in close.columns:
                mask = close.index < date
                close.loc[mask, ticker] = close.loc[mask, ticker] / ratio

    # Handle listings: set values to NaN for delisted stocks
    for _, row in listings.iterrows():
        ticker = row['ticker']
        delisting_date = row['delisting_date']
        listing_date = row['listing_date']

        if ticker in close.columns:
            if pd.notna(listing_date):
                close.loc[close.index < listing_date, ticker] = np.nan

            if pd.notna(delisting_date):
                close.loc[close.index > delisting_date, ticker] = np.nan

    # Calculate returns and momentum
    returns = close.pct_change()

    # Calculate 60-day momentum (approximately 3 months of trading days)
    # This uses information from past 60 days to predict next day
    momentum_period = 60
    momentum = returns.rolling(window=momentum_period).sum()

    # Fill NaN with rolling sum of available data
    momentum = momentum.fillna(returns.rolling(window=momentum_period, min_periods=1).sum())

    # Shift momentum by 1 to ensure we're using past information only
    # momentum[t] uses returns from t-60 to t-1, which we can use for position at t
    momentum_for_positions = momentum.shift(1)

    # Standardize momentum across stocks on each day
    def standardize_signal(signal_df):
        standardized = pd.DataFrame(index=signal_df.index, columns=signal_df.columns, dtype=float)
        for date in signal_df.index:
            row = signal_df.loc[date]
            valid = row[~row.isna()]
            if len(valid) > 1:
                mean = valid.mean()
                std = valid.std()
                if std > 0:
                    standardized.loc[date] = (row - mean) / std
                else:
                    standardized.loc[date] = 0
            else:
                standardized.loc[date] = 0
        return standardized

    signal = standardize_signal(momentum_for_positions)

    # Convert signal to portfolio weights
    # Long top quartile, short bottom quartile
    positions = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    positions = positions.fillna(0.0)

    for date in signal.index:
        row = signal.loc[date]
        valid = row[~row.isna()]

        if len(valid) > 2:
            # Get quantiles
            q75 = valid.quantile(0.75)
            q25 = valid.quantile(0.25)

            # Long top quartile
            long_mask = row >= q75
            # Short bottom quartile
            short_mask = row <= q25

            weights = pd.Series(0.0, index=close.columns)
            n_long = long_mask.sum()
            n_short = short_mask.sum()

            if n_long > 0:
                weights[long_mask] = 1.0 / (2 * n_long)
            if n_short > 0:
                weights[short_mask] = -1.0 / (2 * n_short)

            positions.loc[date] = weights

    return positions
