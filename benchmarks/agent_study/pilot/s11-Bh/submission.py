"""Cross-sectional equity signal strategy combining LLM and news sentiment."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def build_positions(data_dir: str) -> "pandas.DataFrame":
    """Return dates x tickers portfolio weights.

    Row t holds the weights you are IN over session t: they may use information
    available strictly before the close of session t. Weights should be roughly
    dollar-neutral and sum of absolute values <= 1 per row. Missing = 0.
    """

    # Read all data files
    close = pd.read_csv(f"{data_dir}/close_quoted.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(f"{data_dir}/volume.csv", index_col=0, parse_dates=True)
    llm_score = pd.read_csv(f"{data_dir}/llm_score.csv", index_col=0, parse_dates=True)
    news_feed = pd.read_csv(f"{data_dir}/news_feed.csv", parse_dates=["feed_ts"])
    listings = pd.read_csv(f"{data_dir}/listings.csv", parse_dates=["listing_date", "delisting_date"])

    # Define evaluation window
    eval_start = pd.Timestamp("2021-01-04")
    eval_end = pd.Timestamp("2023-12-29")

    # Filter data to evaluation period
    eval_close = close[(close.index >= eval_start) & (close.index <= eval_end)]
    eval_volume = volume[(volume.index >= eval_start) & (volume.index <= eval_end)]
    eval_llm = llm_score[(llm_score.index >= eval_start) & (llm_score.index <= eval_end)]

    # Process news feed: pivot to date x ticker format
    news_pivot = news_feed.groupby(['feed_ts', 'ticker'])['score'].first().unstack(fill_value=np.nan)
    news_pivot.index.name = None
    news_eval = news_pivot[(news_pivot.index >= eval_start) & (news_pivot.index <= eval_end)]

    # Calculate price momentum (5-day past returns as signal component)
    returns = close.pct_change()
    momentum = returns.rolling(window=5).mean()
    momentum_eval = momentum[(momentum.index >= eval_start) & (momentum.index <= eval_end)]

    # Create listings date filter
    listings_dict = {}
    for _, row in listings.iterrows():
        ticker = row['ticker']
        list_date = row['listing_date']
        delist_date = row['delisting_date']
        listings_dict[ticker] = (list_date, delist_date)

    # Build positions
    positions = pd.DataFrame(0.0, index=eval_close.index, columns=eval_close.columns)

    for date in eval_close.index:
        # Filter for valid stocks on this date
        valid_mask = pd.Series(True, index=eval_close.columns)

        # Filter by listing status
        for ticker in eval_close.columns:
            if ticker in listings_dict:
                list_date, delist_date = listings_dict[ticker]
                if pd.notna(list_date) and list_date > date:
                    valid_mask[ticker] = False
                if pd.notna(delist_date) and delist_date <= date:
                    valid_mask[ticker] = False

        # Filter by volume (avoid illiquid names)
        if date in eval_volume.index:
            vol_row = eval_volume.loc[date]
            valid_mask = valid_mask & (vol_row > 100)

        # Filter by price
        if date in eval_close.index:
            price_row = eval_close.loc[date]
            valid_mask = valid_mask & (price_row > 0)

        # Combine signals for this date
        combined_signal = pd.Series(0.0, index=eval_close.columns)

        # Add LLM score (primary signal)
        if date in eval_llm.index:
            llm_sig = eval_llm.loc[date].copy()
            llm_sig[~valid_mask] = np.nan
            combined_signal = combined_signal + llm_sig.fillna(0) * 0.7  # 70% weight

        # Add news sentiment (secondary signal)
        if date in news_eval.index:
            news_sig = news_eval.loc[date].copy()
            news_sig[~valid_mask] = np.nan
            combined_signal = combined_signal + news_sig.fillna(0) * 0.3  # 30% weight

        # Filter by valid signals
        combined_signal[~valid_mask] = np.nan

        # Only proceed if we have enough valid signals
        valid_signals = combined_signal.dropna()
        if len(valid_signals) >= 5:
            # Standardize signal cross-sectionally
            mean_sig = valid_signals.mean()
            std_sig = valid_signals.std()

            if std_sig > 1e-6:
                sig_normalized = (combined_signal - mean_sig) / std_sig
            else:
                sig_normalized = combined_signal - mean_sig

            # Create long-short portfolio using top/bottom quintiles
            n_valid = len(valid_signals)
            n_per_side = max(1, n_valid // 5)

            # Get tickers in top and bottom quintiles
            valid_sig = sig_normalized.dropna()
            top_q = valid_sig.nlargest(n_per_side)
            bottom_q = valid_sig.nsmallest(n_per_side)

            # Assign equal weights within each portfolio
            if len(top_q) > 0:
                for ticker in top_q.index:
                    positions.loc[date, ticker] = 1.0 / len(top_q)

            if len(bottom_q) > 0:
                for ticker in bottom_q.index:
                    positions.loc[date, ticker] = -1.0 / len(bottom_q)

    # Ensure no NaN values (replace with 0)
    positions = positions.fillna(0.0)

    return positions
