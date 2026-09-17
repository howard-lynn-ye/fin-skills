import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def build_positions(data_dir: str) -> "pandas.DataFrame":
    """Return dates x tickers portfolio weights.

    Row t holds the weights you are IN over session t: they may use information
    available strictly before the close of session t. Weights should be roughly
    dollar-neutral and sum of absolute values <= 1 per row. Missing = 0.
    """

    # Read data files
    close_prices = pd.read_csv(os.path.join(data_dir, 'close_quoted.csv'), index_col=0)
    close_prices.index = pd.to_datetime(close_prices.index)

    volume = pd.read_csv(os.path.join(data_dir, 'volume.csv'), index_col=0)
    volume.index = pd.to_datetime(volume.index)

    listings = pd.read_csv(os.path.join(data_dir, 'listings.csv'))
    listings['listing_date'] = pd.to_datetime(listings['listing_date'])
    listings['delisting_date'] = pd.to_datetime(listings['delisting_date'], errors='coerce')

    corporate_actions = pd.read_csv(os.path.join(data_dir, 'corporate_actions.csv'))
    corporate_actions['date'] = pd.to_datetime(corporate_actions['date'])

    news_feed = pd.read_csv(os.path.join(data_dir, 'news_feed.csv'))
    news_feed['feed_ts'] = pd.to_datetime(news_feed['feed_ts'])

    llm_score = pd.read_csv(os.path.join(data_dir, 'llm_score.csv'), index_col=0)
    llm_score.index = pd.to_datetime(llm_score.index)

    fundamentals = pd.read_csv(os.path.join(data_dir, 'fundamentals.csv'))
    fundamentals['start'] = pd.to_datetime(fundamentals['start'])
    fundamentals['end'] = pd.to_datetime(fundamentals['end'])
    fundamentals['filed'] = pd.to_datetime(fundamentals['filed'])

    # Adjust prices for stock splits
    close_adj = close_prices.copy()
    for _, row in corporate_actions.iterrows():
        ticker = row['ticker']
        split_date = row['date']
        ratio = row['ratio']
        if ticker in close_adj.columns:
            # Adjust prices before split date
            close_adj.loc[close_adj.index < split_date, ticker] /= ratio

    # Define evaluation window
    eval_start = pd.Timestamp('2021-01-04')
    eval_end = pd.Timestamp('2023-12-29')

    # Generate trading dates
    eval_dates = close_prices.index[(close_prices.index >= eval_start) & (close_prices.index <= eval_end)]

    # Initialize weights dataframe - daily weights
    weights = pd.DataFrame(0.0, index=eval_dates, columns=close_prices.columns)

    # Rebalance frequency: weekly (every 5 days)
    rebalance_freq = 5
    last_rebalance_idx = -1
    last_weights = pd.Series(0.0, index=close_prices.columns)

    # Parameters for signals
    short_lookback = 5
    medium_lookback = 20
    long_lookback = 60

    for idx, date in enumerate(eval_dates):
        # Get data available strictly before this date
        prev_dates = close_prices.index[close_prices.index < date]
        if len(prev_dates) < long_lookback + 1:
            continue

        # Check if we should rebalance (weekly)
        should_rebalance = (idx - last_rebalance_idx) >= rebalance_freq

        if should_rebalance:
            # Get price data - adjusted for splits
            price_slice = close_adj.loc[prev_dates[-long_lookback-1]:prev_dates[-1]]

            # Calculate returns at different horizons
            returns_short = price_slice.iloc[-short_lookback:].pct_change().mean()
            returns_medium = price_slice.iloc[-medium_lookback:].pct_change().mean()
            returns_long = price_slice.iloc[-long_lookback:].pct_change().mean()

            # Calculate volatility
            vol_returns = price_slice.pct_change().dropna()
            volatility = vol_returns.std()

            # Calculate volume score
            vol_slice = volume.loc[prev_dates[-min(medium_lookback, len(prev_dates))]:prev_dates[-1]]
            avg_volume = vol_slice.mean()
            volume_score = (avg_volume > 0).astype(float)

            # Get news sentiment
            news_before = news_feed[news_feed['feed_ts'] < date]
            if len(news_before) > 0:
                recent_news = news_before[news_before['feed_ts'] >= date - timedelta(days=10)]
                news_dict = {}
                for ticker in close_prices.columns:
                    ticker_news = recent_news[recent_news['ticker'] == ticker]
                    if len(ticker_news) > 0:
                        scores = ticker_news['score'].replace('', np.nan).astype(float)
                        news_dict[ticker] = scores.mean()
                    else:
                        news_dict[ticker] = 0.0
                news_scores = pd.Series(news_dict)
            else:
                news_scores = pd.Series(0.0, index=close_prices.columns)

            # Get LLM scores (pre-trained, can use directly)
            if len(llm_score[llm_score.index < date]) > 0:
                llm_scores = llm_score.loc[llm_score.index < date].iloc[-1]
            else:
                llm_scores = pd.Series(0.0, index=close_prices.columns)

            # Check listing status for each ticker
            ticker_list = close_prices.columns
            active_stocks = []

            for ticker in ticker_list:
                ticker_info = listings[listings['ticker'] == ticker]
                if len(ticker_info) == 0:
                    continue

                listing_date = ticker_info.iloc[0]['listing_date']
                delisting_date = ticker_info.iloc[0]['delisting_date']

                # Check if stock was active on this date
                if date < listing_date:
                    continue
                if pd.notna(delisting_date) and date > delisting_date:
                    continue

                active_stocks.append(ticker)

            # Build composite signal using mean reversion
            signal = pd.Series(0.0, index=ticker_list)

            for ticker in active_stocks:
                # Recent short-term reversal (mean reversion)
                ret_short = returns_short.get(ticker, np.nan)
                if np.isnan(ret_short):
                    ret_short = 0.0

                # Medium-term momentum
                ret_medium = returns_medium.get(ticker, np.nan)
                if np.isnan(ret_medium):
                    ret_medium = 0.0

                # Long-term trend
                ret_long = returns_long.get(ticker, np.nan)
                if np.isnan(ret_long):
                    ret_long = 0.0

                # Volatility adjustment (prefer lower volatility)
                vol = volatility.get(ticker, np.nan)
                vol_score_adj = 0.0
                if not np.isnan(vol) and vol > 0:
                    vol_score_adj = -1.0 / (1.0 + vol)

                # News signal
                news = news_scores.get(ticker, 0.0)
                if pd.isna(news):
                    news = 0.0

                # LLM signal
                llm = llm_scores.get(ticker, 0.0)
                if pd.isna(llm):
                    llm = 0.0

                # Volume filter
                vol_filt = volume_score.get(ticker, 0.0)
                if vol_filt == 0:
                    continue

                # Composite signal: mean reversion on short term, momentum on medium/long
                composite = (-1.0 * ret_short +
                            0.5 * ret_medium +
                            0.3 * ret_long +
                            0.2 * vol_score_adj +
                            0.15 * news +
                            0.1 * llm)

                signal[ticker] = composite

            # Create positions based on signal ranking
            signal_values = signal[signal.index.isin(active_stocks)]
            signal_values = signal_values[~signal_values.isna()]

            if len(signal_values) > 3:
                # Rank signals
                ranked = signal_values.rank()

                # Create long/short positions based on top/bottom quartiles
                long_threshold = ranked.quantile(0.70)
                short_threshold = ranked.quantile(0.30)

                pos = pd.Series(0.0, index=ticker_list)

                for ticker in ranked.index:
                    r = ranked[ticker]
                    if r >= long_threshold:
                        pos[ticker] = 1.0
                    elif r <= short_threshold:
                        pos[ticker] = -1.0

                # Make dollar-neutral and normalize
                long_sum = (pos[pos > 0]).sum()
                short_sum = abs((pos[pos < 0]).sum())

                if long_sum > 0.01 and short_sum > 0.01:
                    # Balance long and short
                    target_notional = min(long_sum, short_sum)
                    pos_long = pos[pos > 0] / long_sum * target_notional
                    pos_short = pos[pos < 0] / short_sum * target_notional
                    pos = pd.Series(0.0, index=ticker_list)
                    pos.loc[pos_long.index] = pos_long
                    pos.loc[pos_short.index] = pos_short
                elif long_sum > 0.01:
                    pos[pos > 0] = pos[pos > 0] / (2.0 * long_sum)
                elif short_sum > 0.01:
                    pos[pos < 0] = pos[pos < 0] / (2.0 * short_sum)

                # Final leverage constraint
                gross = pos.abs().sum()
                if gross > 1.0:
                    pos = pos / gross

                last_weights = pos
                last_rebalance_idx = idx

        # Hold last computed positions
        weights.loc[date] = last_weights

    return weights
