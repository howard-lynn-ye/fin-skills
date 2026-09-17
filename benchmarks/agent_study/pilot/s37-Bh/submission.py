import pandas as pd
import numpy as np
import os


def build_positions(data_dir: str) -> "pandas.DataFrame":
    """Return dates x tickers portfolio weights.
    
    Row t holds the weights you are IN over session t: they may use information
    available strictly before the close of session t. Weights should be roughly
    dollar-neutral and sum of absolute values <= 1 per row. Missing = 0.
    """
    
    # Load data files
    close_path = os.path.join(data_dir, 'close_quoted.csv')
    volume_path = os.path.join(data_dir, 'volume.csv')
    listings_path = os.path.join(data_dir, 'listings.csv')
    
    # Load price data
    df_close = pd.read_csv(close_path, index_col=0)
    df_close.index = pd.to_datetime(df_close.index)
    df_close.index.name = 'date'
    
    # Load volume
    df_volume = pd.read_csv(volume_path, index_col=0)
    df_volume.index = pd.to_datetime(df_volume.index)
    df_volume.index.name = 'date'
    
    # Load listings
    df_listings = pd.read_csv(listings_path)
    df_listings['listing_date'] = pd.to_datetime(df_listings['listing_date'])
    df_listings['delisting_date'] = pd.to_datetime(df_listings['delisting_date'])
    
    # Training period: through 2020-12-31
    # Evaluation period: 2021-01-04 to 2023-12-29
    eval_start = pd.Timestamp('2021-01-04')
    eval_end = pd.Timestamp('2023-12-29')
    
    # Get dates for evaluation period
    eval_dates = df_close.loc[eval_start:eval_end].index
    
    # Compute returns
    returns_30 = df_close.pct_change(30)
    returns_60 = df_close.pct_change(60)
    
    # Initialize weights dataframe with 0s
    weights = pd.DataFrame(0.0, index=eval_dates, columns=df_close.columns)
    
    # Rebalance every 5 trading days
    rebalance_interval = 5
    
    # Build positions for evaluation period
    last_weights = pd.Series(0.0, index=df_close.columns)
    last_rebalance_idx = 0
    
    for idx, date in enumerate(eval_dates):
        # Rebalance every 5 trading days
        if idx - last_rebalance_idx >= rebalance_interval or idx == 0:
            last_rebalance_idx = idx
            
            # Get the universe of active stocks at this date
            active_mask = (
                (df_listings['listing_date'] <= date) &
                ((df_listings['delisting_date'].isna()) | (df_listings['delisting_date'] > date))
            )
            active_tickers = df_listings.loc[active_mask, 'ticker'].values
            
            # Check liquidity
            if date in df_volume.index:
                lookback_start = date - pd.Timedelta(days=20)
                recent_volume = df_volume.loc[lookback_start:date]
                if len(recent_volume) > 0:
                    avg_volume = recent_volume.mean()
                    liquid_threshold = avg_volume.quantile(0.25)
                    liquid_mask = avg_volume > liquid_threshold
                    liquid_tickers = list(avg_volume[liquid_mask].index.intersection(active_tickers))
                else:
                    liquid_tickers = list(active_tickers)
            else:
                liquid_tickers = list(active_tickers)
            
            if len(liquid_tickers) == 0:
                weights.loc[date:] = last_weights
                continue
            
            # Get momentum signals (longer-term)
            signals = {}
            for ticker in liquid_tickers:
                if ticker in returns_30.columns and date in returns_30.index:
                    ret_30 = returns_30.loc[date, ticker]
                else:
                    ret_30 = 0.0
                
                if ticker in returns_60.columns and date in returns_60.index:
                    ret_60 = returns_60.loc[date, ticker]
                else:
                    ret_60 = 0.0
                
                # Use 30-day and 60-day momentum
                signal = 0.6 * ret_30 + 0.4 * ret_60
                signals[ticker] = signal
            
            if len(signals) == 0:
                weights.loc[date:] = last_weights
                continue
            
            signals_series = pd.Series(signals)
            
            # Z-score normalize
            signal_mean = signals_series.mean()
            signal_std = signals_series.std()
            if signal_std > 1e-8:
                z_scores = (signals_series - signal_mean) / signal_std
            else:
                z_scores = signals_series * 0.0
            
            # Create market-neutral weights
            combined_weights = pd.Series(0.0, index=df_close.columns)

            # Split into long and short
            long_positions = z_scores[z_scores > 0]
            short_positions = z_scores[z_scores < 0]

            # Normalize long positions to sum to 0.5
            if len(long_positions) > 0:
                long_sum = long_positions.sum()
                if long_sum > 1e-8:
                    long_weights = long_positions / long_sum * 0.5
                    combined_weights.loc[long_weights.index] = long_weights.values

            # Normalize short positions to sum to -0.5
            if len(short_positions) > 0:
                # short_positions are negative z-scores
                # We want their weights to sum to -0.5
                short_sum = short_positions.sum()  # This will be negative
                if short_sum < -1e-8:
                    # Divide by sum (negative) and multiply by -0.5 to get negative weights
                    short_weights = short_positions / short_sum * (-0.5)
                    combined_weights.loc[short_weights.index] = short_weights.values

            last_weights = combined_weights
        
        weights.loc[date] = last_weights
    
    return weights
