"""
Cross-sectional equity signal using linear model trained on historical data.

Strategy:
1. Fit a linear regression model on training period (through 2020-12-31) to predict
   next-day returns using LLM scores and lagged news sentiment
2. Apply the fitted model to evaluation period to generate predictions
3. Create long-short positions based on predicted returns
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def build_positions(data_dir: str) -> "pandas.DataFrame":
    """Return dates x tickers portfolio weights.

    Row t holds the weights you are IN over session t: they may use information
    available strictly before the close of session t. Weights should be roughly
    dollar-neutral and sum of absolute values <= 1 per row. Missing = 0.
    """

    # Load data
    close = pd.read_csv(f'{data_dir}/close_quoted.csv', index_col=0)
    close.index = pd.to_datetime(close.index)

    llm_score = pd.read_csv(f'{data_dir}/llm_score.csv', index_col=0)
    llm_score.index = pd.to_datetime(llm_score.index)

    news = pd.read_csv(f'{data_dir}/news_feed.csv')
    news['feed_ts'] = pd.to_datetime(news['feed_ts'])

    corporate_actions = pd.read_csv(f'{data_dir}/corporate_actions.csv')
    corporate_actions['date'] = pd.to_datetime(corporate_actions['date'])

    # Adjust close prices for splits
    close_adj = close.copy()
    for _, action in corporate_actions.iterrows():
        ticker = action['ticker']
        split_date = action['date']
        ratio = action['ratio']

        if ticker in close_adj.columns:
            mask = close_adj.index < split_date
            close_adj.loc[mask, ticker] = close_adj.loc[mask, ticker] * ratio

    # Calculate returns
    returns = close_adj.pct_change()

    # Build news sentiment signal
    news_signal = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for _, row in news.iterrows():
        ticker = row['ticker']
        date = row['feed_ts']
        score = row['score']

        if pd.notna(score) and ticker in close.columns:
            if date in news_signal.index:
                existing = news_signal.loc[date, ticker]
                if pd.isna(existing):
                    news_signal.loc[date, ticker] = score
                else:
                    # Average if multiple news on same day
                    news_signal.loc[date, ticker] = 0.7 * existing + 0.3 * score

    # Lag news signal (news published today can be used for tomorrow's positions)
    news_signal_lag1 = news_signal.shift(1)

    # ===== TRAIN MODEL ON HISTORICAL DATA =====
    train_end = pd.Timestamp('2020-12-31')
    train_dates = llm_score.index[llm_score.index <= train_end]

    X_list = []
    y_list = []

    # Build training data: for each date, predict next day's return
    for i in range(len(train_dates) - 1):
        date = train_dates[i]
        next_date = train_dates[i + 1]

        # Features (strictly before close of date)
        llm_today = llm_score.loc[date].values
        news_today = news_signal_lag1.loc[date].values

        # Target (next day's actual return)
        ret_tomorrow = returns.loc[next_date].values

        # Remove NaNs and combine
        valid = ~(np.isnan(llm_today) | np.isnan(news_today) | np.isnan(ret_tomorrow))

        if valid.sum() > 10:
            X = np.column_stack([llm_today[valid], news_today[valid]])
            y = ret_tomorrow[valid]

            X_list.append(X)
            y_list.append(y)

    # Fit model
    X_train = np.vstack(X_list)
    y_train = np.hstack(y_list)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # ===== EVALUATE ON TEST PERIOD =====
    eval_start = pd.Timestamp('2021-01-04')
    eval_end = pd.Timestamp('2023-12-29')
    eval_dates = close.index[(close.index >= eval_start) & (close.index <= eval_end)]

    positions = pd.DataFrame(0.0, index=eval_dates, columns=close.columns)

    for date in eval_dates:
        # Get features for this date
        llm_today = llm_score.loc[date].values
        news_today = news_signal_lag1.loc[date].values

        # Identify valid (non-NaN) stocks
        valid = ~(np.isnan(llm_today) | np.isnan(news_today))

        if valid.sum() > 2:
            # Prepare input for model
            X_pred = np.column_stack([llm_today[valid], news_today[valid]])
            X_pred_scaled = scaler.transform(X_pred)

            # Get predictions
            pred = model.predict(X_pred_scaled)

            # Create full prediction vector
            pred_full = np.full_like(llm_today, np.nan)
            pred_full[valid] = pred

            # Convert predictions to positions
            # Z-score normalize predictions
            pred_valid = pred_full[~np.isnan(pred_full)]
            if len(pred_valid) > 2:
                pred_mean = pred_valid.mean()
                pred_std = pred_valid.std()

                if pred_std > 0:
                    z_scores = (pred_full - pred_mean) / pred_std
                else:
                    z_scores = pred_full - pred_mean

                # Create long-short positions
                # Convert to pandas Series for easier indexing
                z_series = pd.Series(z_scores, index=close.columns)

                longs = z_series[z_series > 0]
                shorts = z_series[z_series < 0]

                long_sum = longs.sum()
                short_sum = -shorts.sum()

                if long_sum > 0 and short_sum > 0:
                    # Dollar-neutral: longs sum to 0.5, shorts sum to -0.5
                    positions.loc[date, longs.index] = (longs / long_sum) * 0.5
                    # shorts are already negative, so divide by short_sum then scale by 0.5
                    positions.loc[date, shorts.index] = (shorts / short_sum) * 0.5
                elif long_sum > 0:
                    # Only longs: sum to 1.0
                    positions.loc[date, longs.index] = longs / long_sum
                elif short_sum > 0:
                    # Only shorts: sum to -1.0 (but normalized to 1.0 absolute value)
                    positions.loc[date, shorts.index] = shorts / short_sum

    # Fill missing with 0
    positions = positions.fillna(0)

    return positions
