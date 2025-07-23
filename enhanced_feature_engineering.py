import pandas as pd
import numpy as np
import os
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)


def load_processed_data(file_path):
    """
    Loads the preprocessed data from a CSV file.
    从CSV文件加载预处理后的数据。
    """
    print(f"Loading processed data from: {file_path}...")
    try:
        df = pd.read_csv(file_path, index_col='日期', parse_dates=True, encoding='utf-8-sig')
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the path is correct.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def calculate_rsi(series, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    计算相对强弱指数（RSI）。
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD) indicator.
    计算MACD指标。
    """
    ema12 = series.ewm(span=fastperiod, adjust=False, min_periods=1).mean()
    ema26 = series.ewm(span=slowperiod, adjust=False, min_periods=1).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=signalperiod, adjust=False, min_periods=1).mean()
    hist = macd - signal
    return macd, signal, hist


def calculate_adx(df, window=14):
    """
    Calculates the Average Directional Index (ADX).
    计算平均方向指数（ADX）。
    Requires '高', '低', '收盘' columns.
    """
    if not all(col in df.columns for col in ['高', '低', '收盘']):
        print("Warning: Missing '高', '低', or '收盘' columns for ADX calculation. Skipping ADX.")
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    high = df['高']
    low = df['低']
    close = df['收盘']

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # True Range (TR)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # Directional Movement (DM)
    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Ensure correct comparison for DM
    true_plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    true_minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)

    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/window)
    # Use min_periods=1 for initial values
    atr = tr.ewm(span=window, adjust=False, min_periods=1).mean()
    plus_di = 100 * (pd.Series(true_plus_dm).ewm(span=window, adjust=False, min_periods=1).mean() / atr)
    minus_di = 100 * (pd.Series(true_minus_dm).ewm(span=window, adjust=False, min_periods=1).mean() / atr)

    # Directional Index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero

    # Average Directional Index (ADX)
    adx = dx.ewm(span=window, adjust=False, min_periods=1).mean()

    return plus_di, minus_di, adx


def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    Calculates Bollinger Bands.
    计算布林带。
    """
    middle_band = series.rolling(window=window, min_periods=1).mean()
    std_dev = series.rolling(window=window, min_periods=1).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band


def calculate_atr(df, window=14):
    """
    Calculates Average True Range (ATR).
    计算平均真实波幅（ATR）。
    Requires '高', '低', '收盘' columns.
    """
    if not all(col in df.columns for col in ['高', '低', '收盘']):
        print("Warning: Missing '高', '低', or '收盘' columns for ATR calculation. Skipping ATR.")
        return pd.Series(np.nan, index=df.index)

    high = df['高']
    low = df['低']
    close = df['收盘']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = true_range.ewm(span=window, adjust=False, min_periods=1).mean()  # Use EMA for ATR smoothing
    return atr


def calculate_obv(df):
    """
    Calculates On-Balance Volume (OBV).
    计算能量潮（OBV）。
    Requires '收盘', '交易量' columns.
    """
    if not all(col in df.columns for col in ['收盘', '交易量']):
        print("Warning: Missing '收盘' or '交易量' columns for OBV calculation. Skipping OBV.")
        return pd.Series(np.nan, index=df.index)

    close = df['收盘']
    volume = df['交易量']

    obv = pd.Series(0, index=df.index)
    obv.iloc[0] = volume.iloc[0]  # Initialize OBV with first volume

    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


def add_enhanced_features(df):
    """
    Adds enhanced features for trend and momentum analysis.
    添加用于趋势和动量分析的增强特征。
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure '收盘', '开盘', '高', '低', '交易量', '涨跌幅' are numeric
    # '涨跌幅' is crucial for new daily_return and next_day_direction
    for col in ['收盘', '开盘', '高', '低', '交易量', '涨跌幅']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean() if df[col].mean() is not np.nan else 0)

    # 1. 动量特征 (Momentum Features)
    # Modified: 'daily_return' based on previous day's '涨跌幅' * 100
    df['daily_return'] = df['涨跌幅'].shift(1) * 100
    df['cum_return_5d'] = df['收盘'].pct_change(periods=5)  # 5日累计涨跌幅
    df['cum_return_20d'] = df['收盘'].pct_change(periods=20)  # 20日累计涨跌幅
    df['rsi_14'] = calculate_rsi(df['收盘'], window=14)  # 相对强弱指数

    # 2. 趋势强度特征 (Trend Strength Features)
    # Moving Averages (for position relationship)
    df['ma_5'] = df['收盘'].rolling(window=5, min_periods=1).mean()
    df['ma_20'] = df['收盘'].rolling(window=20, min_periods=1).mean()
    df['ma_60'] = df['收盘'].rolling(window=60, min_periods=1).mean()

    # Closing Price vs. MA Position
    df['close_above_ma5'] = (df['收盘'] > df['ma_5']).astype(int)
    df['close_above_ma20'] = (df['收盘'] > df['ma_20']).astype(int)
    df['close_above_ma60'] = (df['收盘'] > df['ma_60']).astype(int)

    # Average Directional Index (ADX)
    plus_di, minus_di, adx = calculate_adx(df.copy(), window=14)  # Pass a copy to avoid modifying original df
    df['plus_di_14'] = plus_di
    df['minus_di_14'] = minus_di
    df['adx_14'] = adx

    # 3. 波动性特征 (Volatility Features)
    # Bollinger Bands
    df['bb_upper_20_2'], df['bb_middle_20_2'], df['bb_lower_20_2'] = calculate_bollinger_bands(df['收盘'], window=20,
                                                                                               num_std=2)
    df['is_above_bb_upper'] = (df['收盘'] > df['bb_upper_20_2']).astype(int)
    df['is_below_bb_lower'] = (df['收盘'] < df['bb_lower_20_2']).astype(int)

    # Average True Range (ATR)
    df['atr_14'] = calculate_atr(df.copy(), window=14)  # Pass a copy

    # 4. 成交量特征 (Volume Features)
    df['ma_volume_5'] = df['交易量'].rolling(window=5, min_periods=1).mean()
    df['volume_up_strong'] = ((df['daily_return'] > 0) & (df['交易量'] > df['ma_volume_5'])).astype(
        int)  # 当日上涨且成交量 > 5日平均
    df['volume_down_strong'] = ((df['daily_return'] < 0) & (df['交易量'] > df['ma_volume_5'])).astype(
        int)  # 当日下跌且成交量 > 5日平均
    df['obv'] = calculate_obv(df.copy())  # 能量潮

    # 5. 形态特征 (Pattern Features) - Specific requests
    # Consecutive Up/Down Days
    is_up = df['收盘'] > df['收盘'].shift(1)
    is_down = df['收盘'] < df['收盘'].shift(1)

    # Calculate consecutive up/down counts
    df['consecutive_up_days'] = is_up.astype(int).groupby((is_up != is_up.shift()).cumsum()).cumsum() * is_up.astype(
        int)
    df['consecutive_down_days'] = is_down.astype(int).groupby(
        (is_down != is_down.shift()).cumsum()).cumsum() * is_down.astype(int)

    # Binary indicators for consecutive days
    for i in range(2, 8):  # 2 to 7 days
        df[f'consecutive_up_{i}d'] = (df['consecutive_up_days'] >= i).astype(int)
        df[f'consecutive_down_{i}d'] = (df['consecutive_down_days'] >= i).astype(int)

    # Price vs. MA Slope Ratio (using 5-day period for slope)
    # Price slope
    df['price_slope_5d'] = (df['收盘'] - df['收盘'].shift(5)) / 5
    # MA slope (using 20-day MA for its slope)
    df['ma20_slope_5d'] = (df['ma_20'] - df['ma_20'].shift(5)) / 5
    # Ratio, handle division by zero
    df['price_ma20_slope_ratio'] = np.where(df['ma20_slope_5d'] == 0, np.nan,
                                            df['price_slope_5d'] / df['ma20_slope_5d'])

    # Specific features requested at the end:
    # 1. 涨幅 (Daily Return) - Already done: 'daily_return'
    # 2. 前一天收盘价格是否在5日均线之上
    df['prev_close_above_ma5'] = (df['收盘'].shift(1) > df['ma_5'].shift(1)).astype(int)
    # 3. 是否连涨2、3、4、5、6、7天 - Already done: 'consecutive_up_Xd'
    # 4. 是否连跌2、3、4、5、6、7天 - Already done: 'consecutive_down_Xd'

    # Remove original columns that should not be used as features for prediction
    # These columns are usually not known for the prediction day.
    # '收盘' is kept until target calculation, then handled in main for final X_enhanced.
    # '涨跌幅' is also kept for next_day_direction calculation, then dropped.
    cols_to_drop_after_feature_creation = ['开盘', '高', '低', '交易量']  # Keep '收盘' and '涨跌幅' for now
    for col in cols_to_drop_after_feature_creation:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Define the target variable for classification: next day's direction (1 for up, 0 for down)
    # Modified: 'next_day_direction' based on next day's '涨跌幅'
    df['next_day_direction'] = (df['涨跌幅'] > 0).astype(int)

    # Fill NaNs for all feature columns first
    # This is crucial for features that have leading NaNs due to rolling/shifting
    # Apply ffill then bfill for time-series context
    df = df.fillna(method='ffill').fillna(method='bfill')

    # For any remaining NaNs (e.g., if the entire column was NaN, or if ffill/bfill couldn't fill everything)
    # Fill with 0. This is a fallback for columns that might still have NaNs.
    df = df.fillna(0)

    # Drop rows where 'next_day_direction' is NaN (only the very last row usually)
    df = df.dropna(subset=['next_day_direction'])

    # '涨跌幅' is an original column and is used to derive 'next_day_direction'.
    # It should be dropped from the final features (X_enhanced) to avoid data leakage.
    if '涨跌幅' in df.columns:
        df = df.drop(columns=['涨跌幅'])

    # 'next_day_close' is no longer needed as an intermediate for direction, so ensure it's not present
    if 'next_day_close' in df.columns:
        df = df.drop(columns=['next_day_close'])

    # --- NEW: Filter data from '2006-07-01' onwards AFTER all feature calculations ---
    initial_rows_before_filter = len(df)
    df = df[df.index >= '2006-07-01']
    print(
        f"Filtered data from 2006-07-01 onwards. Original rows before final filter: {initial_rows_before_filter}, Filtered rows: {len(df)}")
    if df.empty:
        print(
            "After filtering, DataFrame is empty. This might indicate insufficient data after the start date or excessive NaN removal. Exiting.")
        return pd.DataFrame()
    # --- END NEW FILTERING ---

    return df


def main():
    # File paths
    processed_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_processed.csv'
    output_features_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100_enhanced_features.csv'
    output_target_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100_direction_target.csv'

    # 1. Load processed data
    print("Loading preprocessed data...")
    df = load_processed_data(processed_file)
    if df.empty:
        print("Data loading failed or file is empty, exiting.")
        return

    # --- REMOVED: Early filtering from '2006-07-01' onwards here ---
    # This filtering is now moved inside add_enhanced_features to ensure full history for calculations.

    # 2. Add enhanced features
    print("\nAdding enhanced features...")
    # Pass a copy to avoid modifying the original df loaded from CSV
    df_with_enhanced_features = add_enhanced_features(df.copy())
    if df_with_enhanced_features.empty:
        print("Enhanced feature engineering failed or resulted in empty DataFrame after filtering, exiting.")
        return

    # Separate features (X) and target (y)
    X_enhanced = df_with_enhanced_features.drop(columns=['next_day_direction'])
    y_direction = df_with_enhanced_features['next_day_direction']

    # --- Drop specific columns from X_enhanced before saving ---
    # '收盘', 'ma_5', 'ma_20', 'ma_60' are explicitly excluded from the final feature set.
    cols_to_exclude_from_features = ['收盘', 'ma_5', 'ma_20', 'ma_60']
    for col in cols_to_exclude_from_features:
        if col in X_enhanced.columns:
            X_enhanced = X_enhanced.drop(columns=[col])
    print(f"Excluded columns from final features: {cols_to_exclude_from_features}")
    # --- END EXCLUSION ---

    # 3. Save enhanced features and target
    print(f"\nSaving enhanced features to: {output_features_file}")
    X_enhanced.to_csv(output_features_file, index=True, index_label='日期', encoding='utf-8-sig')

    print(f"Saving direction target to: {output_target_file}")
    y_direction.to_csv(output_target_file, index=True, index_label='日期', encoding='utf-8-sig',
                       header=['next_day_direction'])

    print(f"\nEnhanced feature engineering completed!")
    print(f"Enhanced Features shape: {X_enhanced.shape}")
    print(f"Direction Target shape: {y_direction.shape}")
    print(f"First 5 rows of Enhanced Features:\n{X_enhanced.head()}")
    print(f"First 5 rows of Direction Target:\n{y_direction.head()}")


if __name__ == "__main__":
    main()
