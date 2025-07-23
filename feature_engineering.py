import pandas as pd
import numpy as np


def calculate_rsi(series, window=14):
    """
    计算相对强弱指数（RSI）
    Calculates the Relative Strength Index (RSI).
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate average gain and average loss using rolling mean
    # min_periods=1 ensures that it calculates from the first available data point
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate Relative Strength (RS)
    # Handle division by zero for avg_loss to avoid RuntimeWarning and get inf where avg_loss is 0
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    计算MACD指标
    Calculates the Moving Average Convergence Divergence (MACD) indicator.
    """
    # Manually implement MACD using Exponential Moving Averages (EMA)
    # Calculate 12-period EMA
    ema12 = series.ewm(span=fastperiod, adjust=False).mean()
    # Calculate 26-period EMA
    ema26 = series.ewm(span=slowperiod, adjust=False).mean()

    # Calculate MACD line
    macd = ema12 - ema26
    # Calculate Signal line (9-period EMA of MACD)
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    # Calculate MACD Histogram
    hist = macd - signal
    return macd, signal, hist


def add_features(df):
    """
    添加特征到数据框
    Adds various features to the DataFrame for stock prediction.
    """
    # Lag features: Previous day's close prices
    # These features capture the historical price information
    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        df[f'lag_{lag}_close'] = df['收盘'].shift(lag)

    # Moving Average features: Simple Moving Averages (SMA)
    # These features smooth out price data to identify trends
    ma_windows = [5, 10, 20, 50]
    for window in ma_windows:
        # Use min_periods=1 to allow calculation even with fewer data points at the start
        df[f'ma_{window}'] = df['收盘'].rolling(window=window, min_periods=1).mean()

    # Volatility (Standard Deviation)
    # Measures the dispersion of prices around the mean, indicating risk
    df['volatility_5'] = df['收盘'].rolling(5, min_periods=1).std()
    df['volatility_10'] = df['收盘'].rolling(10, min_periods=1).std()

    # Calculate RSI (Relative Strength Index)
    # Momentum oscillator that measures the speed and change of price movements
    df['rsi_14'] = calculate_rsi(df['收盘'], window=14)

    # Calculate MACD (Moving Average Convergence Divergence)
    # Trend-following momentum indicator that shows the relationship between two moving averages of a security’s price
    macd, signal, hist = calculate_macd(df['收盘'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist

    # Date features: Day of week, month, quarter
    # These features can capture cyclical patterns in stock prices
    # Ensure the index is DatetimeIndex before extracting date features
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
    else:
        print(
            "Warning: DataFrame index is not DatetimeIndex. Date features (day_of_week, month, quarter) will not be added.")

    # Drop original columns that are not available for future prediction
    # These columns ('开盘', '高', '低', '交易量', '涨跌幅') are not known for future dates,
    # so they cannot be used as features for iterative prediction.
    # '收盘' is kept as it's the base for derived features and will be iteratively updated.
    cols_to_drop_after_feature_creation = ['开盘', '高', '低', '交易量', '涨跌幅']
    for col in cols_to_drop_after_feature_creation:
        if col in df.columns:  # Check if column exists before dropping
            df = df.drop(columns=[col])

    # Target variable: Next day's closing price
    # This is what we want to predict
    df['next_day_close'] = df['收盘'].shift(-1)

    # Drop rows with NaN values that result from lag features and moving averages
    # These NaNs appear at the beginning of the DataFrame due to shifting and rolling window calculations
    # Also, the last row will have NaN for 'next_day_close'
    df = df.dropna()

    return df


def split_data(df, test_size=0.2):
    """
    按时间顺序划分训练集和测试集
    Splits the data into training and testing sets based on time order.
    """
    # Split data chronologically: e.g., first 80% for training, last 20% for testing
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def main():
    # File paths
    processed_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_processed.csv'
    output_features_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_index_with_features.csv'
    output_target_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_target_next_day_close.csv'  # New file for target

    # Read preprocessed data
    print("读取预处理后的数据...")
    # Ensure '日期' column is read as datetime and set as index
    # Added encoding='utf-8-sig' to match save_processed_data
    df = pd.read_csv(processed_file, index_col='日期', parse_dates=True, encoding='utf-8-sig')

    # Add features
    print("添加特征...")
    df_with_features = add_features(df)

    # --- START OF NEW LOGIC ---
    # Extract 'next_day_close' and its '日期' (index) into a separate DataFrame
    df_target = df_with_features[['next_day_close']].copy()

    # Save the target DataFrame to a separate CSV file
    df_target.to_csv(output_target_file, index=True, index_label='日期', encoding='utf-8-sig')
    print(f"目标变量 'next_day_close' 已保存至: {output_target_file}")

    # Drop 'next_day_close' from the features DataFrame before saving it
    df_with_features = df_with_features.drop(columns=['next_day_close'])
    # --- END OF NEW LOGIC ---

    # Split into training and testing sets (now operates on features without target)
    print("划分训练集和测试集...")
    # Note: split_data will now receive df_with_features which no longer has 'next_day_close'
    # This means train and test will only contain features.
    # The actual split for X and y will happen in train_xgboost.py's prepare_data
    train, test = split_data(df_with_features, test_size=0.2)

    # Save data with features (without 'next_day_close')
    # Explicitly save the DataFrame with headers and index label
    # The index (日期) will be the first column, labeled '日期'
    # All other columns will have their correct headers
    df_with_features.to_csv(output_features_file, index=True, index_label='日期', encoding='utf-8-sig')
    print(f"特征工程后的数据（不含目标变量）已保存至: {output_features_file}")

    # Print dataset information
    print(f"\n数据集信息:")
    print(f"总数据量: {len(df_with_features)}")
    print(f"训练集大小 (特征): {len(train)} ({len(train) / len(df_with_features) * 100:.1f}%)")
    print(f"测试集大小 (特征): {len(test)} ({len(test) / len(df_with_features) * 100:.1f}%)")
    print(f"训练集时间范围: {train.index.min().date()} 到 {train.index.max().date()}")
    print(f"测试集时间范围: {test.index.min().date()} 到 {test.index.max().date()}")

    print("\n特征工程完成!")


if __name__ == "__main__":
    main()
