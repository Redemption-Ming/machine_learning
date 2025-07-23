import pandas as pd
import numpy as np


# 1. 数据加载
def load_data(file_path):
    """
    加载CSV文件，处理特殊字符和命名问题
    """
    try:
        # 读取CSV文件，处理可能的BOM字符
        df = pd.read_csv(file_path, encoding='utf-8')

        # 处理可能的BOM字符问题
        df.columns = df.columns.str.replace('\ufeff', '')

        return df
    except UnicodeDecodeError:
        # 如果utf-8解码失败，尝试ISO-8859-1
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return pd.DataFrame()  # 返回空DataFrame避免后续错误


# 2. 数据探索函数
def explore_data(df):
    """
    探索性数据分析
    """
    if df.empty:
        print("DataFrame为空，无法进行探索。")
        return

    print("=" * 50)
    print("数据概览:")
    print(f"数据集形状: {df.shape}")

    # 检查'日期'列是否存在且为datetime类型
    if '日期' in df.columns and pd.api.types.is_datetime64_any_dtype(df['日期']):
        print(f"时间范围: {df['日期'].min()} 到 {df['日期'].max()}")
    else:
        print("警告: '日期'列不存在或不是日期时间类型，无法显示时间范围。")


    print("\n前5行数据:")
    print(df.head())

    print("\n基本信息:")
    print(df.info())

    print("\n统计摘要:")
    print(df.describe())

    print("\n缺失值检查:")
    print(df.isnull().sum())

    print("\n唯一值计数:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} 个唯一值")


# 3. 数据预处理函数
def preprocess_data(df):
    """
    执行数据预处理
    """
    if df.empty:
        return pd.DataFrame()

    # 处理日期格式和排序
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce') # errors='coerce' 将无法解析的日期转换为NaT
        df = df.dropna(subset=['日期']) # 删除日期为NaT的行
        df = df.sort_values('日期', ascending=True).reset_index(drop=True)
    else:
        print("错误: CSV文件中未找到'日期'列。请确保日期列名为'日期'。")
        return pd.DataFrame() # 如果没有日期列，则无法进行时间序列分析

    # 处理带逗号和引号的数值字段
    numeric_cols_to_clean = ['收盘', '开盘', '高', '低']
    for col in numeric_cols_to_clean:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace('"', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce') # errors='coerce' 将无法转换的转换为NaN

    # 处理交易量 (可能有M/B后缀)
    if '交易量' in df.columns:
        # 先将交易量列转换为字符串，以便进行字符串操作
        df['交易量_processed'] = df['交易量'].astype(str).str.strip().str.upper()

        # 使用正则表达式替换 'B' 和 'M' 并进行乘法
        # 使用 apply 函数处理每个值，更健壮
        def convert_volume(volume_str):
            if pd.isna(volume_str) or volume_str == 'NAN': # Handle NaN strings
                return np.nan
            if 'B' in volume_str:
                return float(volume_str.replace('B', '')) * 1e9
            elif 'M' in volume_str:
                return float(volume_str.replace('M', '')) * 1e6
            else:
                try:
                    return float(volume_str.replace(',', '')) # Remove commas for pure numbers
                except ValueError:
                    return np.nan # Return NaN if conversion fails
        df['交易量'] = df['交易量_processed'].apply(convert_volume)
        df = df.drop(columns=['交易量_processed']) # 删除临时列
    else:
        print("警告: CSV文件中未找到'交易量'列。")


    # 处理涨跌幅 (百分比字符串)
    if '涨跌幅' in df.columns and df['涨跌幅'].dtype == object:
        df['涨跌幅'] = df['涨跌幅'].astype(str).str.replace('%', '', regex=False).str.replace('"', '', regex=False)
        df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce') / 100
    else:
        print("警告: CSV文件中未找到'涨跌幅'列或其类型不正确。")


    # 设置日期为索引
    df = df.set_index('日期')

    # 处理缺失值：对所有数值列进行时间序列插值，然后填充剩余的NaN
    numeric_cols_for_interpolation = ['收盘', '开盘', '高', '低', '交易量', '涨跌幅']
    for col in numeric_cols_for_interpolation:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # 确保索引是DatetimeIndex才能使用method='time'
            if isinstance(df.index, pd.DatetimeIndex):
                df[col] = df[col].interpolate(method='time', limit_direction='both')
            else:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')  # 非时间索引用线性插值
            df[col] = df[col].fillna(df[col].mean())  # 填充插值后可能仍存在的NaN（如开头或结尾的NaN）
        elif col in df.columns:
            # 如果列存在但不是数值类型，尝试强制转换为数值，然后填充
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean() if df[col].mean() is not np.nan else 0) # 如果mean也是NaN，则填充0

    return df


# 4. 保存处理后的数据
def save_processed_data(df, save_path):
    """
    保存处理后的数据为CSV
    """
    if not df.empty:
        # Save DataFrame to CSV, explicitly including index label and header
        # The index (日期) will be the first column, labeled '日期'
        # All other columns will have their correct headers
        df.to_csv(save_path, index=True, index_label='日期', encoding='utf-8-sig') # Added encoding='utf-8-sig'
        print(f"处理后的数据已保存至: {save_path}")
    else:
        print("DataFrame为空，未保存数据。")


# 主函数
def main():
    # 文件路径
    raw_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集).csv'
    processed_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_processed.csv'

    # 1. 加载数据
    print("加载原始数据...")
    df = load_data(raw_file)
    if df.empty:
        print("数据加载失败或文件为空，程序退出。")
        return

    # 2. 探索原始数据
    print("\n探索原始数据...")
    explore_data(df.copy())  # 使用副本进行探索，不影响原始数据

    # 3. 预处理数据
    print("\n预处理数据...")
    processed_df = preprocess_data(df)
    if processed_df.empty:
        print("数据预处理失败，程序退出。")
        return

    # 4. 探索处理后的数据
    print("\n探索处理后的数据...")
    explore_data(processed_df.copy())

    # 5. 保存处理后的数据
    save_processed_data(processed_df, processed_file)

    print("\n数据预处理完成!")


if __name__ == "__main__":
    main()
