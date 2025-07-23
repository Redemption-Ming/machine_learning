import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Added for Chinese font support in Matplotlib ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # Use SimHei or Arial Unicode MS for Chinese characters
plt.rcParams['axes.unicode_minus'] = False # Solve the problem of '-' displaying as a square
# --- End of font configuration ---

# Ignore warnings (this line is for general warnings, not specific font warnings)
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100)


def load_feature_engineered_data(file_path):
    """
    Loads the feature-engineered data from a CSV file.
    从CSV文件加载特征工程后的数据。
    """
    print(f"Loading feature-engineered data from: {file_path}...")
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


def analyze_correlation(df):
    """
    Calculates the correlation matrix for the features and prints it.
    计算特征的相关性矩阵并打印。
    """
    if df.empty:
        print("DataFrame is empty, cannot perform correlation analysis.")
        return None

    # Drop the target variable if it exists, as we are interested in feature correlation
    # Assuming 'next_day_direction' is the target for classification tasks.
    # If the file contains 'next_day_close' (for regression), it should also be dropped.
    features_df = df.drop(columns=['next_day_direction', 'next_day_close'], errors='ignore')

    # Calculate the correlation matrix
    # .corr() method by default handles NaNs by pairwise deletion (only uses non-NaN pairs)
    # If an entire column is NaN, its correlations will be NaN.
    correlation_matrix = features_df.corr()

    print("\n--- Feature Correlation Matrix ---")
    print(correlation_matrix)

    return correlation_matrix


def visualize_correlation(correlation_matrix, save_path=None):
    """
    Visualizes the correlation matrix as a heatmap.
    将相关性矩阵可视化为热力图。
    """
    if correlation_matrix is None or correlation_matrix.empty:
        print("No correlation matrix to visualize.")
        return

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Matrix Heatmap', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"\nCorrelation heatmap saved to: {save_path}")

    plt.show()


def main():
    # File path for the feature-engineered data
    # Ensure this path points to the output of enhanced_feature_engineering.py
    features_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100_enhanced_features.csv'

    # Define path for saving the correlation heatmap
    figure_save_dir = r'D:\machine_learning\pythonProject1\result_figure'
    correlation_plot_path = os.path.join(figure_save_dir, 'feature_correlation_heatmap.png')

    # 1. Load the feature-engineered data
    df_features = load_feature_engineered_data(features_file)
    if df_features.empty:
        return

    # 2. Analyze correlation
    correlation_matrix = analyze_correlation(df_features)

    # 3. Visualize correlation
    if correlation_matrix is not None:
        visualize_correlation(correlation_matrix, save_path=correlation_plot_path)

    print("\nFeature correlation analysis completed!")


if __name__ == "__main__":
    main()
