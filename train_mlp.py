import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import warnings
import joblib
import os

# Ignore warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100)


def load_features_and_target_data(features_file_path, target_file_path):
    """
    Load feature data and target data from separate CSV files.
    加载特征数据和目标数据，从单独的CSV文件加载。
    """
    print("Loading feature data...")
    X_df = pd.read_csv(features_file_path, index_col='日期', parse_dates=True, encoding='utf-8-sig')
    print(f"Feature data shape: {X_df.shape}")

    print("Loading target data...")
    y_series = pd.read_csv(target_file_path, index_col='日期', parse_dates=True, encoding='utf-8-sig')
    # Ensure y_series is a Series, not a DataFrame with one column
    y_series = y_series.iloc[:, 0]
    print(f"Target data shape: {y_series.shape}")

    # Align indices to ensure features and target match perfectly
    common_index = X_df.index.intersection(y_series.index)
    X_df = X_df.loc[common_index]
    y_series = y_series.loc[common_index]

    print(f"Aligned data shape: {X_df.shape} for features, {y_series.shape} for target.")

    return X_df, y_series


def prepare_data(X_df, y_series):
    """
    Prepare training and testing data from separate features and target.
    准备训练和测试数据，从分离的特征和目标中获取。
    """
    # Time series split: first 80% for training, last 20% for testing
    split_index = int(len(X_df) * 0.8)

    X_train, X_test = X_df.iloc[:split_index], X_df.iloc[split_index:]
    y_train, y_test = y_series.iloc[:split_index], y_series.iloc[split_index:]

    print(f"Training set size: {X_train.shape[0]} ({(X_train.shape[0] / len(X_df)) * 100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({(X_test.shape[0] / len(X_df)) * 100:.1f}%)")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def main():
    # File paths
    features_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_index_with_features.csv'
    target_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_target_next_day_close.csv'

    # Define paths for saving model and scaler
    model_dir = r'D:\machine_learning\pythonProject1\model'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define path for saving figures
    result_figure_dir = r'D:\machine_learning\pythonProject1\result_figure'
    os.makedirs(result_figure_dir, exist_ok=True) # Create the directory if it doesn't exist

    # 1. Load data (features and target separately)
    X_df, y_series = load_features_and_target_data(features_file, target_file)

    # 2. Prepare data (split into train/test)
    X_train, X_test, y_train, y_test = prepare_data(X_df, y_series)

    # 3. Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames to retain column names for clarity
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Get the '收盘' (closing price) from X_test for directional accuracy calculation
    current_day_close_test = X_test['收盘']

    # --- Train and Evaluate MLP Regressor with Best Parameters ---
    print("\nTraining MLP Regressor with the best parameters...")

    # Best parameters from previous run
    best_hidden_layer_sizes = (70, 35)
    best_activation = 'relu'
    best_solver = 'lbfgs'
    best_learning_rate_init = 0.0005
    best_alpha = 0.01

    # Initialize MLP Regressor model with the best parameters
    mlp_model = MLPRegressor(hidden_layer_sizes=best_hidden_layer_sizes,
                             activation=best_activation,
                             solver=best_solver,
                             learning_rate_init=best_learning_rate_init,
                             alpha=best_alpha,
                             max_iter=1000,
                             random_state=42,
                             early_stopping=True,
                             n_iter_no_change=10,
                             verbose=True) # Keep verbose=True to see training progress

    # Train model on scaled data
    mlp_model.fit(X_train_scaled_df, y_train)

    # Predict on scaled test data
    y_pred = mlp_model.predict(X_test_scaled_df)

    # Evaluation metrics for regression
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Directional Accuracy
    true_direction = np.sign(y_test - current_day_close_test)
    predicted_direction = np.sign(y_pred - current_day_close_test)
    directional_accuracy = np.mean(true_direction == predicted_direction)

    print("\n--- Final MLP Regressor Evaluation Results with Best Parameters ---")
    print(f"  Hidden Layers: {best_hidden_layer_sizes}")
    print(f"  Activation: {best_activation}")
    print(f"  Solver: {best_solver}")
    print(f"  Learning Rate Init: {best_learning_rate_init}")
    print(f"  Alpha: {best_alpha}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Directional Accuracy: {directional_accuracy:.2%}")

    # --- Plotting Actual vs. Predicted Values ---
    print("\nGenerating prediction plot...")
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Closing Price', color='blue', linewidth=2)
    plt.plot(y_test.index, y_pred, label='Predicted Closing Price', color='red', linestyle='--', linewidth=2)
    plt.title('MLP Regression Model: Actual vs. Predicted Values', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    # Save the plot to the specified directory
    plot_save_path = os.path.join(result_figure_dir, 'mlp_actual_vs_predicted.png')
    plt.savefig(plot_save_path)
    print(f"Prediction plot saved to: {plot_save_path}")

    plt.show() # Display the plot


    # 7. Save the best MLP model and the scaler
    print("\nSaving the MLP model and its scaler...")
    best_hls_str = str(best_hidden_layer_sizes).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
    model_save_path = os.path.join(model_dir,
                                   f'best_stock_prediction_model_MLP_Regressor_hls_{best_hls_str}_solver_{best_solver}_lr_{best_learning_rate_init}_alpha_{best_alpha}.joblib')
    scaler_save_path = os.path.join(model_dir, f'feature_scaler_MLP_hls_{best_hls_str}.joblib')

    joblib.dump(mlp_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\nMLP model saved to {model_save_path}")
    print(f"Corresponding scaler saved to {scaler_save_path}")

    print("\nMLP model training, evaluation, and saving process completed!")


if __name__ == "__main__":
    main()
