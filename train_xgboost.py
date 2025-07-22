import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler  # Added for robust scaling
import warnings
import joblib  # Import joblib for saving/loading models

# Ignore warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100)


def load_features_and_target_data(features_file_path, target_file_path):
    """
    Load feature data and target data from separate CSV files.
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
    """
    # Time series split: first 80% for training, last 20% for testing
    split_index = int(len(X_df) * 0.8)

    X_train, X_test = X_df.iloc[:split_index], X_df.iloc[split_index:]
    y_train, y_test = y_series.iloc[:split_index], y_series.iloc[split_index:]

    print(f"Training set size: {X_train.shape[0]} ({(X_train.shape[0] / len(X_df)) * 100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({(X_test.shape[0] / len(X_df)) * 100:.1f}%)")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def train_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple traditional models"""
    # List of models - now only includes linear models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1, max_iter=10000)  # Increased max_iter for Lasso
    }

    results = []
    trained_models = {}  # Dictionary to store trained models

    for name, model in models.items():
        print(f"\nTraining {name} model...")

        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model  # Store the trained model

        # Predict
        y_pred = model.predict(X_test)

        # Evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Save results
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })

        print(f"{name} Evaluation Results:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")

        # Visualize prediction results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test.values, label='Actual Values', color='blue')
        plt.plot(y_test.index, y_pred, label='Predicted Values', color='red', alpha=0.7)
        plt.title(f'{name} - Actual vs. Predicted')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.show()

    # Display all model results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAE')
    print("\nModel Evaluation Results Comparison:")
    print(results_df)

    return results_df, trained_models  # Return trained models as well


def main():
    # File paths
    features_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100_index_with_features.csv'
    target_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100_target_next_day_close.csv'

    # Define paths for saving model and scaler
    model_save_base_path = r'D:\machine_learning\pythonProject1\best_stock_prediction_model'  # Base name for saving
    scaler_save_path = r'D:\machine_learning\pythonProject1\feature_scaler.joblib'

    # 1. Load data (features and target separately)
    X_df, y_series = load_features_and_target_data(features_file, target_file)

    # 2. Prepare data (split into train/test)
    X_train, X_test, y_train, y_test = prepare_data(X_df, y_series)

    # 3. Train and evaluate traditional models (now only linear models)
    results_df, trained_traditional_models = train_evaluate_models(X_train, X_test, y_train, y_test)

    results_df = results_df.sort_values('MAE')  # Re-sort after potential removal

    # 7. Save the best model and its scaler
    print("\nSaving the best model and its scaler...")

    # Identify the best model based on MAE
    best_model_name = results_df.iloc[0]['Model']

    # Prepare full data for retraining and scaling
    X_full = X_df.copy()  # X_df already contains all features
    y_full = y_series.copy()  # y_series already contains all targets

    # Initialize and fit a StandardScaler on the full dataset
    full_data_scaler = StandardScaler()
    X_full_scaled = full_data_scaler.fit_transform(X_full)
    X_full_scaled_df = pd.DataFrame(X_full_scaled, columns=X_full.columns, index=X_full.index)

    print(f"Retraining the best model ({best_model_name}) on the full dataset before saving...")

    # Simplified saving logic as only linear models remain
    if best_model_name == 'Linear Regression':
        retrain_model = LinearRegression()
    elif best_model_name == 'Ridge Regression':
        retrain_model = Ridge(alpha=1.0)
    elif best_model_name == 'Lasso Regression':
        retrain_model = Lasso(alpha=0.1, max_iter=10000)
    else:
        print("Warning: Unknown best model type. Skipping saving.")
        return

    # Train the model on the full scaled dataset
    retrain_model.fit(X_full_scaled_df, y_full)

    # Save the retrained model and the scaler
    joblib.dump(retrain_model, f"{model_save_base_path}_{best_model_name}.joblib")
    joblib.dump(full_data_scaler, scaler_save_path)
    print(f"Best model ({best_model_name}) saved to {model_save_base_path}_{best_model_name}.joblib")
    print(f"Scaler saved to {scaler_save_path}")

    print("Model training and saving process completed!")


if __name__ == "__main__":
    main()
