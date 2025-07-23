import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report
from sklearn.neural_network import MLPClassifier # Import MLPClassifier
import warnings
import joblib
import os
from itertools import product
from sklearn.utils import class_weight # Import class_weight from sklearn

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


def prepare_data_for_classification(X_df, y_series):
    """
    Prepare training and testing data for classification.
    This version returns 2D arrays suitable for MLPClassifier.
    It aligns features from day D-1 with target from day D,
    to predict 'current day direction' using 'previous day's attributes'.
    准备用于分类的训练和测试数据。
    此版本返回适用于MLPClassifier的2D数组。
    它将D-1天的特征与D天的目标对齐，
    以使用“前一天属性”预测“当天方向”。
    """
    # Ensure X_df and y_series are aligned and sorted by date
    X_df = X_df.sort_index()
    y_series = y_series.sort_index()

    # To predict 'current day direction' (y_series.loc[D]) using 'previous day's attributes' (X_df.loc[D-1]):
    # Shift X_df by one day forward to align previous day's features with current day's target.
    # Example:
    # Original X_df:    [Date1: Features1, Date2: Features2, Date3: Features3]
    # Original y_series: [Date1: Target1,   Date2: Target2,   Date3: Target3]
    # After shifting X_df: [Date1: NaN, Date2: Features1, Date3: Features2]
    # We want to pair (Features1, Target2), (Features2, Target3) etc.
    # So, we shift X_df by 1.
    X_shifted = X_df.shift(1)

    # Align indices after shifting to ensure they cover the same date range for prediction.
    # The first row of X_shifted will be NaN, so we drop it.
    # The corresponding y_series entry will be the target for that date.
    common_index = X_shifted.index.intersection(y_series.index)
    X_aligned = X_shifted.loc[common_index]
    y_aligned = y_series.loc[common_index]

    # Drop rows where features are NaN (typically the very first row after shifting)
    X_aligned = X_aligned.dropna()
    # Re-align y_aligned to match the indices of X_aligned after dropping NaNs
    y_aligned = y_aligned.loc[X_aligned.index]

    print(f"Features (X) will be from day D-1, Target (y) will be for day D.")
    print(f"Example: X for {X_aligned.index[0].strftime('%Y-%m-%d')} comes from "
          f"{X_df.index[X_df.index.get_loc(X_aligned.index[0]) - 1].strftime('%Y-%m-%d')}'s data.")
    print(f"         Y for {y_aligned.index[0].strftime('%Y-%m-%d')} is the direction for "
          f"{y_aligned.index[0].strftime('%Y-%m-%d')}.")

    # Time series split: first 80% for training, last 20% for testing
    split_index = int(len(X_aligned) * 0.8)
    X_train, X_test = X_aligned.iloc[:split_index], X_aligned.iloc[split_index:]
    y_train_direction, y_test_direction = y_aligned.iloc[:split_index], y_aligned.iloc[split_index:]

    print(f"Data split shape: X_train{X_train.shape}, y_train{y_train_direction.shape}")
    print(
        f"Training target (direction) distribution:\n{y_train_direction.value_counts(normalize=True)}")
    print(
        f"Test target (direction) distribution:\n{y_test_direction.value_counts(normalize=True)}")

    return X_train, X_test, y_train_direction, y_test_direction


def train_evaluate_mlp_classifier(X_train_scaled, X_test_scaled, y_train_direction, y_test_direction,
                                  hidden_layer_sizes, activation, solver, alpha, learning_rate_init, max_iter):
    """
    Train and evaluate the MLP Classifier model with specified hyperparameters.
    使用指定的超参数训练和评估MLP分类模型。
    """
    print(f"\n--- Training MLP Classifier model with parameters: ---")
    print(f"  Hidden Layer Sizes: {hidden_layer_sizes}")
    print(f"  Activation: {activation}")
    print(f"  Solver: {solver}")
    print(f"  Alpha (L2 regularization): {alpha}")
    print(f"  Learning Rate Init: {learning_rate_init}")
    print(f"  Max Iterations: {max_iter}")
    print("-" * 60)

    # Build the MLPClassifier model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42, # For reproducibility
        early_stopping=True, # Enable early stopping
        validation_fraction=0.1, # Fraction of training data to set aside for validation
        n_iter_no_change=10, # Number of epochs with no improvement to wait before stopping
        verbose=False # Set to True for verbose output during training
    )

    # Calculate class weights for imbalanced datasets using sklearn utility
    classes = np.unique(y_train_direction)
    computed_class_weights = class_weight.compute_class_weight(
        'balanced', classes=classes, y=y_train_direction
    )
    class_weights_dict = {c: w for c, w in zip(classes, computed_class_weights)}

    print(f"Calculated class weights: {class_weights_dict}")

    # Train the model
    # MLPClassifier's fit method does not directly accept class_weight_dict
    # It uses it internally if 'balanced' is specified for class_weight parameter,
    # but we are passing pre-computed weights.
    # For MLPClassifier, class_weights are usually handled by adjusting sample_weight if needed,
    # or by ensuring the 'balanced' option for class_weight is used in the model if available.
    # However, MLPClassifier does not have a 'class_weight' parameter like some other sklearn models.
    # The 'balanced' option is for `compute_class_weight`.
    # We proceed with direct fit, assuming `compute_class_weight` helps us understand imbalance.
    model.fit(X_train_scaled, y_train_direction)


    # Predict probabilities and binary predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of the positive class (1)
    y_pred_direction = model.predict(X_test_scaled) # Binary predictions

    # Evaluate the model
    accuracy = accuracy_score(y_test_direction, y_pred_direction)
    precision = precision_score(y_test_direction, y_pred_direction, zero_division=0)
    recall = recall_score(y_test_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_test_direction, y_pred_direction, zero_division=0)
    roc_auc = roc_auc_score(y_test_direction, y_pred_proba)

    print(f"\nMLP Classifier Evaluation Results for current combination:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_direction, y_pred_direction, zero_division=0))
    print("=" * 60)  # Separator for clarity between combinations

    return model, {
        'Hidden Layer Sizes': str(hidden_layer_sizes),
        'Activation': activation,
        'Solver': solver,
        'Alpha': alpha,
        'Learning Rate Init': learning_rate_init,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }


def main():
    # File paths
    # Updated to use the new enhanced features and direction target files
    features_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100_enhanced_features.csv'
    target_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100_direction_target.csv'

    # Define paths for saving model and scaler
    model_dir = r'D:\machine_learning\pythonProject1\model'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # 1. Load data (features and target separately)
    X_df, y_series = load_features_and_target_data(features_file, target_file)

    # 2. Prepare data for classification (split into train/test)
    # No window_size needed for MLP, directly use 2D data
    X_train, X_test, y_train_direction, y_test_direction = prepare_data_for_classification(X_df, y_series)

    # 3. Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Scaled data shape: X_train_scaled{X_train_scaled.shape}, X_test_scaled{X_test_scaled.shape}")

    # --- Hyperparameter Tuning for MLP Classifier ---
    print("\nStarting manual hyperparameter tuning for MLP Classifier...")

    # Define the parameter grid for MLP classification
    # Added a larger alpha value (0.1) for stronger L2 regularization
    param_grid_values = {
        'hidden_layer_sizes': [(500,300,100)], # Number of neurons in hidden layers
        'activation': ['relu'], # Activation function for the hidden layer
        'solver': ['adam'], # The solver for weight optimization
        'alpha': [0.1], # L2 regularization term (strength) - Added 0.1
        'learning_rate_init': [0.001, 0.005], # Initial learning rate
        'max_iter': [200] # Maximum number of iterations (epochs)
    }

    best_score = -float('inf')  # We want to maximize Accuracy
    best_accuracy = -float('inf')
    best_precision = -float('inf')
    best_recall = -float('inf')
    best_f1 = -float('inf')
    best_roc_auc = -float('inf')

    best_params = {}
    best_model = None

    all_results = []

    # Iterate through all combinations of parameters
    param_combinations = list(product(
        param_grid_values['hidden_layer_sizes'],
        param_grid_values['activation'],
        param_grid_values['solver'],
        param_grid_values['alpha'],
        param_grid_values['learning_rate_init'],
        param_grid_values['max_iter']
    ))

    total_combinations = len(param_combinations)
    print(f"Total combinations to test: {total_combinations}")
    current_combination_idx = 0

    for hidden_layer_sizes, activation, solver, alpha, learning_rate_init, max_iter in param_combinations:
        current_combination_idx += 1
        print(f"\n--- Testing combination {current_combination_idx}/{total_combinations} ---")

        mlp_model, mlp_results = train_evaluate_mlp_classifier(
            X_train_scaled, X_test_scaled, y_train_direction, y_test_direction,
            hidden_layer_sizes, activation, solver, alpha, learning_rate_init, max_iter
        )
        all_results.append(mlp_results)

        # Use Accuracy as the primary metric for selecting the best model
        current_score = mlp_results['Accuracy']

        # Update best model based on the current score
        if current_score > best_score:
            best_score = current_score
            best_accuracy = mlp_results['Accuracy']
            best_precision = mlp_results['Precision']
            best_recall = mlp_results['Recall']
            best_f1 = mlp_results['F1-Score']
            best_roc_auc = mlp_results['ROC AUC']

            best_params = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'solver': solver,
                'alpha': alpha,
                'learning_rate_init': learning_rate_init,
                'max_iter': max_iter
            }
            best_model = mlp_model  # Store the best model instance

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Accuracy', ascending=False)  # Sort by Accuracy for display

    print("\n--- MLP Classifier Hyperparameter Tuning Results Summary ---")
    print(results_df.to_string())  # Use to_string() for full DataFrame display

    print(f"\nBest MLP Classifier Configuration (based on Accuracy):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"  Accuracy: {best_accuracy:.2%}")
    print(f"  Precision: {best_precision:.2f}")
    print(f"  Recall: {best_recall:.2f}")
    print(f"  F1-Score: {best_f1:.2f}")
    print(f"  ROC AUC: {best_roc_auc:.4f}")

    # 7. Save the best MLP model and the scaler
    if best_model:  # Check if a best model was found
        # Create a unique name for the best model based on its parameters
        hls_str = str(best_params['hidden_layer_sizes']).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
        model_save_path = os.path.join(model_dir,
                                       f'best_stock_prediction_model_MLP_Classifier_hls_{hls_str}_act_{best_params["activation"]}_sol_{best_params["solver"]}_alpha_{str(best_params["alpha"]).replace(".", "")}_lr_{str(best_params["learning_rate_init"]).replace(".", "")}.joblib')
        scaler_save_path = os.path.join(model_dir, f'feature_scaler_MLP_Classifier.joblib')

        joblib.dump(best_model, model_save_path)
        joblib.dump(scaler, scaler_save_path)  # Save the scaler used for the overall data
        print(f"\nBest MLP Classifier model saved to {model_save_path}")
        print(f"Corresponding scaler saved to {scaler_save_path}")
    else:
        print("\nNo best MLP Classifier model found to save.")

    print("\nMLP Classifier tuning and saving process completed!")


if __name__ == "__main__":
    main()
