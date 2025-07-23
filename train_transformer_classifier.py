import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report
import warnings
import joblib
import os
from itertools import product

# Import TensorFlow and Keras for building the Transformer model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


# 修改prepare_data_for_classification函数
def prepare_data_for_classification(X_df, y_series, window_size=30):
    # 删除原始收盘价列（防止数据泄露）
    if '收盘' in X_df.columns:
        X_df = X_df.drop(columns=['收盘'])

    # 创建时间窗口序列
    X_sequences = []
    y_direction = []

    for i in range(window_size, len(X_df)):
        # 窗口序列特征 (window_size, num_features)
        seq = X_df.iloc[i - window_size:i].values
        # 比较次日收盘与当日收盘
        current_close = X_df.iloc[i - 1]['调整收盘']  # 使用调整后收盘价
        next_close = y_series.iloc[i]
        direction = 1 if next_close > current_close else 0
        X_sequences.append(seq)
        y_direction.append(direction)

    # 转换为3D数组 (samples, timesteps, features)
    X_seq_array = np.array(X_sequences)
    y_dir_array = np.array(y_direction).reshape(-1, 1)

    # 划分训练测试集 (保留时间顺序)
    split_index = int(len(X_seq_array) * 0.8)
    X_train, X_test = X_seq_array[:split_index], X_seq_array[split_index:]
    y_train, y_test = y_dir_array[:split_index], y_dir_array[split_index:]

    print(f"序列数据形状: X_train{X_train.shape}, y_train{y_train.shape}")
    return X_train, X_test, y_train, y_test


# --- Transformer Model Components ---

class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention layer.
    多头自注意力层。
    """

    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.proj_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, proj_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, proj_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, proj_dim)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, proj_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    """
    A single Transformer Encoder Block.
    单个Transformer编码器块。
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):  # Added training=None default
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_classifier_model(input_shape, embed_dim, num_heads, ff_dim,
                                       num_transformer_blocks, mlp_units, dropout_rate):
    # 输入形状应为 (timesteps, features)
    inputs = keras.Input(shape=input_shape)  # 例如(30,18)

    # 位置编码层（新增）
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = layers.Embedding(
        input_dim=input_shape[0], output_dim=embed_dim)(positions)

    # 特征嵌入
    x = layers.Dense(embed_dim)(inputs)  # (batch, timesteps, embed_dim)
    x += position_embedding  # 加入位置信息

    # Transformer模块（处理序列）
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # 全局平均池化
    x = layers.GlobalAveragePooling1D()(x)

    # MLP分类头
    for units in mlp_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_evaluate_transformer_classifier(X_train_scaled_df, X_test_scaled_df, y_train_direction, y_test_direction,
                                          embed_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units,
                                          dropout_rate):
    """
    Train and evaluate the Transformer Classifier model with specified hyperparameters.
    使用指定的超参数训练和评估Transformer分类模型。
    """
    print(f"\n--- Training Transformer Classifier model with parameters: ---")
    print(f"  Embed Dim: {embed_dim}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Feed Forward Dim: {ff_dim}")
    print(f"  Num Transformer Blocks: {num_transformer_blocks}")
    print(f"  MLP Units: {mlp_units}")
    print(f"  Dropout Rate: {dropout_rate}")
    print("-" * 60)

    # Build the model
    input_shape = (X_train_scaled_df.shape[1],)  # Input shape is just the number of features
    model = build_transformer_classifier_model(input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks,
                                               mlp_units, dropout_rate)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Fixed learning rate for model compilation
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]  # Add more metrics
    )

    # Define callbacks for early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]
    class_counts = np.bincount(y_train_direction.flatten())
    class_weights = {0: sum(class_counts)/class_counts[0],
                     1: sum(class_counts)/class_counts[1]}
    # Train the model
    history = model.fit(
        X_train_scaled_df, y_train_direction,
        class_weight=class_weights,  # 关键平衡参数
        batch_size=32,
        epochs=100,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model on the test set
    loss, accuracy, precision, recall, roc_auc = model.evaluate(X_test_scaled_df, y_test_direction, verbose=0)

    # Predict probabilities for ROC AUC (model.evaluate only gives scalar AUC if metric is added)
    y_pred_proba = model.predict(X_test_scaled_df).flatten()
    y_pred_direction = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate F1-score and classification report manually
    f1 = f1_score(y_test_direction, y_pred_direction, zero_division=0)

    print(f"\nTransformer Classifier Evaluation Results for current combination:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  ROC AUC: {roc_auc:.4f}")  # This roc_auc is from model.evaluate, which is usually correct.
    print("\nClassification Report:")
    print(classification_report(y_test_direction, y_pred_direction, zero_division=0))
    print("=" * 60)  # Separator for clarity between combinations

    return model, {
        'Embed Dim': embed_dim,
        'Num Heads': num_heads,
        'Feed Forward Dim': ff_dim,
        'Num Transformer Blocks': num_transformer_blocks,
        'MLP Units': str(mlp_units),
        'Dropout Rate': dropout_rate,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }


def main():
    # File paths
    features_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_index_with_features.csv'
    target_file = r'D:\machine_learning\pythonProject1\data\纳斯达克100指数(25年数据集)_target_next_day_close.csv'

    # Define paths for saving model and scaler
    model_dir = r'D:\machine_learning\pythonProject1\model'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # 1. Load data (features and target separately)
    X_df, y_series = load_features_and_target_data(features_file, target_file)

    # 2. Prepare data for classification (split into train/test and create binary target)
    X_train, X_test, y_train_direction, y_test_direction = prepare_data_for_classification(X_df, y_series)

    # 3. Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames to retain column names for clarity
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # --- Hyperparameter Tuning for Transformer Classifier ---
    print("\nStarting manual hyperparameter tuning for Transformer Classifier...")

    # Define the parameter grid for Transformer classification
    # These are illustrative values, you might need to adjust them based on results.
    param_grid_values = {
        'embed_dim': [32, 64],  # Dimension of the feature embeddings
        'num_heads': [2, 4],  # Number of attention heads (must divide embed_dim)
        'ff_dim': [64, 128],  # Hidden layer size in feed forward network
        'num_transformer_blocks': [1, 2],  # Number of Transformer blocks
        'mlp_units': [(32,), (64, 32)],  # Units in the final MLP classification head
        'dropout_rate': [0.1, 0.2]
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
        param_grid_values['embed_dim'],
        param_grid_values['num_heads'],
        param_grid_values['ff_dim'],
        param_grid_values['num_transformer_blocks'],
        param_grid_values['mlp_units'],
        param_grid_values['dropout_rate']
    ))

    total_combinations = len(param_combinations)
    print(f"Total combinations to test: {total_combinations}")
    current_combination_idx = 0

    for embed_dim, num_heads, ff_dim, num_blocks, mlp_units, dropout_rate in param_combinations:
        # Skip combinations where embed_dim is not divisible by num_heads
        if embed_dim % num_heads != 0:
            print(
                f"Skipping combination: embed_dim={embed_dim}, num_heads={num_heads} (embed_dim must be divisible by num_heads)")
            continue

        current_combination_idx += 1
        print(f"\n--- Testing combination {current_combination_idx}/{total_combinations} ---")

        transformer_model, transformer_results = train_evaluate_transformer_classifier(
            X_train_scaled_df, X_test_scaled_df, y_train_direction, y_test_direction,
            embed_dim, num_heads, ff_dim, num_blocks, mlp_units, dropout_rate
        )
        all_results.append(transformer_results)

        # Use Accuracy as the primary metric for selecting the best model
        current_score = transformer_results['Accuracy']

        # Update best model based on the current score
        if current_score > best_score:
            best_score = current_score
            best_accuracy = transformer_results['Accuracy']
            best_precision = transformer_results['Precision']
            best_recall = transformer_results['Recall']
            best_f1 = transformer_results['F1-Score']
            best_roc_auc = transformer_results['ROC AUC']

            best_params = {
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'ff_dim': ff_dim,
                'num_transformer_blocks': num_blocks,
                'mlp_units': mlp_units,
                'dropout_rate': dropout_rate
            }
            best_model = transformer_model  # Store the best model instance

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Accuracy', ascending=False)  # Sort by Accuracy for display

    print("\n--- Transformer Classifier Hyperparameter Tuning Results Summary ---")
    print(results_df.to_string())  # Use to_string() for full DataFrame display

    print(f"\nBest Transformer Classifier Configuration (based on Accuracy):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"  Accuracy: {best_accuracy:.2%}")
    print(f"  Precision: {best_precision:.2f}")
    print(f"  Recall: {best_recall:.2f}")
    print(f"  F1-Score: {best_f1:.2f}")
    print(f"  ROC AUC: {best_roc_auc:.4f}")

    # 7. Save the best Transformer model and the scaler
    if best_model:  # Check if a best model was found
        # Create a unique name for the best model based on its parameters
        hls_str = str(best_params['mlp_units']).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
        model_save_path = os.path.join(model_dir,
                                       f'best_stock_prediction_model_Transformer_Classifier_ed{best_params["embed_dim"]}_nh{best_params["num_heads"]}_ff{best_params["ff_dim"]}_nb{best_params["num_transformer_blocks"]}_mlp{hls_str}_dr{str(best_params["dropout_rate"]).replace(".", "")}.h5')
        scaler_save_path = os.path.join(model_dir, f'feature_scaler_Transformer_Classifier.joblib')

        # Save Keras model in .h5 format
        best_model.save(model_save_path)
        joblib.dump(scaler, scaler_save_path)  # Save the scaler used for the overall data
        print(f"\nBest Transformer Classifier model saved to {model_save_path}")
        print(f"Corresponding scaler saved to {scaler_save_path}")
    else:
        print("\nNo best Transformer Classifier model found to save.")

    print("\nTransformer Classifier tuning and saving process completed!")


if __name__ == "__main__":
    main()