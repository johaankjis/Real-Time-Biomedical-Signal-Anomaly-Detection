"""
Model training and evaluation for anomaly detection.
Implements Isolation Forest and LSTM Autoencoder.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layer```python file="src/models.py"
"""
Model training and evaluation for anomaly detection.
Implements Isolation Forest and LSTM Autoencoder.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def train_iforest(X_train, contamination=0.01, n_estimators=200, random_state=42):
    """
    Train Isolation Forest model.
    
    Args:
        X_train: Training features
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees
        random_state: Random seed
    
    Returns:
        Trained Isolation Forest model
    """
    print(f"Training Isolation Forest with {n_estimators} estimators...")
    
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    iso.fit(X_train)
    
    print("Isolation Forest training complete!")
    return iso


def evaluate_iforest(model, X_test, y_true, threshold_percentile=99):
    """
    Evaluate Isolation Forest model.
    
    Args:
        model: Trained Isolation Forest
        X_test: Test features
        y_true: True labels
        threshold_percentile: Percentile for anomaly threshold
    
    Returns:
        Dictionary with metrics and predictions
    """
    # Get anomaly scores (higher = more anomalous)
    scores = -model.decision_function(X_test)
    
    # Determine threshold
    threshold = np.percentile(scores, threshold_percentile)
    
    # Predict anomalies
    y_pred = (scores > threshold).astype(int)
    
    # Calculate metrics
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except:
        roc_auc = 0.0
    
    metrics = {
        "precision": p,
        "recall": r,
        "f1": f,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "confusion_matrix": cm
    }
    
    print(f"\nIsolation Forest Metrics:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall: {r:.4f}")
    print(f"  F1 Score: {f:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return metrics, scores, y_pred


def build_lstm_autoencoder(input_dim, encoding_dim=32):
    """
    Build LSTM Autoencoder for anomaly detection.
    
    Args:
        input_dim: Number of input features
        encoding_dim: Dimension of encoded representation
    
    Returns:
        Compiled Keras model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dim, 1))
    
    # LSTM layers
    x = layers.LSTM(64, activation='relu', return_sequences=True)(encoder_inputs)
    x = layers.LSTM(encoding_dim, activation='relu', return_sequences=False)(x)
    
    # Decoder
    x = layers.RepeatVector(input_dim)(x)
    x = layers.LSTM(encoding_dim, activation='relu', return_sequences=True)(x)
    x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
    
    # Output
    decoder_outputs = layers.TimeDistributed(layers.Dense(1))(x)
    
    # Autoencoder model
    autoencoder = keras.Model(encoder_inputs, decoder_outputs, name='lstm_autoencoder')
    
    # Compile
    autoencoder.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder


def train_lstm_autoencoder(X_train, epochs=50, batch_size=256, validation_split=0.1):
    """
    Train LSTM Autoencoder on normal data.
    
    Args:
        X_train: Training features (normal data only)
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
    
    Returns:
        Trained autoencoder model
    """
    print(f"Training LSTM Autoencoder...")
    
    # Reshape for LSTM: (samples, timesteps, features)
    X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Build model
    model = build_lstm_autoencoder(input_dim=X_train.shape[1])
    
    print(model.summary())
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        X_train_reshaped,
        X_train_reshaped,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("LSTM Autoencoder training complete!")
    return model, history


def evaluate_lstm_autoencoder(model, X_test, y_true, threshold_percentile=99):
    """
    Evaluate LSTM Autoencoder.
    
    Args:
        model: Trained autoencoder
        X_test: Test features
        y_true: True labels
        threshold_percentile: Percentile for anomaly threshold
    
    Returns:
        Dictionary with metrics and predictions
    """
    # Reshape for LSTM
    X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Reconstruct
    X_reconstructed = model.predict(X_test_reshaped, verbose=0)
    
    # Calculate reconstruction error (MSE per sample)
    reconstruction_error = np.mean(np.square(X_test_reshaped - X_reconstructed), axis=(1, 2))
    
    # Determine threshold
    threshold = np.percentile(reconstruction_error, threshold_percentile)
    
    # Predict anomalies
    y_pred = (reconstruction_error > threshold).astype(int)
    
    # Calculate metrics
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, reconstruction_error)
    except:
        roc_auc = 0.0
    
    metrics = {
        "precision": p,
        "recall": r,
        "f1": f,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "confusion_matrix": cm
    }
    
    print(f"\nLSTM Autoencoder Metrics:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall: {r:.4f}")
    print(f"  F1 Score: {f:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return metrics, reconstruction_error, y_pred


def get_feature_importance(model, X_train, feature_names):
    """
    Get feature importance for Isolation Forest using permutation importance.
    
    Args:
        model: Trained Isolation Forest
        X_train: Training features
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance scores
    """
    from sklearn.inspection import permutation_importance
    
    print("Computing feature importance...")
    
    # Use a subset for faster computation
    n_samples = min(5000, len(X_train))
    X_subset = X_train.iloc[:n_samples]
    
    # Compute permutation importance
    result = permutation_importance(
        model, 
        X_subset, 
        np.zeros(len(X_subset)),  # Dummy labels
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    print("=" * 60)
    print("BIOMEDICAL SIGNAL ANOMALY DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train anomalies: {train_df['label'].sum()} ({train_df['label'].mean()*100:.2f}%)")
    print(f"Test anomalies: {test_df['label'].sum()} ({test_df['label'].mean()*100:.2f}%)")
    
    # Extract features
    print("\n[2/6] Extracting features...")
    from features import featurize
    
    X_train = featurize(train_df, window=50, freq=100)
    X_test = featurize(test_df, window=50, freq=100)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"Features extracted: {X_train.shape[1]}")
    
    # Standardize features
    print("\n[3/6] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # Train Isolation Forest
    print("\n[4/6] Training Isolation Forest...")
    iforest = train_iforest(X_train_scaled, contamination=0.01, n_estimators=200)
    
    # Evaluate Isolation Forest
    print("\n[5/6] Evaluating Isolation Forest...")
    iforest_metrics, iforest_scores, iforest_preds = evaluate_iforest(
        iforest, X_test_scaled, y_test, threshold_percentile=99
    )
    
    # Get feature importance
    feature_importance = get_feature_importance(iforest, X_train_scaled, X_train.columns)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Save models
    print("\n[6/6] Saving models...")
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(iforest, "models/iforest.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    feature_importance.to_csv("models/feature_importance.csv", index=False)
    
    # Save metrics
    metrics_summary = {
        "iforest_f1": iforest_metrics["f1"],
        "iforest_precision": iforest_metrics["precision"],
        "iforest_recall": iforest_metrics["recall"],
        "iforest_roc_auc": iforest_metrics["roc_auc"],
        "threshold": iforest_metrics["threshold"]
    }
    
    pd.DataFrame([metrics_summary]).to_csv("models/metrics.csv", index=False)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal F1 Score: {iforest_metrics['f1']:.4f}")
    
    if iforest_metrics['f1'] >= 0.80:
        print("✅ ACCEPTANCE CRITERIA MET: F1 ≥ 0.80")
    else:
        print("⚠️  F1 score below target. Consider hyperparameter tuning.")
    
    print("\nSaved artifacts:")
    print("  - models/iforest.joblib")
    print("  - models/scaler.joblib")
    print("  - models/feature_importance.csv")
    print("  - models/metrics.csv")
