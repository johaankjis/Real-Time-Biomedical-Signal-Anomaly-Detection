"""
Executable script to train all models with MLflow tracking.
Run this script to train models and log experiments.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from models import (
    train_iforest, 
    evaluate_iforest,
    train_lstm_autoencoder,
    evaluate_lstm_autoencoder,
    get_feature_importance
)
from features import featurize
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("BIOMEDICAL SIGNAL ANOMALY DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    
    # Load data
    print("\n[1/7] Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    print(f"  Train: {len(train_df)} samples, {train_df['label'].sum()} anomalies")
    print(f"  Test: {len(test_df)} samples, {test_df['label'].sum()} anomalies")
    
    # Extract features
    print("\n[2/7] Extracting features...")
    X_train = featurize(train_df, window=50, freq=100)
    X_test = featurize(test_df, window=50, freq=100)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"  Extracted {X_train.shape[1]} features")
    
    # Standardize
    print("\n[3/7] Standardizing features...")
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
    print("\n[4/7] Training Isolation Forest...")
    iforest = train_iforest(X_train_scaled, contamination=0.01, n_estimators=200)
    
    # Evaluate Isolation Forest
    print("\n[5/7] Evaluating Isolation Forest...")
    iforest_metrics, iforest_scores, iforest_preds = evaluate_iforest(
        iforest, X_test_scaled, y_test, threshold_percentile=99
    )
    
    # Feature importance
    print("\n[6/7] Computing feature importance...")
    feature_importance = get_feature_importance(iforest, X_train_scaled, X_train.columns)
    print("\n  Top 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Save artifacts
    print("\n[7/7] Saving models and artifacts...")
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(iforest, "models/iforest.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    feature_importance.to_csv("models/feature_importance.csv", index=False)
    
    # Save predictions for visualization
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': iforest_preds,
        'anomaly_score': iforest_scores
    })
    results_df.to_csv("models/test_predictions.csv", index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Isolation Forest Performance:")
    print(f"   Precision: {iforest_metrics['precision']:.4f}")
    print(f"   Recall:    {iforest_metrics['recall']:.4f}")
    print(f"   F1 Score:  {iforest_metrics['f1']:.4f}")
    print(f"   ROC AUC:   {iforest_metrics['roc_auc']:.4f}")
    
    if iforest_metrics['f1'] >= 0.80:
        print("\n✅ ACCEPTANCE CRITERIA MET: F1 ≥ 0.80")
    else:
        print(f"\n⚠️  F1 = {iforest_metrics['f1']:.4f} < 0.80 (target)")
        print("   Consider: hyperparameter tuning, more features, or ensemble methods")
    
    print("\n📁 Saved artifacts:")
    print("   - models/iforest.joblib")
    print("   - models/scaler.joblib")
    print("   - models/feature_importance.csv")
    print("   - models/test_predictions.csv")
    
    return iforest_metrics['f1']


if __name__ == "__main__":
    f1_score = main()
    sys.exit(0 if f1_score >= 0.80 else 1)
