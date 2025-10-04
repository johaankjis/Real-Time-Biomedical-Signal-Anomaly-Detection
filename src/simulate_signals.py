"""
Synthetic biomedical signal generator with injected anomalies.
Generates sine wave with noise, drift, and spike anomalies.
"""
import numpy as np
import pandas as pd


def synth_signals(n=100000, freq=100, anomaly_rate=0.01, seed=42):
    """
    Generate synthetic biomedical signals with anomalies.
    
    Args:
        n: Number of samples
        freq: Sampling frequency (Hz)
        anomaly_rate: Fraction of samples to inject as anomalies
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: ts (timestamp), signal (value), label (0=normal, 1=anomaly)
    """
    np.random.seed(seed)
    
    # Time vector
    t = np.arange(n) / freq
    
    # Base signal: sine wave with noise
    base = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n)
    
    # Add gradual drift
    drift = np.linspace(0, 0.5, n)
    
    # Inject spike anomalies
    n_anomalies = int(anomaly_rate * n)
    spikes_idx = np.random.choice(n, size=n_anomalies, replace=False)
    
    # Combine base + drift
    x = base + drift
    
    # Add large spikes at anomaly indices
    x[spikes_idx] += np.random.randn(len(spikes_idx)) * 3
    
    # Create DataFrame
    df = pd.DataFrame({
        "ts": t,
        "signal": x
    })
    
    # Label anomalies
    df["label"] = 0
    df.loc[spikes_idx, "label"] = 1
    
    return df


def train_test_split_temporal(df, test_size=0.2):
    """
    Split time series data temporally (no shuffle to preserve order).
    
    Args:
        df: DataFrame with time series data
        test_size: Fraction of data for test set
    
    Returns:
        train_df, test_df
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


if __name__ == "__main__":
    # Generate and save synthetic data
    print("Generating synthetic biomedical signals...")
    df = synth_signals(n=100000, freq=100, anomaly_rate=0.01)
    
    print(f"Generated {len(df)} samples")
    print(f"Anomalies: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    
    # Split into train/test
    train_df, test_df = train_test_split_temporal(df, test_size=0.2)
    
    # Save to data directory
    import os
    os.makedirs("data", exist_ok=True)
    
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    print(f"Saved train set: {len(train_df)} samples")
    print(f"Saved test set: {len(test_df)} samples")
