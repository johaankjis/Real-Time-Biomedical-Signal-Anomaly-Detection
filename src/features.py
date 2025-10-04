"""
Feature engineering for time-series anomaly detection.
Extracts rolling statistics, entropy, peaks, and spectral features.
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq


def compute_rolling_features(df, window=50):
    """
    Compute rolling window statistics.
    
    Args:
        df: DataFrame with 'signal' column
        window: Rolling window size
    
    Returns:
        DataFrame with rolling features
    """
    features = pd.DataFrame()
    
    # Rolling mean and std
    features['rolling_mean'] = df['signal'].rolling(window=window, center=True).mean()
    features['rolling_std'] = df['signal'].rolling(window=window, center=True).std()
    
    # Rolling min/max
    features['rolling_min'] = df['signal'].rolling(window=window, center=True).min()
    features['rolling_max'] = df['signal'].rolling(window=window, center=True).max()
    
    # Rolling range
    features['rolling_range'] = features['rolling_max'] - features['rolling_min']
    
    # Deviation from rolling mean
    features['deviation_from_mean'] = df['signal'] - features['rolling_mean']
    
    # Z-score (standardized deviation)
    features['z_score'] = features['deviation_from_mean'] / (features['rolling_std'] + 1e-8)
    
    return features


def compute_entropy_features(df, window=50):
    """
    Compute entropy-based features.
    
    Args:
        df: DataFrame with 'signal' column
        window: Window size for entropy calculation
    
    Returns:
        DataFrame with entropy features
    """
    features = pd.DataFrame()
    
    # Approximate entropy using rolling std
    features['approx_entropy'] = df['signal'].rolling(window=window, center=True).apply(
        lambda x: stats.entropy(np.histogram(x, bins=10)[0] + 1e-8)
    )
    
    return features


def compute_peak_features(df, distance=20):
    """
    Compute peak detection features.
    
    Args:
        df: DataFrame with 'signal' column
        distance: Minimum distance between peaks
    
    Returns:
        DataFrame with peak features
    """
    features = pd.DataFrame(index=df.index)
    
    # Find peaks
    peaks, properties = signal.find_peaks(df['signal'].values, distance=distance)
    
    # Create binary feature for peaks
    features['is_peak'] = 0
    features.loc[peaks, 'is_peak'] = 1
    
    # Distance to nearest peak
    features['dist_to_peak'] = 0
    for i in range(len(df)):
        if i in peaks:
            features.loc[i, 'dist_to_peak'] = 0
        else:
            distances = np.abs(peaks - i)
            if len(distances) > 0:
                features.loc[i, 'dist_to_peak'] = distances.min()
            else:
                features.loc[i, 'dist_to_peak'] = len(df)
    
    return features


def compute_spectral_features(df, window=128, freq=100):
    """
    Compute spectral features using Short-Time Fourier Transform (STFT).
    
    Args:
        df: DataFrame with 'signal' column
        window: Window size for STFT
        freq: Sampling frequency
    
    Returns:
        DataFrame with spectral features
    """
    features = pd.DataFrame(index=df.index)
    
    # Compute STFT
    f, t, Zxx = signal.stft(df['signal'].values, fs=freq, nperseg=window)
    
    # Magnitude spectrum
    magnitude = np.abs(Zxx)
    
    # Interpolate to match original signal length
    from scipy.interpolate import interp1d
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(magnitude, axis=0)
    dominant_freq = f[dominant_freq_idx]
    
    # Interpolate to original length
    interp_func = interp1d(t, dominant_freq, kind='linear', fill_value='extrapolate')
    features['dominant_freq'] = interp_func(df['ts'].values)
    
    # Spectral energy
    spectral_energy = np.sum(magnitude, axis=0)
    interp_func = interp1d(t, spectral_energy, kind='linear', fill_value='extrapolate')
    features['spectral_energy'] = interp_func(df['ts'].values)
    
    return features


def compute_derivative_features(df):
    """
    Compute first and second derivatives.
    
    Args:
        df: DataFrame with 'signal' column
    
    Returns:
        DataFrame with derivative features
    """
    features = pd.DataFrame()
    
    # First derivative (velocity)
    features['first_derivative'] = np.gradient(df['signal'].values)
    
    # Second derivative (acceleration)
    features['second_derivative'] = np.gradient(features['first_derivative'].values)
    
    # Absolute derivatives
    features['abs_first_derivative'] = np.abs(features['first_derivative'])
    features['abs_second_derivative'] = np.abs(features['second_derivative'])
    
    return features


def featurize(df, window=50, freq=100):
    """
    Extract all features from signal data.
    
    Args:
        df: DataFrame with 'signal' column
        window: Window size for rolling features
        freq: Sampling frequency
    
    Returns:
        DataFrame with all features
    """
    # Original signal
    features = df[['signal']].copy()
    
    # Rolling statistics
    rolling_feats = compute_rolling_features(df, window=window)
    features = pd.concat([features, rolling_feats], axis=1)
    
    # Entropy features
    entropy_feats = compute_entropy_features(df, window=window)
    features = pd.concat([features, entropy_feats], axis=1)
    
    # Peak features
    peak_feats = compute_peak_features(df)
    features = pd.concat([features, peak_feats], axis=1)
    
    # Spectral features
    spectral_feats = compute_spectral_features(df, window=128, freq=freq)
    features = pd.concat([features, spectral_feats], axis=1)
    
    # Derivative features
    derivative_feats = compute_derivative_features(df)
    features = pd.concat([features, derivative_feats], axis=1)
    
    # Fill NaN values (from rolling windows)
    features = features.fillna(method='bfill').fillna(method='ffill')
    
    return features


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")
    
    # Load train data
    train_df = pd.read_csv("data/train.csv")
    
    print(f"Loaded {len(train_df)} training samples")
    
    # Extract features
    print("Extracting features...")
    X_train = featurize(train_df, window=50, freq=100)
    
    print(f"Extracted {X_train.shape[1]} features")
    print(f"Feature names: {list(X_train.columns)}")
    
    # Save features
    X_train.to_csv("data/X_train.csv", index=False)
    print("Saved training features to data/X_train.csv")
