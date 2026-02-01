"""
Data preprocessing for air quality prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


def load_air_quality_data(filepath: str) -> pd.DataFrame:
    """Load UCI Air Quality dataset (European CSV format)."""
    df = pd.read_csv(filepath, sep=';', decimal=',', na_values=['-200', -200], encoding='utf-8')
    df = df.dropna(axis=1, how='all')
    df = df.dropna(how='all')
    
    print(f"[INFO] Loaded: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create datetime index and cyclical temporal features."""
    df = df.copy()
    
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False),
        format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )
    
    df = df.set_index('DateTime')
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Extract temporal features
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    
    # Cyclical encoding
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    print(f"[INFO] Datetime features created: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'interpolate', max_gap: int = 3) -> pd.DataFrame:
    """Handle missing values."""
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    print(f"[INFO] Missing before: {missing_before}")
    
    if strategy == 'interpolate':
        df = df.interpolate(method='linear', limit=max_gap, limit_direction='both')
        df = df.dropna()
    elif strategy == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        df = df.fillna(df.mean())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    missing_after = df.isnull().sum().sum()
    print(f"[INFO] Missing after: {missing_after}")
    print(f"[INFO] Final shape: {df.shape}")
    return df


def select_features_and_target(
    df: pd.DataFrame,
    target_column: str = 'C6H6(GT)',
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Select features and target variable."""
    df = df.copy()
    
    if exclude_columns is None:
        exclude_columns = ['Hour', 'DayOfWeek', 'Month', 'DayOfYear']
    
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found")
    
    y = df[target_column].copy()
    
    if feature_columns is not None:
        X = df[feature_columns].copy()
    else:
        feature_cols = [col for col in df.columns if col != target_column and col not in exclude_columns]
        X = df[feature_cols].copy()
    
    print(f"[INFO] Features: {len(X.columns)}")
    print(f"[INFO] Target: {target_column}")
    return X, y


def scale_data(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """Scale features and target (fit on training data only)."""
    if scaler_type == 'standard':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif scaler_type == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)
    
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"[INFO] Data scaled ({scaler_type})")
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            feature_scaler, target_scaler)


def split_data(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = 0.15, val_size: float = 0.15,
    random_state: int = 42, shuffle: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test sets."""
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=shuffle)
    
    print(f"[INFO] Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM."""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"[INFO] Sequences: X={X_seq.shape}, y={y_seq.shape}")
    return X_seq, y_seq


def prepare_data_for_ann(
    filepath: str,
    target_column: str = 'C6H6(GT)',
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler_type: str = 'standard',
    random_state: int = 42
) -> dict:
    """Complete preprocessing pipeline for ANN."""
    print("=" * 60)
    print("PREPROCESSING FOR ANN")
    print("=" * 60)
    
    df = load_air_quality_data(filepath)
    df = create_datetime_features(df)
    df = handle_missing_values(df, strategy='interpolate')
    X, y = select_features_and_target(df, target_column=target_column)
    feature_names = list(X.columns)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state, shuffle=False)
    
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train_scaled, y_val_scaled, y_test_scaled,
     feature_scaler, target_scaler) = scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_type=scaler_type)
    
    print("=" * 60)
    
    return {
        'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_scaled, 'y_val': y_val_scaled, 'y_test': y_test_scaled,
        'feature_scaler': feature_scaler, 'target_scaler': target_scaler,
        'feature_names': feature_names, 'n_features': X_train_scaled.shape[1],
        'n_train': X_train_scaled.shape[0], 'n_val': X_val_scaled.shape[0], 'n_test': X_test_scaled.shape[0]
    }


def prepare_data_for_lstm(
    filepath: str,
    target_column: str = 'C6H6(GT)',
    sequence_length: int = 24,
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler_type: str = 'standard',
    random_state: int = 42
) -> dict:
    """Complete preprocessing pipeline for LSTM."""
    print("=" * 60)
    print("PREPROCESSING FOR LSTM")
    print("=" * 60)
    
    df = load_air_quality_data(filepath)
    df = create_datetime_features(df)
    df = handle_missing_values(df, strategy='interpolate')
    X, y = select_features_and_target(df, target_column=target_column)
    feature_names = list(X.columns)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state, shuffle=False)
    
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train_scaled, y_val_scaled, y_test_scaled,
     feature_scaler, target_scaler) = scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_type=scaler_type)
    
    print("\n[INFO] Creating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
    
    print("=" * 60)
    
    return {
        'X_train': X_train_seq, 'X_val': X_val_seq, 'X_test': X_test_seq,
        'y_train': y_train_seq, 'y_val': y_val_seq, 'y_test': y_test_seq,
        'feature_scaler': feature_scaler, 'target_scaler': target_scaler,
        'feature_names': feature_names, 'sequence_length': sequence_length,
        'n_features': X_train_seq.shape[2],
        'n_train': X_train_seq.shape[0], 'n_val': X_val_seq.shape[0], 'n_test': X_test_seq.shape[0]
    }


def prepare_data_for_lstm_univariate(
    filepath: str,
    target_column: str = 'C6H6(GT)',
    sequence_length: int = 24,
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler_type: str = 'minmax',
    random_state: int = 42
) -> dict:
    """Prepare data for univariate LSTM (single variable)."""
    print("=" * 60)
    print("PREPROCESSING FOR UNIVARIATE LSTM")
    print("=" * 60)
    
    df = load_air_quality_data(filepath)
    df = create_datetime_features(df)
    df = handle_missing_values(df, strategy='interpolate')
    
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found")
    
    y = df[target_column].values.reshape(-1, 1)
    print(f"[INFO] Univariate: {target_column}, samples: {len(y)}")
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()
    
    y_scaled = scaler.fit_transform(y)
    
    n_samples = len(y_scaled)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    y_train = y_scaled[:n_train]
    y_val = y_scaled[n_train:n_train + n_val]
    y_test = y_scaled[n_train + n_val:]
    
    print(f"[INFO] Split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    def create_univariate_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), 0])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train_seq = create_univariate_sequences(y_train, sequence_length)
    X_val, y_val_seq = create_univariate_sequences(y_val, sequence_length)
    X_test, y_test_seq = create_univariate_sequences(y_test, sequence_length)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"[INFO] Sequences: X_train={X_train.shape}")
    print("=" * 60)
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train_seq, 'y_val': y_val_seq, 'y_test': y_test_seq,
        'target_scaler': scaler, 'feature_scaler': scaler,
        'feature_names': [target_column], 'sequence_length': sequence_length,
        'n_features': 1, 'n_train': X_train.shape[0], 'n_val': X_val.shape[0], 'n_test': X_test.shape[0],
        'univariate': True
    }


def inverse_transform_predictions(y_pred: np.ndarray, target_scaler: object) -> np.ndarray:
    """Convert scaled predictions back to original scale."""
    return target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()


def get_data_summary(data_dict: dict) -> pd.DataFrame:
    """Generate data summary."""
    return pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Samples': [data_dict['n_train'], data_dict['n_val'], data_dict['n_test']],
        'Features': [data_dict['n_features']] * 3
    })


if __name__ == "__main__":
    print("Testing preprocessing module...")
    
    filepath = "../data/AirQuality.csv"
    
    print("\nTesting ANN preprocessing...")
    ann_data = prepare_data_for_ann(filepath)
    print(get_data_summary(ann_data))
    
    print("\nTesting LSTM preprocessing...")
    lstm_data = prepare_data_for_lstm(filepath, sequence_length=24)
    print(get_data_summary(lstm_data))
