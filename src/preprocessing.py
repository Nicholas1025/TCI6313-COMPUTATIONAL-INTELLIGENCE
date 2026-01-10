"""
preprocessing.py - Data Preprocessing Module for Air Quality Prediction

This module handles all data preprocessing tasks including:
- Loading and parsing the UCI Air Quality dataset
- Handling missing values (encoded as -200 in the dataset)
- Feature engineering and datetime processing
- Data normalization using StandardScaler/MinMaxScaler
- Train/validation/test splitting
- Sliding window generation for LSTM models

Author: TCI6313 Student
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_air_quality_data(filepath: str) -> pd.DataFrame:
    """
    Load the UCI Air Quality dataset from CSV file.
    
    The dataset uses European format:
    - Semicolon (;) as delimiter
    - Comma (,) as decimal separator
    
    Parameters:
    -----------
    filepath : str
        Path to the AirQuality.csv file
        
    Returns:
    --------
    pd.DataFrame
        Raw dataframe with proper parsing
        
    Notes:
    ------
    The dataset contains hourly averaged responses from an array of 5 metal oxide 
    chemical sensors embedded in an Air Quality Chemical Multisensor Device.
    Missing values are tagged with -200 value.
    """
    # Load with European format (semicolon delimiter, comma decimal)
    df = pd.read_csv(
        filepath,
        sep=';',
        decimal=',',
        na_values=['-200', -200],  # Mark -200 as NaN during loading
        encoding='utf-8'
    )
    
    # Remove empty columns (last two columns are typically empty)
    df = df.dropna(axis=1, how='all')
    
    # Remove rows where all values are NaN
    df = df.dropna(how='all')
    
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    return df


def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create datetime index and extract temporal features.
    
    This function:
    1. Combines Date and Time columns into a datetime index
    2. Extracts useful temporal features (hour, day_of_week, month)
    3. Creates cyclical encoding for periodic features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Date' and 'Time' columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and temporal features
        
    Notes:
    ------
    Cyclical encoding using sin/cos transforms is used because:
    - Hour 23 should be close to hour 0
    - December should be close to January
    This is important for neural networks to understand periodicity.
    """
    df = df.copy()
    
    # Parse datetime - handle the format DD/MM/YYYY and HH.MM.SS
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False),
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    
    # Set datetime as index
    df = df.set_index('DateTime')
    
    # Drop original Date and Time columns
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Extract temporal features
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    
    # Cyclical encoding for periodic features
    # This ensures that hour 23 is close to hour 0, etc.
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    print(f"[INFO] Created datetime features. New shape: {df.shape}")
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'interpolate',
    max_gap: int = 3
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with potential missing values
    strategy : str
        Strategy for handling missing values:
        - 'interpolate': Linear interpolation (recommended for time series)
        - 'ffill': Forward fill
        - 'drop': Drop rows with missing values
        - 'mean': Fill with column mean
    max_gap : int
        Maximum number of consecutive NaN values to interpolate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
        
    Notes:
    ------
    For time-series data, interpolation is generally preferred because:
    1. It preserves temporal continuity
    2. It doesn't introduce sharp discontinuities
    3. It works well for short gaps in sensor data
    """
    df = df.copy()
    
    # Count missing values before
    missing_before = df.isnull().sum().sum()
    print(f"[INFO] Missing values before handling: {missing_before}")
    
    if strategy == 'interpolate':
        # Linear interpolation with limit on consecutive NaNs
        df = df.interpolate(method='linear', limit=max_gap, limit_direction='both')
        # Drop remaining NaNs (gaps too large to interpolate)
        df = df.dropna()
        
    elif strategy == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
        
    elif strategy == 'drop':
        df = df.dropna()
        
    elif strategy == 'mean':
        df = df.fillna(df.mean())
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Count missing values after
    missing_after = df.isnull().sum().sum()
    print(f"[INFO] Missing values after handling: {missing_after}")
    print(f"[INFO] Final shape after handling missing values: {df.shape}")
    
    return df


def select_features_and_target(
    df: pd.DataFrame,
    target_column: str = 'C6H6(GT)',
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features and target variable for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    target_column : str
        Name of the target column to predict (default: 'C6H6(GT)' - Benzene concentration)
    feature_columns : List[str], optional
        Specific columns to use as features. If None, uses all except target.
    exclude_columns : List[str], optional
        Columns to exclude from features (e.g., raw temporal features)
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X is features and y is target
        
    Notes:
    ------
    Default target is C6H6(GT) (Benzene) because:
    1. It's an important air quality indicator
    2. It has relatively complete data
    3. It shows clear correlation with sensor readings
    """
    df = df.copy()
    
    # Default columns to exclude (raw temporal - we use cyclical encoding instead)
    if exclude_columns is None:
        exclude_columns = ['Hour', 'DayOfWeek', 'Month', 'DayOfYear']
    
    # Extract target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    y = df[target_column].copy()
    
    # Select features
    if feature_columns is not None:
        X = df[feature_columns].copy()
    else:
        # Use all columns except target and excluded columns
        feature_cols = [col for col in df.columns 
                       if col != target_column and col not in exclude_columns]
        X = df[feature_cols].copy()
    
    print(f"[INFO] Selected {len(X.columns)} features: {list(X.columns)}")
    print(f"[INFO] Target variable: {target_column}")
    
    return X, y


def scale_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """
    Scale features and target using StandardScaler or MinMaxScaler.
    
    Parameters:
    -----------
    X_train, X_val, X_test : np.ndarray
        Feature arrays for train, validation, and test sets
    y_train, y_val, y_test : np.ndarray
        Target arrays for train, validation, and test sets
    scaler_type : str
        Type of scaler: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        
    Returns:
    --------
    Tuple containing:
        - Scaled X_train, X_val, X_test
        - Scaled y_train, y_val, y_test
        - feature_scaler object (for inverse transform)
        - target_scaler object (for inverse transform)
        
    Notes:
    ------
    IMPORTANT: Scalers are fit ONLY on training data to prevent data leakage.
    The same transformation is then applied to validation and test sets.
    
    StandardScaler is preferred for neural networks because:
    - Centers data around 0 with unit variance
    - Works well with gradient-based optimization
    - More robust to outliers than MinMaxScaler
    """
    # Initialize scalers
    if scaler_type == 'standard':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif scaler_type == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    # Fit scalers on training data only (CRITICAL: prevents data leakage)
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale target (reshape for sklearn)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"[INFO] Data scaled using {scaler_type} scaler")
    print(f"[INFO] Feature range after scaling - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            feature_scaler, target_scaler)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    shuffle: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame
    y : pd.Series
        Target Series
    test_size : float
        Proportion of data for test set (default: 0.15)
    val_size : float
        Proportion of data for validation set (default: 0.15)
    random_state : int
        Random seed for reproducibility
    shuffle : bool
        Whether to shuffle data before splitting.
        For time-series: False (maintain temporal order)
        For non-temporal: True
        
    Returns:
    --------
    Tuple of numpy arrays:
        X_train, X_val, X_test, y_train, y_val, y_test
        
    Notes:
    ------
    For time-series data, we typically don't shuffle to maintain temporal order.
    This prevents future data from leaking into the training set.
    
    Split proportions:
    - Train: 70% (default)
    - Validation: 15% (for hyperparameter tuning)
    - Test: 15% (for final evaluation)
    """
    # Convert to numpy arrays
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_arr, y_arr,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    
    # Second split: separate validation from training
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=shuffle
    )
    
    print(f"[INFO] Data split completed:")
    print(f"       Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_arr)*100:.1f}%)")
    print(f"       Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X_arr)*100:.1f}%)")
    print(f"       Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_arr)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM input.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature array of shape (n_samples, n_features)
    y : np.ndarray
        Target array of shape (n_samples,)
    sequence_length : int
        Number of time steps to look back (default: 24 hours)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X_seq: shape (n_sequences, sequence_length, n_features)
        y_seq: shape (n_sequences,)
        
    Notes:
    ------
    For hourly data, sequence_length=24 means using past 24 hours to predict.
    
    The sliding window approach:
    - Window 1: X[0:24] -> y[24]
    - Window 2: X[1:25] -> y[25]
    - ...and so on
    
    This is essential for LSTM because:
    1. LSTMs process sequential data
    2. The temporal context (past 24 hours) helps capture patterns
    3. Each prediction uses historical context
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"[INFO] Created sequences with length {sequence_length}")
    print(f"       X_seq shape: {X_seq.shape}")
    print(f"       y_seq shape: {y_seq.shape}")
    
    return X_seq, y_seq


def prepare_data_for_ann(
    filepath: str,
    target_column: str = 'C6H6(GT)',
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler_type: str = 'standard',
    random_state: int = 42
) -> dict:
    """
    Complete preprocessing pipeline for ANN model.
    
    This is a convenience function that combines all preprocessing steps
    for the feedforward ANN baseline model.
    
    Parameters:
    -----------
    filepath : str
        Path to AirQuality.csv
    target_column : str
        Column to predict
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set
    scaler_type : str
        'standard' or 'minmax'
    random_state : int
        Random seed
        
    Returns:
    --------
    dict containing:
        - All train/val/test data (scaled)
        - Scalers for inverse transformation
        - Feature names
        - Original data shapes
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE FOR ANN")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_air_quality_data(filepath)
    
    # Step 2: Create datetime features
    df = create_datetime_features(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df, strategy='interpolate')
    
    # Step 4: Select features and target
    X, y = select_features_and_target(df, target_column=target_column)
    feature_names = list(X.columns)
    
    # Step 5: Split data (no shuffle for time-series)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size,
        random_state=random_state, shuffle=False
    )
    
    # Step 6: Scale data
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train_scaled, y_val_scaled, y_test_scaled,
     feature_scaler, target_scaler) = scale_data(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler_type=scaler_type
    )
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_scaled,
        'y_val': y_val_scaled,
        'y_test': y_test_scaled,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_names,
        'n_features': X_train_scaled.shape[1],
        'n_train': X_train_scaled.shape[0],
        'n_val': X_val_scaled.shape[0],
        'n_test': X_test_scaled.shape[0]
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
    """
    Complete preprocessing pipeline for LSTM model.
    
    This function extends the ANN preprocessing with sequence generation
    for time-series modeling with LSTM.
    
    Parameters:
    -----------
    filepath : str
        Path to AirQuality.csv
    target_column : str
        Column to predict
    sequence_length : int
        Number of time steps for LSTM input (default: 24 hours)
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set
    scaler_type : str
        'standard' or 'minmax'
    random_state : int
        Random seed
        
    Returns:
    --------
    dict containing:
        - All train/val/test sequences (scaled)
        - Scalers for inverse transformation
        - Feature names
        - Sequence length
        
    Notes:
    ------
    Key difference from ANN preprocessing:
    - Data is scaled BEFORE sequence creation
    - This ensures consistent scaling across sequences
    - Sequences are 3D: (samples, timesteps, features)
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE FOR LSTM")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_air_quality_data(filepath)
    
    # Step 2: Create datetime features
    df = create_datetime_features(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df, strategy='interpolate')
    
    # Step 4: Select features and target
    X, y = select_features_and_target(df, target_column=target_column)
    feature_names = list(X.columns)
    
    # Step 5: Split data BEFORE scaling (no shuffle for time-series)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size,
        random_state=random_state, shuffle=False
    )
    
    # Step 6: Scale data
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train_scaled, y_val_scaled, y_test_scaled,
     feature_scaler, target_scaler) = scale_data(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler_type=scaler_type
    )
    
    # Step 7: Create sequences for LSTM
    print("\n[INFO] Creating sequences for LSTM...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return {
        'X_train': X_train_seq,
        'X_val': X_val_seq,
        'X_test': X_test_seq,
        'y_train': y_train_seq,
        'y_val': y_val_seq,
        'y_test': y_test_seq,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_names,
        'sequence_length': sequence_length,
        'n_features': X_train_seq.shape[2],
        'n_train': X_train_seq.shape[0],
        'n_val': X_val_seq.shape[0],
        'n_test': X_test_seq.shape[0]
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def inverse_transform_predictions(
    y_pred: np.ndarray,
    target_scaler: object
) -> np.ndarray:
    """
    Convert scaled predictions back to original scale.
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Scaled predictions
    target_scaler : object
        Fitted scaler object
        
    Returns:
    --------
    np.ndarray
        Predictions in original scale
    """
    return target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()


def get_data_summary(data_dict: dict) -> pd.DataFrame:
    """
    Generate a summary of the preprocessed data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary returned by prepare_data_for_ann or prepare_data_for_lstm
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    summary = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'Samples': [data_dict['n_train'], data_dict['n_val'], data_dict['n_test']],
        'Features': [data_dict['n_features']] * 3
    }
    
    return pd.DataFrame(summary)


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing module...")
    
    # Test with sample path (adjust as needed)
    filepath = "../data/AirQuality.csv"
    
    # Test ANN preprocessing
    print("\n" + "="*60)
    print("Testing ANN preprocessing...")
    ann_data = prepare_data_for_ann(filepath)
    print(get_data_summary(ann_data))
    
    # Test LSTM preprocessing
    print("\n" + "="*60)
    print("Testing LSTM preprocessing...")
    lstm_data = prepare_data_for_lstm(filepath, sequence_length=24)
    print(get_data_summary(lstm_data))
