"""
lstm_model.py - LSTM Model for Time-Series Air Quality Prediction

This module implements an LSTM (Long Short-Term Memory) neural network
for sequential air quality prediction, leveraging temporal dependencies.

Key Features:
- Sequence-based input processing
- Configurable LSTM architecture (layers, units)
- Bidirectional LSTM option
- Integration with preprocessing module for windowed data

Computational Intelligence Context:
- LSTMs are a type of Recurrent Neural Network (RNN)
- They can learn long-term dependencies in sequential data
- The gating mechanism (forget, input, output gates) allows selective memory
- Particularly suited for time-series where past values influence future

Why LSTM for Air Quality?
- Air pollution follows temporal patterns (rush hours, seasons)
- Past readings influence future concentrations
- Weather patterns affect air quality with time delays
- LSTM can capture these complex temporal relationships

Author: TCI6313 Student
Date: 2026
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
import warnings

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
        Input, Bidirectional
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
except ImportError:
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class LSTMModel:
    """
    LSTM Neural Network for Time-Series Air Quality Prediction.
    
    This class implements an LSTM-based model that processes sequential
    input data to capture temporal dependencies in air quality measurements.
    
    Attributes:
    -----------
    sequence_length : int
        Number of time steps in input sequences
    n_features : int
        Number of features per time step
    lstm_layers : List[int]
        Number of units in each LSTM layer
    dense_layers : List[int]
        Number of neurons in dense layers after LSTM
    learning_rate : float
        Learning rate for optimizer
    dropout_rate : float
        Dropout rate for regularization
    bidirectional : bool
        Whether to use bidirectional LSTM
    model : keras.Model
        The compiled Keras model
    history : dict
        Training history
    training_time : float
        Training duration in seconds
        
    Example:
    --------
    >>> model = LSTMModel(sequence_length=24, n_features=15)
    >>> model.build()
    >>> history = model.train(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        lstm_layers: List[int] = [64, 32],
        dense_layers: List[int] = [32],
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        recurrent_dropout: float = 0.1,
        l2_reg: float = 0.0001,
        bidirectional: bool = False,
        random_seed: int = 42
    ):
        """
        Initialize the LSTM model configuration.
        
        Parameters:
        -----------
        sequence_length : int
            Number of time steps to look back (e.g., 24 for 24 hours)
        n_features : int
            Number of input features per time step
        lstm_layers : List[int]
            Units in each LSTM layer. [64, 32] = two LSTM layers.
        dense_layers : List[int]
            Neurons in dense layers after LSTM encoding
        learning_rate : float
            Learning rate for Adam optimizer
        dropout_rate : float
            Dropout rate after each layer
        recurrent_dropout : float
            Dropout rate for recurrent connections (LSTM-specific)
        l2_reg : float
            L2 regularization strength
        bidirectional : bool
            If True, use Bidirectional LSTM (processes sequence both ways)
        random_seed : int
            Random seed for reproducibility
            
        Notes:
        ------
        Bidirectional LSTM:
        - Processes sequence forward AND backward
        - Can capture dependencies in both directions
        - Useful when future context helps (not always true for prediction)
        - Doubles the number of parameters
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.bidirectional = bidirectional
        self.random_seed = random_seed
        
        self.model = None
        self.history = None
        self.training_time = None
        
        # Set seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    
    def build(self) -> None:
        """
        Build and compile the LSTM model.
        
        Architecture:
        1. Input: (sequence_length, n_features)
        2. LSTM layers with dropout
        3. Dense layers for final processing
        4. Output: Single neuron (regression)
        
        LSTM Layer Details:
        - return_sequences=True for stacked LSTM (except last)
        - Recurrent dropout helps regularize temporal connections
        - Kernel regularization prevents weight explosion
        """
        print("[INFO] Building LSTM model...")
        
        self.model = Sequential(name='LSTM_TimeSeries')
        
        # Input shape: (timesteps, features)
        self.model.add(Input(
            shape=(self.sequence_length, self.n_features),
            name='input'
        ))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_layers):
            # return_sequences=True for all but the last LSTM layer
            return_sequences = (i < len(self.lstm_layers) - 1)
            
            lstm_layer = LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l2(self.l2_reg),
                name=f'lstm_{i+1}'
            )
            
            # Optionally wrap with Bidirectional
            if self.bidirectional:
                lstm_layer = Bidirectional(lstm_layer, name=f'bidirectional_{i+1}')
            
            self.model.add(lstm_layer)
        
        # Dense layers after LSTM encoding
        for i, neurons in enumerate(self.dense_layers):
            self.model.add(Dense(
                neurons,
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                name=f'dense_{i+1}'
            ))
            self.model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            self.model.add(Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer (linear for regression)
        self.model.add(Dense(1, activation='linear', name='output'))
        
        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Print model summary
        print("\n" + "="*60)
        print("LSTM MODEL ARCHITECTURE")
        print("="*60)
        self.model.summary()
        print("="*60 + "\n")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 10,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training sequences, shape (n_samples, sequence_length, n_features)
        y_train : np.ndarray
            Training targets, shape (n_samples,)
        X_val : np.ndarray
            Validation sequences
        y_val : np.ndarray
            Validation targets
        epochs : int
            Maximum training epochs
        batch_size : int
            Samples per gradient update
        early_stopping_patience : int
            Epochs to wait before early stopping
        reduce_lr_patience : int
            Epochs to wait before reducing learning rate
        verbose : int
            Training verbosity
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history
            
        Notes:
        ------
        LSTM training considerations:
        - Generally slower than feedforward networks
        - May need more epochs to converge
        - Batch size affects gradient noise and convergence
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        print("[INFO] Starting LSTM training...")
        print(f"       Epochs: {epochs}, Batch size: {batch_size}")
        print(f"       Sequence length: {self.sequence_length}")
        print(f"       Training samples: {len(X_train)}")
        print(f"       Validation samples: {len(X_val)}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Record training time
        start_time = time.time()
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        self.history = history.history
        
        print(f"\n[INFO] Training completed in {self.training_time:.2f} seconds")
        print(f"[INFO] Final training loss: {self.history['loss'][-1]:.6f}")
        print(f"[INFO] Final validation loss: {self.history['val_loss'][-1]:.6f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences, shape (n_samples, sequence_length, n_features)
            
        Returns:
        --------
        np.ndarray
            Predictions, shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        y : np.ndarray
            True targets
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with 'loss' (MSE) and 'mae' values
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        results = self.model.evaluate(X, y, verbose=0)
        return {'loss': results[0], 'mae': results[1]}
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.save(filepath)
        print(f"[INFO] Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        self.model = load_model(filepath)
        print(f"[INFO] Model loaded from {filepath}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
        --------
        Dict[str, Any]
            Model configuration parameters
        """
        return {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_layers': self.lstm_layers,
            'dense_layers': self.dense_layers,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout': self.recurrent_dropout,
            'bidirectional': self.bidirectional,
            'training_time': self.training_time
        }


def create_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_layers: List[int] = [64, 32],
    dense_layers: List[int] = [32],
    learning_rate: float = 0.001,
    bidirectional: bool = False
) -> LSTMModel:
    """
    Factory function to create and build an LSTM model.
    
    Parameters:
    -----------
    sequence_length : int
        Number of time steps
    n_features : int
        Number of features per time step
    lstm_layers : List[int]
        Units per LSTM layer
    dense_layers : List[int]
        Neurons per dense layer
    learning_rate : float
        Learning rate
    bidirectional : bool
        Use bidirectional LSTM
        
    Returns:
    --------
    LSTMModel
        Built and ready-to-train model
    """
    model = LSTMModel(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_layers=lstm_layers,
        dense_layers=dense_layers,
        learning_rate=learning_rate,
        bidirectional=bidirectional
    )
    model.build()
    return model


def train_and_evaluate_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sequence_length: int,
    n_features: int,
    lstm_layers: List[int] = [64, 32],
    dense_layers: List[int] = [32],
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    bidirectional: bool = False
) -> Tuple[LSTMModel, Dict[str, Any]]:
    """
    Complete training and evaluation pipeline for LSTM.
    
    Parameters:
    -----------
    X_train, y_train : Training data (sequences)
    X_val, y_val : Validation data
    X_test, y_test : Test data
    sequence_length : Time steps per sequence
    n_features : Features per time step
    lstm_layers : LSTM architecture
    dense_layers : Dense layer architecture
    learning_rate : Learning rate
    epochs : Maximum epochs
    batch_size : Batch size
    bidirectional : Use bidirectional LSTM
    
    Returns:
    --------
    Tuple[LSTMModel, Dict[str, Any]]
        Trained model and results dictionary
    """
    # Create and build model
    model = LSTMModel(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_layers=lstm_layers,
        dense_layers=dense_layers,
        learning_rate=learning_rate,
        bidirectional=bidirectional
    )
    model.build()
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test)
    
    # Get predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Compile results
    results = {
        'model_type': 'LSTM',
        'config': model.get_config(),
        'history': history,
        'test_loss': test_results['loss'],
        'test_mae': test_results['mae'],
        'training_time': model.training_time,
        'predictions': {
            'train': y_pred_train,
            'val': y_pred_val,
            'test': y_pred_test
        }
    }
    
    print("\n" + "="*60)
    print("LSTM MODEL RESULTS")
    print("="*60)
    print(f"Test MSE Loss: {test_results['loss']:.6f}")
    print(f"Test MAE: {test_results['mae']:.6f}")
    print(f"Training Time: {model.training_time:.2f} seconds")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing LSTM model module...")
    
    # Generate synthetic sequential data
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 24
    n_features = 15
    
    # Create sequences
    X = np.random.randn(n_samples, sequence_length, n_features)
    # Target depends on recent values
    y = np.mean(X[:, -5:, :3], axis=(1, 2)) + np.random.randn(n_samples) * 0.1
    
    # Split data
    X_train, X_val, X_test = X[:700], X[700:850], X[850:]
    y_train, y_val, y_test = y[:700], y[700:850], y[850:]
    
    # Create and train model
    model = create_lstm_model(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_layers=[32, 16],
        dense_layers=[16]
    )
    model.train(X_train, y_train, X_val, y_val, epochs=30, verbose=0)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    print(f"\nTest Results: Loss={results['loss']:.4f}, MAE={results['mae']:.4f}")
    print("\nLSTM model module test completed successfully!")
