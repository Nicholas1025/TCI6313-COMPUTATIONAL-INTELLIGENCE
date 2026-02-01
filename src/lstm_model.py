"""
LSTM model for time-series air quality prediction.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class LSTMModel:
    """LSTM for time-series prediction."""
    
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
        
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    
    def build(self) -> None:
        """Build and compile the model."""
        print("[INFO] Building LSTM model...")
        
        self.model = Sequential(name='LSTM_TimeSeries')
        self.model.add(Input(shape=(self.sequence_length, self.n_features), name='input'))
        
        for i, units in enumerate(self.lstm_layers):
            return_sequences = (i < len(self.lstm_layers) - 1)
            
            lstm_layer = LSTM(
                units, return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l2(self.l2_reg),
                name=f'lstm_{i+1}'
            )
            
            if self.bidirectional:
                lstm_layer = Bidirectional(lstm_layer, name=f'bidirectional_{i+1}')
            
            self.model.add(lstm_layer)
        
        for i, neurons in enumerate(self.dense_layers):
            self.model.add(Dense(neurons, activation='relu',
                kernel_regularizer=l2(self.l2_reg), name=f'dense_{i+1}'))
            self.model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            self.model.add(Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        self.model.add(Dense(1, activation='linear', name='output'))
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
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
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        print("[INFO] Starting LSTM training...")
        print(f"       Epochs: {epochs}, Batch size: {batch_size}")
        print(f"       Sequence length: {self.sequence_length}")
        print(f"       Training: {len(X_train)}, Validation: {len(X_val)}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience,
                restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                patience=reduce_lr_patience, min_lr=1e-7, verbose=1)
        ]
        
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        self.history = history.history
        
        print(f"\n[INFO] Training completed in {self.training_time:.2f}s")
        print(f"[INFO] Final train loss: {self.history['loss'][-1]:.6f}")
        print(f"[INFO] Final val loss: {self.history['val_loss'][-1]:.6f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model."""
        if self.model is None:
            raise ValueError("Model not built.")
        results = self.model.evaluate(X, y, verbose=0)
        return {'loss': results[0], 'mae': results[1]}
    
    def save(self, filepath: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.save(filepath)
        print(f"[INFO] Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model."""
        self.model = load_model(filepath)
        print(f"[INFO] Model loaded from {filepath}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get model config."""
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
    """Factory function to create and build LSTM model."""
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
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    sequence_length: int,
    n_features: int,
    lstm_layers: List[int] = [64, 32],
    dense_layers: List[int] = [32],
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    bidirectional: bool = False
) -> Tuple[LSTMModel, Dict[str, Any]]:
    """Complete training and evaluation pipeline."""
    model = LSTMModel(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_layers=lstm_layers,
        dense_layers=dense_layers,
        learning_rate=learning_rate,
        bidirectional=bidirectional
    )
    model.build()
    
    history = model.train(X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size)
    
    test_results = model.evaluate(X_test, y_test)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    results = {
        'model_type': 'LSTM',
        'config': model.get_config(),
        'history': history,
        'test_loss': test_results['loss'],
        'test_mae': test_results['mae'],
        'training_time': model.training_time,
        'predictions': {'train': y_pred_train, 'val': y_pred_val, 'test': y_pred_test}
    }
    
    print("\n" + "="*60)
    print("LSTM MODEL RESULTS")
    print("="*60)
    print(f"Test MSE: {test_results['loss']:.6f}")
    print(f"Test MAE: {test_results['mae']:.6f}")
    print(f"Training Time: {model.training_time:.2f}s")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    print("Testing LSTM model...")
    
    np.random.seed(42)
    n_samples, sequence_length, n_features = 1000, 24, 15
    
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.mean(X[:, -5:, :3], axis=(1, 2)) + np.random.randn(n_samples) * 0.1
    
    X_train, X_val, X_test = X[:700], X[700:850], X[850:]
    y_train, y_val, y_test = y[:700], y[700:850], y[850:]
    
    model = create_lstm_model(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_layers=[32, 16],
        dense_layers=[16]
    )
    model.train(X_train, y_train, X_val, y_val, epochs=30, verbose=0)
    
    results = model.evaluate(X_test, y_test)
    print(f"\nTest: Loss={results['loss']:.4f}, MAE={results['mae']:.4f}")
    print("\nLSTM model test completed!")
