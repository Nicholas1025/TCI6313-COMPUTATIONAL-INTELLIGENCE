"""
Feedforward ANN for air quality prediction - baseline CI model.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class ANNModel:
    """Feedforward ANN for regression tasks."""
    
    def __init__(
        self,
        n_features: int,
        hidden_layers: List[int] = [64, 32],
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        l2_reg: float = 0.0001,
        activation: str = 'relu',
        optimizer: str = 'adam',
        random_seed: int = 42
    ):
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.activation = activation
        self.optimizer_name = optimizer
        self.random_seed = random_seed
        
        self.model = None
        self.history = None
        self.training_time = None
        
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    
    def build(self) -> None:
        """Build and compile the model."""
        print("[INFO] Building ANN model...")
        
        self.model = Sequential(name='ANN_Baseline')
        self.model.add(Input(shape=(self.n_features,), name='input'))
        
        for i, neurons in enumerate(self.hidden_layers):
            self.model.add(Dense(neurons, activation=self.activation,
                kernel_regularizer=l2(self.l2_reg), name=f'hidden_{i+1}'))
            self.model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            if self.dropout_rate > 0:
                self.model.add(Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        self.model.add(Dense(1, activation='linear', name='output'))
        
        optimizer = self._get_optimizer()
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        self.model.summary()
        print("="*60 + "\n")
    
    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get configured optimizer."""
        if self.optimizer_name.lower() == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            return SGD(learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer_name.lower() == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
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
        
        print("[INFO] Starting training...")
        print(f"       Epochs: {epochs}, Batch size: {batch_size}")
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
            'n_features': self.n_features,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'activation': self.activation,
            'optimizer': self.optimizer_name,
            'training_time': self.training_time
        }


def create_ann_model(
    n_features: int,
    hidden_layers: List[int] = [64, 32],
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2
) -> ANNModel:
    """Factory function to create and build ANN model."""
    model = ANNModel(
        n_features=n_features,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )
    model.build()
    return model


def train_and_evaluate_ann(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_layers: List[int] = [64, 32],
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32
) -> Tuple[ANNModel, Dict[str, Any]]:
    """Complete training and evaluation pipeline."""
    n_features = X_train.shape[1]
    
    model = ANNModel(n_features=n_features, hidden_layers=hidden_layers,
        learning_rate=learning_rate)
    model.build()
    
    history = model.train(X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size)
    
    test_results = model.evaluate(X_test, y_test)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    results = {
        'model_type': 'ANN',
        'config': model.get_config(),
        'history': history,
        'test_loss': test_results['loss'],
        'test_mae': test_results['mae'],
        'training_time': model.training_time,
        'predictions': {'train': y_pred_train, 'val': y_pred_val, 'test': y_pred_test}
    }
    
    print("\n" + "="*60)
    print("ANN BASELINE RESULTS")
    print("="*60)
    print(f"Test MSE: {test_results['loss']:.6f}")
    print(f"Test MAE: {test_results['mae']:.6f}")
    print(f"Training Time: {model.training_time:.2f}s")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    print("Testing ANN model...")
    
    np.random.seed(42)
    n_samples, n_features = 1000, 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1
    
    X_train, X_val, X_test = X[:700], X[700:850], X[850:]
    y_train, y_val, y_test = y[:700], y[700:850], y[850:]
    
    model = create_ann_model(n_features=n_features, hidden_layers=[32, 16])
    model.train(X_train, y_train, X_val, y_val, epochs=50, verbose=0)
    
    results = model.evaluate(X_test, y_test)
    print(f"\nTest: Loss={results['loss']:.4f}, MAE={results['mae']:.4f}")
    print("\nANN model test completed!")
