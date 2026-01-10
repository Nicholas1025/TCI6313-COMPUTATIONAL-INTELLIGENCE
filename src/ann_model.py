"""
ann_model.py - Feedforward Artificial Neural Network Model

This module implements a configurable feedforward ANN for air quality prediction.
It serves as the BASELINE CI model for comparison with LSTM and GA-optimized variants.

Key Features:
- Configurable architecture (layers, neurons, activation)
- Explicit learning rate control
- Regularization options (dropout, L2)
- Training history tracking
- Model saving/loading

Computational Intelligence Context:
- ANNs are the foundation of modern CI
- Universal function approximators
- Gradient-based learning (backpropagation)
- This baseline helps quantify improvements from temporal modeling (LSTM)
  and hyperparameter optimization (GA)

Author: TCI6313 Student
Date: 2026
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
import warnings

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
except ImportError:
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class ANNModel:
    """
    Feedforward Artificial Neural Network for Air Quality Prediction.
    
    This class encapsulates the ANN architecture and training logic,
    providing a clean interface for the baseline CI model.
    
    Attributes:
    -----------
    n_features : int
        Number of input features
    hidden_layers : List[int]
        Number of neurons in each hidden layer
    learning_rate : float
        Learning rate for optimizer
    dropout_rate : float
        Dropout rate for regularization
    l2_reg : float
        L2 regularization strength
    model : keras.Model
        The compiled Keras model
    history : dict
        Training history
    training_time : float
        Time taken for training (seconds)
        
    Example:
    --------
    >>> model = ANNModel(n_features=15, hidden_layers=[64, 32], learning_rate=0.001)
    >>> model.build()
    >>> history = model.train(X_train, y_train, X_val, y_val, epochs=100)
    >>> predictions = model.predict(X_test)
    """
    
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
        """
        Initialize the ANN model configuration.
        
        Parameters:
        -----------
        n_features : int
            Number of input features
        hidden_layers : List[int]
            Number of neurons in each hidden layer.
            Example: [64, 32] creates two hidden layers with 64 and 32 neurons.
        learning_rate : float
            Learning rate for the optimizer. 
            Lower values = more stable but slower training.
            Typical range: 0.0001 to 0.01
        dropout_rate : float
            Fraction of neurons to drop during training (regularization).
            Helps prevent overfitting. Range: 0.0 to 0.5
        l2_reg : float
            L2 regularization strength. Penalizes large weights.
            Helps prevent overfitting. Typical: 0.0001 to 0.01
        activation : str
            Activation function for hidden layers.
            'relu' is standard for regression tasks.
        optimizer : str
            Optimizer type: 'adam', 'sgd', or 'rmsprop'
        random_seed : int
            Random seed for reproducibility
        """
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
        
        # Set seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    
    def build(self) -> None:
        """
        Build and compile the ANN model.
        
        Architecture:
        - Input layer: n_features neurons
        - Hidden layers: As specified, with ReLU activation
        - Dropout after each hidden layer
        - Output layer: 1 neuron (linear activation for regression)
        
        Notes:
        ------
        Why ReLU activation?
        - Avoids vanishing gradient problem
        - Computationally efficient
        - Works well for most regression tasks
        
        Why linear output activation?
        - Regression task requires unbounded continuous output
        - Softmax/Sigmoid would constrain output range
        """
        print("[INFO] Building ANN model...")
        
        # Initialize sequential model
        self.model = Sequential(name='ANN_Baseline')
        
        # Input layer
        self.model.add(Input(shape=(self.n_features,), name='input'))
        
        # Hidden layers
        for i, neurons in enumerate(self.hidden_layers):
            # Dense layer with L2 regularization
            self.model.add(Dense(
                neurons,
                activation=self.activation,
                kernel_regularizer=l2(self.l2_reg),
                name=f'hidden_{i+1}'
            ))
            
            # Batch normalization for stable training
            self.model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            
            # Dropout for regularization
            if self.dropout_rate > 0:
                self.model.add(Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer (linear activation for regression)
        self.model.add(Dense(1, activation='linear', name='output'))
        
        # Select optimizer
        optimizer = self._get_optimizer()
        
        # Compile model
        # MSE loss is standard for regression
        # MAE metric provides interpretable error in original units
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Print model summary
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        self.model.summary()
        print("="*60 + "\n")
    
    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """
        Get the configured optimizer.
        
        Returns:
        --------
        keras.optimizers.Optimizer
            Configured optimizer instance
        """
        if self.optimizer_name.lower() == 'adam':
            # Adam: Adaptive learning rate, good default choice
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            # SGD: Simple but may need momentum for good performance
            return SGD(learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer_name.lower() == 'rmsprop':
            # RMSprop: Good for recurrent networks, also works for feedforward
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
        """
        Train the ANN model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features)
        y_train : np.ndarray
            Training targets, shape (n_samples,)
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation targets
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Number of samples per gradient update
        early_stopping_patience : int
            Stop training if validation loss doesn't improve for this many epochs
        reduce_lr_patience : int
            Reduce learning rate if validation loss doesn't improve
        verbose : int
            Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history with loss and metrics
            
        Notes:
        ------
        Callbacks used:
        1. EarlyStopping: Prevents overfitting by stopping when val_loss plateaus
        2. ReduceLROnPlateau: Reduces learning rate when training stalls
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        print("[INFO] Starting training...")
        print(f"       Epochs: {epochs}, Batch size: {batch_size}")
        print(f"       Training samples: {len(X_train)}")
        print(f"       Validation samples: {len(X_val)}")
        
        # Define callbacks
        callbacks = [
            # Stop training when validation loss stops improving
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when training plateaus
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
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Record training time
        self.training_time = time.time() - start_time
        self.history = history.history
        
        print(f"\n[INFO] Training completed in {self.training_time:.2f} seconds")
        print(f"[INFO] Final training loss: {self.history['loss'][-1]:.6f}")
        print(f"[INFO] Final validation loss: {self.history['val_loss'][-1]:.6f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict on, shape (n_samples, n_features)
            
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
            Features
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
            Path to save the model (e.g., 'models/ann_model.keras')
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
        Get the model configuration as a dictionary.
        
        Returns:
        --------
        Dict[str, Any]
            Model configuration parameters
        """
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
    """
    Factory function to create and build an ANN model.
    
    This is a convenience function for quick model creation.
    
    Parameters:
    -----------
    n_features : int
        Number of input features
    hidden_layers : List[int]
        Neurons per hidden layer
    learning_rate : float
        Learning rate
    dropout_rate : float
        Dropout rate
        
    Returns:
    --------
    ANNModel
        Built and ready-to-train model
        
    Example:
    --------
    >>> model = create_ann_model(n_features=15, hidden_layers=[64, 32])
    >>> model.train(X_train, y_train, X_val, y_val)
    """
    model = ANNModel(
        n_features=n_features,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )
    model.build()
    return model


def train_and_evaluate_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layers: List[int] = [64, 32],
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32
) -> Tuple[ANNModel, Dict[str, Any]]:
    """
    Complete training and evaluation pipeline for ANN.
    
    This function provides a one-call solution for training and evaluating
    the ANN baseline model.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    X_test, y_test : Test data
    hidden_layers : Network architecture
    learning_rate : Learning rate
    epochs : Maximum epochs
    batch_size : Batch size
    
    Returns:
    --------
    Tuple[ANNModel, Dict[str, Any]]
        Trained model and results dictionary
    """
    n_features = X_train.shape[1]
    
    # Create and build model
    model = ANNModel(
        n_features=n_features,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate
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
        'model_type': 'ANN',
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
    print("ANN BASELINE RESULTS")
    print("="*60)
    print(f"Test MSE Loss: {test_results['loss']:.6f}")
    print(f"Test MAE: {test_results['mae']:.6f}")
    print(f"Training Time: {model.training_time:.2f} seconds")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ANN model module...")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1
    
    # Split data
    X_train, X_val, X_test = X[:700], X[700:850], X[850:]
    y_train, y_val, y_test = y[:700], y[700:850], y[850:]
    
    # Create and train model
    model = create_ann_model(n_features=n_features, hidden_layers=[32, 16])
    model.train(X_train, y_train, X_val, y_val, epochs=50, verbose=0)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    print(f"\nTest Results: Loss={results['loss']:.4f}, MAE={results['mae']:.4f}")
    print("\nANN model module test completed successfully!")
