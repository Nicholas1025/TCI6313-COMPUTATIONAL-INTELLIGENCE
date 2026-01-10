"""
evaluation.py - Model Evaluation Module for Air Quality Prediction

This module provides consistent evaluation metrics and result management
across all CI models (ANN, LSTM, GA-ANN).

Key Features:
- RMSE and MAE computation
- Training time measurement
- Result serialization (JSON export)
- Visualization helpers
- Statistical significance testing

Evaluation Metrics:
===================
1. RMSE (Root Mean Squared Error):
   - Penalizes large errors more heavily
   - Same units as target variable
   - Standard metric for regression

2. MAE (Mean Absolute Error):
   - More robust to outliers
   - Easier to interpret
   - Average absolute deviation

3. R² (Coefficient of Determination):
   - Proportion of variance explained
   - 1.0 = perfect prediction
   - Can be negative for poor models

Author: TCI6313 Student
Date: 2026
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
        
    Notes:
    ------
    RMSE is more sensitive to large errors than MAE.
    This can be good or bad depending on the application.
    For air quality, large errors might be more problematic,
    so RMSE is appropriate.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = mean(|y_true - y_pred|)
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Coefficient of Determination (R²).
    
    R² = 1 - SS_res / SS_tot
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        R² value (can be negative for poor models)
    """
    return r2_score(y_true, y_pred)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        MAPE value (percentage)
        
    Notes:
    ------
    MAPE is undefined when y_true contains zeros.
    We handle this by excluding zero values.
    """
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    training_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a model.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Name of the model
    training_time : float, optional
        Training time in seconds
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all metrics
    """
    metrics = {
        'model_name': model_name,
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred),
        'mape': compute_mape(y_true, y_pred),
        'n_samples': len(y_true),
        'training_time': training_time,
        'timestamp': datetime.now().isoformat()
    }
    
    return metrics


def compare_models(
    results_list: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Parameters:
    -----------
    results_list : List[Dict]
        List of evaluation results from evaluate_model()
        
    Returns:
    --------
    pd.DataFrame
        Comparison table sorted by RMSE
    """
    comparison_data = []
    
    for result in results_list:
        comparison_data.append({
            'Model': result['model_name'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'R²': result['r2'],
            'MAPE (%)': result.get('mape', np.nan),
            'Training Time (s)': result.get('training_time', np.nan)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('RMSE')
    
    return df


def save_results_to_json(
    results: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save evaluation results to JSON file.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary
    filepath : str
        Output file path
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results_clean = convert_numpy(results)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"[INFO] Results saved to {filepath}")


def load_results_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to JSON file
        
    Returns:
    --------
    Dict[str, Any]
        Loaded results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create actual vs predicted scatter plot.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name for title
    title : str, optional
        Custom title
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        rmse = compute_rmse(y_true, y_pred)
        ax.set_title(f'{model_name}: Actual vs Predicted (RMSE: {rmse:.4f})', fontsize=14)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Create residual analysis plots.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title(f'{model_name}: Residuals vs Predicted', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual Value', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'{model_name}: Residual Distribution', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot training history (loss curves).
    
    Parameters:
    -----------
    history : Dict[str, List[float]]
        Training history from model.fit()
    model_name : str
        Model name
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['loss'], label='Training Loss', lw=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', lw=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss (MSE)', fontsize=11)
    axes[0].set_title(f'{model_name}: Training Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot (if available)
    if 'mae' in history:
        axes[1].plot(history['mae'], label='Training MAE', lw=2)
        if 'val_mae' in history:
            axes[1].plot(history['val_mae'], label='Validation MAE', lw=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('MAE', fontsize=11)
        axes[1].set_title(f'{model_name}: Training MAE', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'MAE history not available',
                    ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'RMSE',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create bar chart comparing models.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison DataFrame from compare_models()
    metric : str
        Metric to plot ('RMSE', 'MAE', 'R²', or 'Training Time (s)')
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(comparison_df)))
    
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, comparison_df[metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison: {metric}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")
    
    return fig


def plot_predictions_timeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_points: int = 200,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot actual vs predicted over time (timeline view).
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name
    n_points : int
        Number of points to display
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use last n_points if more available
    if len(y_true) > n_points:
        y_true = y_true[-n_points:]
        y_pred = y_pred[-n_points:]
    
    x = np.arange(len(y_true))
    
    ax.plot(x, y_true, label='Actual', alpha=0.8, lw=1.5)
    ax.plot(x, y_pred, label='Predicted', alpha=0.8, lw=1.5)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'{model_name}: Predictions vs Actual Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")
    
    return fig


def create_comprehensive_report(
    all_results: Dict[str, Dict[str, Any]],
    save_dir: str = 'results'
) -> None:
    """
    Create a comprehensive evaluation report with all visualizations.
    
    Parameters:
    -----------
    all_results : Dict[str, Dict]
        Dictionary with model names as keys and result dictionaries as values
    save_dir : str
        Directory to save results
    """
    print("=" * 60)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect metrics for comparison
    metrics_list = []
    
    for model_name, results in all_results.items():
        # Evaluate test predictions
        y_true = results.get('y_test', None)
        y_pred = results.get('predictions', {}).get('test', None)
        
        if y_true is not None and y_pred is not None:
            metrics = evaluate_model(
                y_true, y_pred,
                model_name=model_name,
                training_time=results.get('training_time')
            )
            metrics_list.append(metrics)
            
            # Individual model plots
            plot_actual_vs_predicted(
                y_true, y_pred, model_name,
                save_path=os.path.join(save_dir, f'{model_name}_actual_vs_pred.png')
            )
            plt.close()
            
            plot_residuals(
                y_true, y_pred, model_name,
                save_path=os.path.join(save_dir, f'{model_name}_residuals.png')
            )
            plt.close()
            
            plot_predictions_timeline(
                y_true, y_pred, model_name,
                save_path=os.path.join(save_dir, f'{model_name}_timeline.png')
            )
            plt.close()
        
        # Training history plot
        if 'history' in results:
            plot_training_history(
                results['history'], model_name,
                save_path=os.path.join(save_dir, f'{model_name}_training_history.png')
            )
            plt.close()
    
    # Comparison DataFrame
    if metrics_list:
        comparison_df = compare_models(metrics_list)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        
        # Save comparison table
        comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
        
        # Comparison bar charts
        for metric in ['RMSE', 'MAE', 'R²']:
            plot_model_comparison(
                comparison_df, metric,
                save_path=os.path.join(save_dir, f'comparison_{metric.replace("²", "2")}.png')
            )
            plt.close()
        
        # Save all metrics to JSON
        save_results_to_json(
            {'comparison': comparison_df.to_dict(), 'individual_results': metrics_list},
            os.path.join(save_dir, 'metrics.json')
        )
    
    print("\n" + "=" * 60)
    print(f"REPORT GENERATED IN: {save_dir}")
    print("=" * 60)


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation summary.
    
    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics dictionary from evaluate_model()
    """
    print("\n" + "=" * 40)
    print(f"EVALUATION: {metrics['model_name']}")
    print("=" * 40)
    print(f"RMSE:         {metrics['rmse']:.6f}")
    print(f"MAE:          {metrics['mae']:.6f}")
    print(f"R²:           {metrics['r2']:.6f}")
    if metrics.get('mape') is not None and not np.isnan(metrics['mape']):
        print(f"MAPE:         {metrics['mape']:.2f}%")
    if metrics.get('training_time') is not None:
        print(f"Train Time:   {metrics['training_time']:.2f}s")
    print(f"Samples:      {metrics['n_samples']}")
    print("=" * 40)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing evaluation module...")
    
    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randn(n_samples) * 10 + 50
    
    # Simulate different model predictions
    y_pred_good = y_true + np.random.randn(n_samples) * 2
    y_pred_medium = y_true + np.random.randn(n_samples) * 5
    y_pred_poor = y_true + np.random.randn(n_samples) * 10
    
    # Evaluate models
    metrics_good = evaluate_model(y_true, y_pred_good, "Good Model", 10.5)
    metrics_medium = evaluate_model(y_true, y_pred_medium, "Medium Model", 8.2)
    metrics_poor = evaluate_model(y_true, y_pred_poor, "Poor Model", 5.1)
    
    # Print summaries
    print_evaluation_summary(metrics_good)
    print_evaluation_summary(metrics_medium)
    print_evaluation_summary(metrics_poor)
    
    # Compare models
    comparison = compare_models([metrics_good, metrics_medium, metrics_poor])
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    
    print("\nEvaluation module test completed successfully!")
