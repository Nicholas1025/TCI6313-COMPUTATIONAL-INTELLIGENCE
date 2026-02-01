"""
Model evaluation metrics and visualization.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE."""
    return mean_absolute_error(y_true, y_pred)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R²."""
    return r2_score(y_true, y_pred)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE (%)."""
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
    """Compute all evaluation metrics."""
    return {
        'model_name': model_name,
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred),
        'mape': compute_mape(y_true, y_pred),
        'n_samples': len(y_true),
        'training_time': training_time,
        'timestamp': datetime.now().isoformat()
    }


def compare_models(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compare multiple models' performance."""
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
    return df.sort_values('RMSE')


def save_results_to_json(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON."""
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"[INFO] Results saved to {filepath}")


def load_results_from_json(filepath: str) -> Dict[str, Any]:
    """Load results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Create actual vs predicted scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, label='Predictions')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    
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
        print(f"[INFO] Saved to {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Create residual plots."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name}: Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name}: Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Plot training history (loss curves)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].plot(history['loss'], label='Train Loss', lw=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title(f'{model_name}: Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if 'mae' in history:
        axes[1].plot(history['mae'], label='Train MAE', lw=2)
        if 'val_mae' in history:
            axes[1].plot(history['val_mae'], label='Val MAE', lw=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'{model_name}: Training MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'MAE not available', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'RMSE',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Create bar chart comparing models."""
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(comparison_df)))
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors, edgecolor='black')
    
    for bar, value in zip(bars, comparison_df[metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.4f}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    ax.set_title(f'Model Comparison: {metric}')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predictions_timeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_points: int = 200,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Plot actual vs predicted over time."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(y_true) > n_points:
        y_true = y_true[-n_points:]
        y_pred = y_pred[-n_points:]
    
    x = np.arange(len(y_true))
    
    ax.plot(x, y_true, label='Actual', alpha=0.8, lw=1.5)
    ax.plot(x, y_pred, label='Predicted', alpha=0.8, lw=1.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(f'{model_name}: Predictions Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comprehensive_report(all_results: Dict[str, Dict[str, Any]], save_dir: str = 'results') -> None:
    """Create comprehensive evaluation report."""
    print("=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    metrics_list = []
    
    for model_name, results in all_results.items():
        y_true = results.get('y_test', None)
        y_pred = results.get('predictions', {}).get('test', None)
        
        if y_true is not None and y_pred is not None:
            metrics = evaluate_model(y_true, y_pred, model_name=model_name,
                training_time=results.get('training_time'))
            metrics_list.append(metrics)
            
            plot_actual_vs_predicted(y_true, y_pred, model_name,
                save_path=os.path.join(save_dir, f'{model_name}_actual_vs_pred.png'))
            plt.close()
            
            plot_residuals(y_true, y_pred, model_name,
                save_path=os.path.join(save_dir, f'{model_name}_residuals.png'))
            plt.close()
            
            plot_predictions_timeline(y_true, y_pred, model_name,
                save_path=os.path.join(save_dir, f'{model_name}_timeline.png'))
            plt.close()
        
        if 'history' in results:
            plot_training_history(results['history'], model_name,
                save_path=os.path.join(save_dir, f'{model_name}_training_history.png'))
            plt.close()
    
    if metrics_list:
        comparison_df = compare_models(metrics_list)
        print("\nMODEL COMPARISON:")
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
        
        for metric in ['RMSE', 'MAE', 'R²']:
            plot_model_comparison(comparison_df, metric,
                save_path=os.path.join(save_dir, f'comparison_{metric.replace("²", "2")}.png'))
            plt.close()
        
        save_results_to_json(
            {'comparison': comparison_df.to_dict(), 'individual_results': metrics_list},
            os.path.join(save_dir, 'metrics.json')
        )
    
    print(f"\nReport saved to: {save_dir}")
    print("=" * 60)


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 40)
    print(f"EVALUATION: {metrics['model_name']}")
    print("=" * 40)
    print(f"RMSE:      {metrics['rmse']:.6f}")
    print(f"MAE:       {metrics['mae']:.6f}")
    print(f"R²:        {metrics['r2']:.6f}")
    if metrics.get('mape') is not None and not np.isnan(metrics['mape']):
        print(f"MAPE:      {metrics['mape']:.2f}%")
    if metrics.get('training_time') is not None:
        print(f"Time:      {metrics['training_time']:.2f}s")
    print(f"Samples:   {metrics['n_samples']}")
    print("=" * 40)


if __name__ == "__main__":
    print("Testing evaluation module...")
    
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randn(n_samples) * 10 + 50
    y_pred_good = y_true + np.random.randn(n_samples) * 2
    y_pred_medium = y_true + np.random.randn(n_samples) * 5
    y_pred_poor = y_true + np.random.randn(n_samples) * 10
    
    metrics_good = evaluate_model(y_true, y_pred_good, "Good Model", 10.5)
    metrics_medium = evaluate_model(y_true, y_pred_medium, "Medium Model", 8.2)
    metrics_poor = evaluate_model(y_true, y_pred_poor, "Poor Model", 5.1)
    
    print_evaluation_summary(metrics_good)
    print_evaluation_summary(metrics_medium)
    print_evaluation_summary(metrics_poor)
    
    comparison = compare_models([metrics_good, metrics_medium, metrics_poor])
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    
    print("\nEvaluation module test completed!")
