"""
Web Dashboard for Air Quality Prediction
TCI6313 Computational Intelligence Project

Flask application for visualizing and comparing model results.
"""

from flask import Flask, render_template, jsonify, request
import json
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def load_json_results(filename):
    """Load results from JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def get_all_model_results():
    """Load all model results."""
    results = {}
    
    # ANN Baseline
    ann = load_json_results('ann_baseline_results.json')
    if ann:
        results['ann'] = {
            'name': 'ANN Baseline',
            'type': 'Feedforward Neural Network',
            'metrics': ann['metrics']['test'],
            'training_time': ann['training_time'],
            'config': ann['model_config'],
            'predictions': {
                'actual': ann['predictions']['test_actual'][-100:],
                'predicted': ann['predictions']['test_predicted'][-100:]
            }
        }
    
    # LSTM
    lstm = load_json_results('lstm_results.json')
    if lstm:
        results['lstm'] = {
            'name': 'LSTM',
            'type': 'Long Short-Term Memory',
            'metrics': lstm['metrics']['test'],
            'training_time': lstm['training_time'],
            'config': lstm['model_config'],
            'predictions': {
                'actual': lstm['predictions']['test_actual'][-100:],
                'predicted': lstm['predictions']['test_predicted'][-100:]
            }
        }
    
    # GA-ANN
    ga = load_json_results('ga_ann_results.json')
    if ga:
        results['ga_ann'] = {
            'name': 'GA-ANN (Hybrid)',
            'type': 'Genetic Algorithm + ANN',
            'metrics': ga['metrics']['test'],
            'training_time': ga['total_time'],
            'config': ga['best_params'],
            'ga_info': ga.get('ga_info', {}),
            'predictions': {
                'actual': ga['predictions']['test_actual'][-100:],
                'predicted': ga['predictions']['test_predicted'][-100:]
            }
        }
    
    return results


@app.route('/')
def index():
    """Render main dashboard."""
    results = get_all_model_results()
    return render_template('index.html', results=results)


@app.route('/api/results')
def api_results():
    """API endpoint for all results."""
    results = get_all_model_results()
    return jsonify(results)


@app.route('/api/model/<model_name>')
def api_model(model_name):
    """API endpoint for specific model results."""
    results = get_all_model_results()
    if model_name in results:
        return jsonify(results[model_name])
    return jsonify({'error': 'Model not found'}), 404


@app.route('/api/comparison')
def api_comparison():
    """API endpoint for model comparison data."""
    results = get_all_model_results()
    
    comparison = []
    for key, model in results.items():
        comparison.append({
            'model': model['name'],
            'rmse': model['metrics']['rmse'],
            'mae': model['metrics']['mae'],
            'r2': model['metrics']['r2'],
            'time': model['training_time']
        })
    
    # Sort by RMSE
    comparison.sort(key=lambda x: x['rmse'])
    
    return jsonify(comparison)


@app.route('/api/predictions/<model_name>')
def api_predictions(model_name):
    """API endpoint for model predictions."""
    results = get_all_model_results()
    if model_name in results:
        return jsonify({
            'actual': results[model_name]['predictions']['actual'],
            'predicted': results[model_name]['predictions']['predicted']
        })
    return jsonify({'error': 'Model not found'}), 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Air Quality Prediction Dashboard")
    print("TCI6313 Computational Intelligence")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
