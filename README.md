# Air Quality Prediction using CI

This project implements various Computational Intelligence (CI) techniques to predict air quality (Benzene concentration) from the UCI Air Quality dataset.

## Methods Implemented
1. **ANN (Baseline)**: Feedforward Neural Network.
2. **LSTM**: Long Short-Term Memory network for time-series forecasting.
3. **GA-ANN (Hybrid)**: Genetic Algorithm optimized Neural Network.
4. **Fuzzy System**: Mamdani Fuzzy Inference System (Knowledge-based).

## Project Structure
- `data/`: Dataset storage
- `src/`: Python source code for models and preprocessing
- `notebooks/`: Jupyter notebooks for experiments
- `results/`: Model outputs, metrics, and plots
- `web_dashboard/`: Flask app to visualize results

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Models
You can run the modules directly to test training and evaluation:

```bash
# Clean and prepare data
python src/preprocessing.py

# Run ANN Baseline
python src/ann_model.py

# Run LSTM Model
python src/lstm_model.py

# Run GA Optimization
python src/ga_optimizer.py

# Run Fuzzy System
python src/fuzzy_model.py
```

### 2. Web Dashboard
Launch the dashboard to compare model performance:

```bash
python web_dashboard/app.py
```
Open http://127.0.0.1:5000 in your browser.

## Results
Results are saved in the `results/` folder as JSON and plots.
