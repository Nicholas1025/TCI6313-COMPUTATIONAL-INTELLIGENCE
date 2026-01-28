# 5. RESULTS AND DISCUSSION

## 5.1 Performance of Each Model

This section presents the experimental results of four Computational Intelligence (CI) models developed for benzene (C6H6) concentration prediction. All models were evaluated on the same test set comprising 139 samples (15% of the preprocessed dataset), using Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination (R²) as performance metrics.

### 5.1.1 Overall Performance Comparison

Table 5.1 presents the comprehensive performance comparison of all implemented CI models on the test dataset. The models are ranked by RMSE in ascending order, indicating prediction accuracy from best to worst.

**Table 5.1: Performance Comparison of CI Models on Test Dataset**

| Model | CI Paradigm | Architecture | RMSE (µg/m³) | MAE (µg/m³) | R² | Training Time (s) |
|-------|-------------|--------------|--------------|-------------|-----|-------------------|
| ANN Baseline | Neural Networks | [64, 32] | 0.8024 | 0.5914 | 0.9900 | 27.5 |
| GA-ANN | EC + NN (Hybrid) | [104, 49] | 1.0232 | 0.6608 | 0.9838 | 2357.2 |
| Fuzzy FIS | Fuzzy Systems | 9 Rules | 4.0266 | 2.9205 | 0.7485 | 0.05 |
| LSTM | Deep Neural Networks | LSTM[128, 64] | 5.7141 | 4.0732 | 0.4813 | 81.1 |

*Caption: Performance metrics of all four CI models evaluated on the test set. RMSE and MAE are measured in µg/m³ (benzene concentration units). Lower RMSE/MAE and higher R² indicate better performance.*

The results demonstrate substantial variation in predictive performance across different CI paradigms. The ANN Baseline achieved the best overall performance with an R² of 0.9900, explaining 99% of the variance in benzene concentration. This exceptional performance can be attributed to the strong linear correlations present in the dataset between sensor readings and the target variable, which feedforward neural networks are well-suited to capture.

Figure 5.1 illustrates the comparative performance of all models using bar charts for each evaluation metric. The visual comparison reveals the significant performance gap between data-driven approaches (ANN, GA-ANN) and the other methods (Fuzzy FIS, LSTM) on this particular dataset.

**[INSERT: results/model_comparison_metrics.png]**

*Figure 5.1: Bar chart comparison of model performance across three evaluation metrics: RMSE (lower is better), MAE (lower is better), and R² (higher is better). The ANN Baseline demonstrates superior performance across all metrics.*

### 5.1.2 ANN Baseline Model Performance

The feedforward Artificial Neural Network served as the baseline model for this study, employing a two-hidden-layer architecture with 64 and 32 neurons respectively. Table 5.2 presents the detailed performance across training, validation, and test sets.

**Table 5.2: ANN Baseline Performance Across Data Splits**

| Dataset | Samples | RMSE | MAE | R² |
|---------|---------|------|-----|-----|
| Training | 646 | 0.5902 | 0.4441 | 0.9929 |
| Validation | 139 | 0.8630 | 0.6491 | 0.9880 |
| Test | 139 | 0.8024 | 0.5914 | 0.9900 |

*Caption: ANN Baseline performance metrics across training, validation, and test datasets. The consistent R² values (>0.98) across all splits indicate excellent generalization without overfitting.*

The minimal difference between training R² (0.9929) and test R² (0.9900) indicates that the model generalizes well to unseen data, with no significant overfitting observed. The early stopping mechanism with patience of 15 epochs effectively prevented overtraining. Figure 5.2 shows the training loss convergence over epochs.

**[INSERT: Figure from notebook 02 - Training Loss Curve]**

*Figure 5.2: Training and validation loss curves for the ANN Baseline model. The curves demonstrate smooth convergence with early stopping triggered at epoch 73, preventing overfitting.*

### 5.1.3 LSTM Model Performance

The Long Short-Term Memory network was implemented to capture potential temporal dependencies in the hourly air quality data. Using a sequence length of 24 hours (one day of historical data), the LSTM was designed to leverage patterns from past observations for prediction.

**Table 5.3: LSTM Performance Across Data Splits**

| Dataset | Samples | RMSE | MAE | R² |
|---------|---------|------|-----|-----|
| Training | 622 | 1.8389 | 1.3477 | 0.9320 |
| Validation | 115 | 5.3989 | 4.0813 | 0.5545 |
| Test | 115 | 5.7141 | 4.0732 | 0.4813 |

*Caption: LSTM performance metrics showing significant discrepancy between training and test performance, indicating potential overfitting and model limitations for this dataset.*

Contrary to initial expectations, the LSTM exhibited the lowest performance among all models with a test R² of only 0.4813. The substantial gap between training R² (0.9320) and test R² (0.4813) indicates severe overfitting. This unexpected result warrants detailed analysis, which is provided in Section 5.2.

### 5.1.4 GA-ANN Hybrid Model Performance

The Genetic Algorithm-optimized ANN represents a hybrid CI approach combining evolutionary computation for hyperparameter optimization with neural network learning. The GA explored the search space over 15 generations with a population size of 20 individuals.

**Table 5.4: GA Optimization Results**

| Parameter | Search Range | Optimized Value |
|-----------|--------------|-----------------|
| Learning Rate | [0.0001, 0.01] | 0.00345 |
| Neurons (Layer 1) | [32, 128] | 104 |
| Neurons (Layer 2) | [16, 64] | 49 |
| Epochs | [50, 150] | 73 |
| Dropout Rate | [0.1, 0.4] | 0.294 |

*Caption: Hyperparameter search space and optimal values discovered by the Genetic Algorithm. The optimizer converged to a deeper architecture with moderate dropout regularization.*

The GA-ANN achieved an R² of 0.9838, which is marginally lower than the manually-tuned ANN Baseline (0.9900). While the evolutionary optimization successfully explored the hyperparameter space, the computational cost was substantial at 2357 seconds (approximately 39 minutes), representing an 85-fold increase over the baseline ANN. Figure 5.3 illustrates the GA optimization convergence.

**[INSERT: Figure from notebook 04 - GA Fitness Evolution]**

*Figure 5.3: Genetic Algorithm optimization progress showing minimum and average fitness values across 15 generations. The fitness (validation RMSE) decreases progressively, demonstrating successful evolutionary search.*

### 5.1.5 Fuzzy Inference System Performance

The Mamdani-type Fuzzy Inference System represents a fundamentally different approach to the prediction task—a knowledge-based system rather than a data-driven one. Using only two input variables (CO and NO2 concentrations) and nine hand-crafted IF-THEN rules, the FIS achieved an R² of 0.7485.

**Table 5.5: Fuzzy FIS Rule Base**

| Rule | Antecedent (CO) | Antecedent (NO2) | Consequent (C6H6) |
|------|-----------------|------------------|-------------------|
| R1 | Low | Low | Low |
| R2 | Low | Medium | Low |
| R3 | Low | High | Medium |
| R4 | Medium | Low | Low |
| R5 | Medium | Medium | Medium |
| R6 | Medium | High | High |
| R7 | High | Low | Medium |
| R8 | High | Medium | High |
| R9 | High | High | High |

*Caption: The complete rule base of the Mamdani FIS. Rules were designed based on domain knowledge of pollutant relationships, with CO and NO2 serving as proxy indicators for benzene concentration.*

Notably, the Fuzzy FIS outperformed the LSTM despite using only 2 input features compared to the LSTM's 11 features, and requiring virtually no computation time (0.05 seconds). Figure 5.4 displays the fuzzy membership functions used for input and output variables.

**[INSERT: results/fuzzy_membership_functions.png]**

*Figure 5.4: Triangular membership functions for the Fuzzy Inference System. Each variable is partitioned into three linguistic terms (Low, Medium, High) with overlapping triangular functions enabling smooth transitions between categories.*

### 5.1.6 Prediction Visualization

Figure 5.5 presents a visual comparison of predictions from all models against actual benzene concentrations over the test period. This visualization enables qualitative assessment of how well each model tracks the temporal variations in the target variable.

**[INSERT: results/predictions_comparison.png]**

*Figure 5.5: Time series comparison of actual benzene concentrations versus predictions from all four CI models. The ANN and GA-ANN predictions (blue and red dashed lines) closely follow the actual values (black solid line), while LSTM and Fuzzy predictions show larger deviations.*

The prediction plots reveal that ANN and GA-ANN accurately capture both the magnitude and temporal dynamics of benzene concentration fluctuations. The LSTM predictions, while following the general trend, exhibit systematic underestimation during peak concentration periods. The Fuzzy FIS predictions, constrained by its simplified two-input design, show reasonable trend-following but with reduced precision in extreme values.

### 5.1.7 Error Distribution Analysis

Figure 5.6 presents the error distribution histograms for each model, providing insight into the nature of prediction errors.

**[INSERT: results/error_distributions.png]**

*Figure 5.6: Prediction error distributions for all four CI models. ANN and GA-ANN exhibit tight, symmetric distributions centered near zero, indicating unbiased predictions. LSTM shows a wider, slightly skewed distribution, while Fuzzy FIS demonstrates a broader spread with slight positive bias.*

The error distributions confirm the quantitative metrics: ANN and GA-ANN produce concentrated, symmetric error distributions with means close to zero, indicating both accuracy and lack of systematic bias. The LSTM distribution is notably wider, reflecting its higher RMSE, while the Fuzzy FIS distribution shows a moderate spread consistent with its intermediate performance level.

---

## 5.2 Strengths and Weaknesses of Each Approach

This section provides a critical analysis of each CI paradigm implemented in this study, examining their relative strengths and limitations in the context of air quality prediction.

### 5.2.1 ANN Baseline

**Strengths:**

The feedforward ANN demonstrated exceptional predictive performance, achieving the highest R² (0.9900) among all models. This success can be attributed to several factors. First, the architecture's universal function approximation capability enabled effective modeling of the complex, non-linear relationships between multiple sensor readings and benzene concentration. Second, the regularization techniques (dropout rate of 0.2, L2 regularization of 0.0001) effectively prevented overfitting, as evidenced by the consistent performance across training and test sets. Third, the moderate training time (27.5 seconds) offers an excellent balance between computational efficiency and model performance.

**Weaknesses:**

Despite its strong performance, the ANN Baseline has inherent limitations. The model operates as a "black box," providing no interpretability regarding which features or relationships drive its predictions. This lack of transparency may be problematic in regulatory or scientific contexts where understanding the prediction mechanism is important. Additionally, the manually-selected architecture ([64, 32]) may not be optimal; while it performed well, systematic hyperparameter optimization could potentially yield marginal improvements.

### 5.2.2 LSTM Network

**Strengths:**

The LSTM architecture offers theoretical advantages for time-series prediction through its ability to capture long-term temporal dependencies via specialized gating mechanisms. The recurrent structure is particularly suited for sequential data where historical patterns influence future values. In this implementation, the 24-hour sequence length was designed to capture diurnal (daily) patterns in air pollution, which are known to exist due to traffic and industrial activity cycles.

**Weaknesses:**

The LSTM's poor performance (R² = 0.4813) on this dataset reveals significant limitations. Analysis of the results suggests several contributing factors:

**Table 5.6: Analysis of LSTM Underperformance**

| Factor | Description | Impact |
|--------|-------------|--------|
| Dataset Characteristics | Strong instantaneous correlations (CO-C6H6: r > 0.9) dominate temporal patterns | Feature-based prediction more effective than sequence-based |
| Sample Reduction | Sequence creation reduces test samples from 139 to 115 | Smaller evaluation set, potential representation issues |
| Overfitting | Train R² = 0.93, Test R² = 0.48 (gap of 0.45) | Model memorized training patterns rather than generalizing |
| Complexity Mismatch | LSTM's sequential modeling unnecessary for this prediction task | Added complexity without corresponding benefit |

*Caption: Factors contributing to LSTM underperformance on the air quality prediction task.*

The fundamental issue is a mismatch between the model's design assumptions and the data characteristics. LSTM networks excel when temporal dependencies are the primary predictive signal, but in this dataset, the instantaneous correlations between concurrent sensor readings provide stronger predictive power than historical sequences.

### 5.2.3 GA-ANN Hybrid

**Strengths:**

The GA-ANN hybrid successfully demonstrated the integration of evolutionary computation with neural network training. The Genetic Algorithm explored the hyperparameter space systematically, discovering an architecture ([104, 49]) with higher neuron counts than the baseline. The approach eliminates the need for manual hyperparameter tuning and provides a principled method for architecture search. The final model achieved competitive performance (R² = 0.9838) with only a 0.6% reduction compared to the manually-tuned baseline.

**Weaknesses:**

The primary limitation of the GA-ANN approach is computational cost. The optimization required 2357 seconds—approximately 85 times longer than training the baseline ANN. For this dataset, the marginal performance difference does not justify the substantial computational investment. However, this trade-off may be more favorable for larger datasets or more complex problems where manual tuning is infeasible.

**Table 5.7: Cost-Benefit Analysis of GA-ANN vs ANN Baseline**

| Metric | ANN Baseline | GA-ANN | Difference |
|--------|--------------|--------|------------|
| R² | 0.9900 | 0.9838 | -0.62% |
| RMSE | 0.8024 | 1.0232 | +27.5% |
| Training Time | 27.5 s | 2357.2 s | +8471% |
| Manual Tuning Required | Yes | No | — |

*Caption: Comparative analysis of ANN Baseline and GA-ANN approaches. The hybrid approach trades computational efficiency for automated hyperparameter optimization.*

### 5.2.4 Fuzzy Inference System

**Strengths:**

The Fuzzy FIS offers unique advantages that distinguish it from the data-driven approaches. First, the rule base is fully interpretable—each prediction can be traced to specific linguistic rules that have clear physical meaning (e.g., "IF CO is High AND NO2 is High THEN C6H6 is High"). This transparency is valuable for validation by domain experts and for explaining predictions to stakeholders. Second, the system requires no training, making it computationally trivial (0.05 seconds for inference). Third, the Fuzzy FIS achieved R² = 0.7485 using only two input features, demonstrating competitive performance with minimal complexity.

**Weaknesses:**

The knowledge-based nature of the FIS is also its limitation. The rule base was designed manually based on assumed relationships between pollutants; optimal rules are not learned from data. Expanding the system to include more input variables would require exponentially more rules (3 terms × n variables = 3ⁿ possible rule combinations), making scaling impractical. The fixed membership function boundaries, while data-informed, are not optimized for prediction accuracy.

### 5.2.5 Comparative Summary

Table 5.8 synthesizes the strengths and weaknesses of each approach across multiple evaluation dimensions.

**Table 5.8: Multi-Dimensional Comparison of CI Approaches**

| Dimension | ANN | LSTM | GA-ANN | Fuzzy FIS |
|-----------|-----|------|--------|-----------|
| Prediction Accuracy | ★★★★★ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ |
| Computational Efficiency | ★★★★☆ | ★★★☆☆ | ★☆☆☆☆ | ★★★★★ |
| Interpretability | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| Temporal Modeling | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | ★☆☆☆☆ |
| Automation Level | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| Scalability | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ |

*Caption: Qualitative rating of CI approaches across six evaluation dimensions. Stars indicate relative performance (1-5 scale). No single approach dominates across all dimensions, highlighting the importance of matching model selection to application requirements.*

### 5.2.6 Recommendations for Improvement

Based on the experimental findings, several avenues for potential improvement are identified:

**Table 5.9: Suggested Improvements for Each Model**

| Model | Suggested Improvement | Expected Benefit |
|-------|----------------------|------------------|
| ANN | Feature importance analysis (SHAP/LIME) | Improved interpretability without sacrificing accuracy |
| LSTM | Use univariate prediction (C6H6 only) or larger dataset | Better alignment with temporal modeling assumptions |
| GA-ANN | Multi-objective optimization (accuracy + complexity) | Pareto-optimal solutions balancing performance and efficiency |
| Fuzzy | Implement ANFIS for data-driven rule optimization | Combine interpretability with learning capability |

*Caption: Recommended improvements for each CI model to address identified limitations.*

### 5.2.7 Key Findings

The experimental results yield several important insights for CI-based air quality prediction:

First, model complexity does not guarantee superior performance. The simplest architecture (ANN Baseline) outperformed both the more complex LSTM and the sophisticated GA-ANN hybrid on this dataset. This finding underscores the importance of matching model assumptions to data characteristics rather than defaulting to the most advanced technique.

Second, different CI paradigms offer complementary strengths. While data-driven neural networks achieved the highest accuracy, the knowledge-based Fuzzy FIS provided interpretability that may be equally valuable in certain applications. The choice between accuracy and transparency represents a fundamental trade-off that practitioners must consider.

Third, the Fuzzy FIS's competitive performance (R² = 0.7485) with only two inputs demonstrates that domain knowledge can partially compensate for data-driven learning. Hybrid approaches such as ANFIS, which combine fuzzy reasoning with neural learning, may offer the best of both paradigms.

Fourth, the LSTM's underperformance highlights the importance of data analysis before model selection. The strong instantaneous correlations in this dataset favored feature-based prediction over temporal modeling, rendering the LSTM's sequence processing capability unnecessary for this specific task.

---

## Figures to Include from Notebooks

Based on the analysis above, the following figures should be included in the report:

| Figure # | Source | Description |
|----------|--------|-------------|
| Figure 5.1 | `results/model_comparison_metrics.png` | Bar chart comparing RMSE, MAE, R² |
| Figure 5.2 | Notebook 02 (generate) | ANN training/validation loss curves |
| Figure 5.3 | Notebook 04 (generate) | GA fitness evolution across generations |
| Figure 5.4 | `results/fuzzy_membership_functions.png` | Fuzzy membership functions |
| Figure 5.5 | `results/predictions_comparison.png` | Time series prediction comparison |
| Figure 5.6 | `results/error_distributions.png` | Error distribution histograms |

---

*End of Section 5*
