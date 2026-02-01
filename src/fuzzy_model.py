"""
Mamdani Fuzzy Inference System for air quality prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import time

warnings.filterwarnings('ignore')


class TriangularMF:
    """Triangular membership function."""
    
    def __init__(self, a: float, b: float, c: float, label: str):
        self.a, self.b, self.c = a, b, c
        self.label = label
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.maximum(0, np.minimum((x - self.a) / (self.b - self.a + 1e-10),
                                         (self.c - x) / (self.c - self.b + 1e-10)))
    
    def __repr__(self):
        return f"TriangularMF({self.label}: [{self.a}, {self.b}, {self.c}])"


class TrapezoidalMF:
    """Trapezoidal membership function."""
    
    def __init__(self, a: float, b: float, c: float, d: float, label: str):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.label = label
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.maximum(0, np.minimum(
            np.minimum((x - self.a) / (self.b - self.a + 1e-10), 1),
            (self.d - x) / (self.d - self.c + 1e-10)
        ))
    
    def __repr__(self):
        return f"TrapezoidalMF({self.label}: [{self.a}, {self.b}, {self.c}, {self.d}])"


class LinguisticVariable:
    """Linguistic variable with membership functions."""
    
    def __init__(self, name: str, universe: Tuple[float, float], mfs: Dict[str, callable] = None):
        self.name = name
        self.universe = universe
        self.mfs = mfs or {}
    
    def add_mf(self, label: str, mf: callable):
        self.mfs[label] = mf
    
    def fuzzify(self, x: float) -> Dict[str, float]:
        return {label: float(mf(x)) for label, mf in self.mfs.items()}
    
    def __repr__(self):
        return f"LinguisticVariable({self.name}, {list(self.mfs.keys())})"


class FuzzyRule:
    """Fuzzy IF-THEN rule."""
    
    def __init__(self, antecedent: Dict[str, str], consequent: Tuple[str, str], weight: float = 1.0):
        self.antecedent = antecedent
        self.consequent = consequent
        self.weight = weight
    
    def __repr__(self):
        ant_str = " AND ".join([f"{k} is {v}" for k, v in self.antecedent.items()])
        return f"IF {ant_str} THEN {self.consequent[0]} is {self.consequent[1]}"


class MamdaniFIS:
    """Mamdani-type Fuzzy Inference System."""
    
    def __init__(self, name: str = "FuzzySystem"):
        self.name = name
        self.input_variables: Dict[str, LinguisticVariable] = {}
        self.output_variable: Optional[LinguisticVariable] = None
        self.rules: List[FuzzyRule] = []
        self.inference_time = 0.0
    
    def add_input(self, variable: LinguisticVariable):
        self.input_variables[variable.name] = variable
    
    def set_output(self, variable: LinguisticVariable):
        self.output_variable = variable
    
    def add_rule(self, rule: FuzzyRule):
        self.rules.append(rule)
    
    def fuzzify_inputs(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                fuzzified[var_name] = self.input_variables[var_name].fuzzify(value)
        return fuzzified
    
    def evaluate_rule(self, rule: FuzzyRule, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """Evaluate rule using min (AND) operator."""
        firing_strength = 1.0
        for var_name, term in rule.antecedent.items():
            if var_name in fuzzified_inputs:
                membership = fuzzified_inputs[var_name].get(term, 0.0)
                firing_strength = min(firing_strength, membership)
        return firing_strength * rule.weight
    
    def aggregate(self, output_universe: np.ndarray, rule_outputs: List[Tuple[float, str]]) -> np.ndarray:
        """Aggregate rule outputs using max operator."""
        aggregated = np.zeros_like(output_universe)
        for firing_strength, output_term in rule_outputs:
            if firing_strength > 0 and output_term in self.output_variable.mfs:
                mf = self.output_variable.mfs[output_term]
                clipped = np.minimum(mf(output_universe), firing_strength)
                aggregated = np.maximum(aggregated, clipped)
        return aggregated
    
    def defuzzify_centroid(self, universe: np.ndarray, aggregated: np.ndarray) -> float:
        """Centroid defuzzification."""
        if np.sum(aggregated) == 0:
            return np.mean(universe)
        return np.sum(universe * aggregated) / np.sum(aggregated)
    
    def infer(self, inputs: Dict[str, float], resolution: int = 1000) -> Tuple[float, Dict]:
        """Perform fuzzy inference."""
        fuzzified = self.fuzzify_inputs(inputs)
        
        rule_outputs = []
        rule_details = []
        
        for rule in self.rules:
            strength = self.evaluate_rule(rule, fuzzified)
            rule_outputs.append((strength, rule.consequent[1]))
            rule_details.append({
                'rule': str(rule),
                'firing_strength': strength,
                'output_term': rule.consequent[1]
            })
        
        output_universe = np.linspace(
            self.output_variable.universe[0],
            self.output_variable.universe[1],
            resolution
        )
        
        aggregated = self.aggregate(output_universe, rule_outputs)
        output = self.defuzzify_centroid(output_universe, aggregated)
        
        return output, {
            'fuzzified_inputs': fuzzified,
            'rule_evaluations': rule_details,
            'aggregated_mf': aggregated.tolist(),
            'output_universe': output_universe.tolist()
        }
    
    def predict(self, X: np.ndarray, input_names: List[str] = None, verbose: bool = False) -> np.ndarray:
        """Predict for multiple samples."""
        if input_names is None:
            input_names = list(self.input_variables.keys())
        
        start_time = time.time()
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            inputs = {name: X[i, j] for j, name in enumerate(input_names)}
            predictions[i], _ = self.infer(inputs)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_samples}...")
        
        self.inference_time = time.time() - start_time
        if verbose:
            print(f"  Inference completed in {self.inference_time:.2f}s")
        
        return predictions
    
    def summary(self) -> str:
        lines = ["=" * 60, f"FUZZY SYSTEM: {self.name}", "=" * 60, "", "INPUT VARIABLES:", "-" * 40]
        
        for name, var in self.input_variables.items():
            lines.append(f"  {name}: {var.universe}, Terms: {list(var.mfs.keys())}")
        
        lines.extend(["", "OUTPUT VARIABLE:", "-" * 40,
            f"  {self.output_variable.name}: {self.output_variable.universe}",
            f"  Terms: {list(self.output_variable.mfs.keys())}",
            "", "RULES:", "-" * 40])
        
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"  R{i}: {rule}")
        
        lines.extend(["", "=" * 60])
        return "\n".join(lines)


def create_air_quality_fis(data_stats: Dict = None) -> MamdaniFIS:
    """Create FIS for C6H6 prediction using CO and NO2."""
    if data_stats is None:
        data_stats = {
            'CO_min': 0.0, 'CO_max': 12.0,
            'NO2_min': 0.0, 'NO2_max': 350.0,
            'C6H6_min': 0.0, 'C6H6_max': 65.0
        }
    
    fis = MamdaniFIS("AirQuality_C6H6_Predictor")
    
    # CO input
    co_min, co_max = data_stats['CO_min'], data_stats['CO_max']
    co_range = co_max - co_min
    co_var = LinguisticVariable("CO", (co_min, co_max))
    co_var.add_mf("Low", TriangularMF(co_min, co_min, co_min + 0.4 * co_range, "Low"))
    co_var.add_mf("Medium", TriangularMF(co_min + 0.2 * co_range, co_min + 0.5 * co_range, co_min + 0.8 * co_range, "Medium"))
    co_var.add_mf("High", TriangularMF(co_min + 0.6 * co_range, co_max, co_max, "High"))
    fis.add_input(co_var)
    
    # NO2 input
    no2_min, no2_max = data_stats['NO2_min'], data_stats['NO2_max']
    no2_range = no2_max - no2_min
    no2_var = LinguisticVariable("NO2", (no2_min, no2_max))
    no2_var.add_mf("Low", TriangularMF(no2_min, no2_min, no2_min + 0.4 * no2_range, "Low"))
    no2_var.add_mf("Medium", TriangularMF(no2_min + 0.2 * no2_range, no2_min + 0.5 * no2_range, no2_min + 0.8 * no2_range, "Medium"))
    no2_var.add_mf("High", TriangularMF(no2_min + 0.6 * no2_range, no2_max, no2_max, "High"))
    fis.add_input(no2_var)
    
    # C6H6 output
    c6h6_min, c6h6_max = data_stats['C6H6_min'], data_stats['C6H6_max']
    c6h6_range = c6h6_max - c6h6_min
    c6h6_var = LinguisticVariable("C6H6", (c6h6_min, c6h6_max))
    c6h6_var.add_mf("Low", TriangularMF(c6h6_min, c6h6_min, c6h6_min + 0.35 * c6h6_range, "Low"))
    c6h6_var.add_mf("Medium", TriangularMF(c6h6_min + 0.15 * c6h6_range, c6h6_min + 0.5 * c6h6_range, c6h6_min + 0.85 * c6h6_range, "Medium"))
    c6h6_var.add_mf("High", TriangularMF(c6h6_min + 0.65 * c6h6_range, c6h6_max, c6h6_max, "High"))
    fis.set_output(c6h6_var)
    
    # 9 fuzzy rules
    fis.add_rule(FuzzyRule({"CO": "Low", "NO2": "Low"}, ("C6H6", "Low")))
    fis.add_rule(FuzzyRule({"CO": "Low", "NO2": "Medium"}, ("C6H6", "Low")))
    fis.add_rule(FuzzyRule({"CO": "Low", "NO2": "High"}, ("C6H6", "Medium")))
    fis.add_rule(FuzzyRule({"CO": "Medium", "NO2": "Low"}, ("C6H6", "Low")))
    fis.add_rule(FuzzyRule({"CO": "Medium", "NO2": "Medium"}, ("C6H6", "Medium")))
    fis.add_rule(FuzzyRule({"CO": "Medium", "NO2": "High"}, ("C6H6", "High")))
    fis.add_rule(FuzzyRule({"CO": "High", "NO2": "Low"}, ("C6H6", "Medium")))
    fis.add_rule(FuzzyRule({"CO": "High", "NO2": "Medium"}, ("C6H6", "High")))
    fis.add_rule(FuzzyRule({"CO": "High", "NO2": "High"}, ("C6H6", "High")))
    
    return fis


def visualize_membership_functions(fis: MamdaniFIS, save_path: str = None):
    """Visualize membership functions."""
    import matplotlib.pyplot as plt
    
    n_vars = len(fis.input_variables) + 1
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    
    if n_vars == 1:
        axes = [axes]
    
    for idx, (name, var) in enumerate(fis.input_variables.items()):
        ax = axes[idx]
        x = np.linspace(var.universe[0], var.universe[1], 500)
        for label, mf in var.mfs.items():
            ax.plot(x, mf(x), label=label, linewidth=2)
        ax.set_title(f'Input: {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel(name)
        ax.set_ylabel('Membership')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
    
    ax = axes[-1]
    var = fis.output_variable
    x = np.linspace(var.universe[0], var.universe[1], 500)
    for label, mf in var.mfs.items():
        ax.plot(x, mf(x), label=label, linewidth=2)
    ax.set_title(f'Output: {var.name}', fontsize=12, fontweight='bold')
    ax.set_xlabel(var.name)
    ax.set_ylabel('Membership')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    
    plt.suptitle('Fuzzy Membership Functions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_rule_surface(fis: MamdaniFIS, resolution: int = 50, save_path: str = None):
    """Visualize FIS control surface."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    var_names = list(fis.input_variables.keys())
    if len(var_names) < 2:
        print("Need at least 2 input variables")
        return None
    
    var1, var2 = var_names[0], var_names[1]
    range1 = fis.input_variables[var1].universe
    range2 = fis.input_variables[var2].universe
    
    x1 = np.linspace(range1[0], range1[1], resolution)
    x2 = np.linspace(range2[0], range2[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)
    
    Z = np.zeros_like(X1)
    for i in range(resolution):
        for j in range(resolution):
            inputs = {var1: X1[i, j], var2: X2[i, j]}
            Z[i, j], _ = fis.infer(inputs, resolution=200)
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel(var1)
    ax1.set_ylabel(var2)
    ax1.set_zlabel(fis.output_variable.name)
    ax1.set_title('Control Surface', fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X1, X2, Z, levels=20, cmap='viridis')
    ax2.set_xlabel(var1)
    ax2.set_ylabel(var2)
    ax2.set_title('Contour', fontweight='bold')
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Testing Fuzzy Inference System...")
    
    fis = create_air_quality_fis()
    print(fis.summary())
    
    test_inputs = [
        {"CO": 2.0, "NO2": 100.0},
        {"CO": 5.0, "NO2": 175.0},
        {"CO": 10.0, "NO2": 300.0},
    ]
    
    print("\nTest Inferences:")
    print("-" * 40)
    for inputs in test_inputs:
        output, _ = fis.infer(inputs)
        print(f"Inputs: {inputs} -> C6H6: {output:.2f}")
