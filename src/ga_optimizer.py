"""
GA optimizer for ANN hyperparameter tuning using DEAP.
Hybrid CI approach: Genetic Algorithm + Neural Networks
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import warnings

from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


@dataclass
class HyperparameterBounds:
    """Search space bounds for GA optimization."""
    learning_rate_bounds: Tuple[float, float] = (0.0001, 0.01)
    neurons_layer1_bounds: Tuple[int, int] = (32, 128)
    neurons_layer2_bounds: Tuple[int, int] = (16, 64)
    epochs_bounds: Tuple[int, int] = (50, 150)
    dropout_bounds: Tuple[float, float] = (0.1, 0.4)


class GeneticANNOptimizer:
    """GA optimizer to find optimal ANN hyperparameters."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        bounds: Optional[HyperparameterBounds] = None,
        population_size: int = 20,
        n_generations: int = 15,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
        random_seed: int = 42
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_features = X_train.shape[1]
        
        self.bounds = bounds if bounds else HyperparameterBounds()
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.random_seed = random_seed
        
        self.best_individual = None
        self.best_fitness = float('inf')
        self.optimization_history = []
        self.total_time = None
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        self._setup_deap()
    
    def _setup_deap(self) -> None:
        """Setup DEAP genetic algorithm framework."""
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual
            
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Chromosome: [learning_rate_log, neurons1, neurons2, epochs, dropout]
        self.toolbox.register("attr_lr", random.uniform,
            np.log10(self.bounds.learning_rate_bounds[0]),
            np.log10(self.bounds.learning_rate_bounds[1]))
        self.toolbox.register("attr_neurons1", random.randint,
            self.bounds.neurons_layer1_bounds[0], self.bounds.neurons_layer1_bounds[1])
        self.toolbox.register("attr_neurons2", random.randint,
            self.bounds.neurons_layer2_bounds[0], self.bounds.neurons_layer2_bounds[1])
        self.toolbox.register("attr_epochs", random.randint,
            self.bounds.epochs_bounds[0], self.bounds.epochs_bounds[1])
        self.toolbox.register("attr_dropout", random.uniform,
            self.bounds.dropout_bounds[0], self.bounds.dropout_bounds[1])
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
            (self.toolbox.attr_lr, self.toolbox.attr_neurons1,
             self.toolbox.attr_neurons2, self.toolbox.attr_epochs,
             self.toolbox.attr_dropout), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
    
    def _decode_individual(self, individual: List) -> Dict[str, Any]:
        """Decode chromosome to hyperparameters."""
        return {
            'learning_rate': 10 ** individual[0],
            'neurons_layer1': int(round(individual[1])),
            'neurons_layer2': int(round(individual[2])),
            'epochs': int(round(individual[3])),
            'dropout_rate': individual[4]
        }
    
    def _evaluate_individual(self, individual: List) -> Tuple[float]:
        """Evaluate fitness (validation RMSE)."""
        params = self._decode_individual(individual)
        
        try:
            model = Sequential([
                Input(shape=(self.n_features,)),
                Dense(params['neurons_layer1'], activation='relu'),
                BatchNormalization(),
                Dropout(params['dropout_rate']),
                Dense(params['neurons_layer2'], activation='relu'),
                BatchNormalization(),
                Dropout(params['dropout_rate']),
                Dense(1, activation='linear')
            ])
            
            model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
            
            model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=params['epochs'],
                batch_size=32,
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            val_predictions = model.predict(self.X_val, verbose=0).flatten()
            val_rmse = np.sqrt(np.mean((self.y_val - val_predictions) ** 2))
            
            del model
            tf.keras.backend.clear_session()
            
            return (val_rmse,)
            
        except Exception as e:
            print(f"[WARNING] Evaluation failed: {e}")
            return (float('inf'),)
    
    def _mutate_individual(self, individual: List) -> Tuple[List]:
        """Mutate individual's genes."""
        # Learning rate
        if random.random() < 0.3:
            individual[0] += random.gauss(0, 0.3)
            individual[0] = np.clip(individual[0],
                np.log10(self.bounds.learning_rate_bounds[0]),
                np.log10(self.bounds.learning_rate_bounds[1]))
        
        # Neurons layer 1
        if random.random() < 0.3:
            individual[1] += random.randint(-16, 16)
            individual[1] = np.clip(individual[1],
                self.bounds.neurons_layer1_bounds[0], self.bounds.neurons_layer1_bounds[1])
        
        # Neurons layer 2
        if random.random() < 0.3:
            individual[2] += random.randint(-8, 8)
            individual[2] = np.clip(individual[2],
                self.bounds.neurons_layer2_bounds[0], self.bounds.neurons_layer2_bounds[1])
        
        # Epochs
        if random.random() < 0.3:
            individual[3] += random.randint(-20, 20)
            individual[3] = np.clip(individual[3],
                self.bounds.epochs_bounds[0], self.bounds.epochs_bounds[1])
        
        # Dropout
        if random.random() < 0.3:
            individual[4] += random.gauss(0, 0.05)
            individual[4] = np.clip(individual[4],
                self.bounds.dropout_bounds[0], self.bounds.dropout_bounds[1])
        
        return (individual,)
    
    def optimize(self, verbose: bool = True) -> Tuple[Dict[str, Any], List[Dict]]:
        """Run GA optimization."""
        print("=" * 60)
        print("GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
        print(f"Population: {self.population_size}, Generations: {self.n_generations}")
        print("=" * 60)
        
        start_time = time.time()
        population = self.toolbox.population(n=self.population_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        
        hof = tools.HallOfFame(5)
        
        if verbose:
            print("\n[INFO] Starting evolution...")
        
        for gen in range(self.n_generations):
            gen_start = time.time()
            
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = offspring
            hof.update(population)
            
            record = stats.compile(population)
            gen_time = time.time() - gen_start
            
            self.optimization_history.append({
                'generation': gen + 1,
                'min_fitness': record['min'],
                'avg_fitness': record['avg'],
                'std_fitness': record['std'],
                'best_params': self._decode_individual(hof[0]),
                'generation_time': gen_time
            })
            
            if verbose:
                print(f"Gen {gen+1:3d} | Min RMSE: {record['min']:.6f} | Avg: {record['avg']:.6f} | Time: {gen_time:.1f}s")
        
        self.total_time = time.time() - start_time
        self.best_individual = hof[0]
        self.best_fitness = hof[0].fitness.values[0]
        best_params = self._decode_individual(hof[0])
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total time: {self.total_time:.2f}s")
        print(f"Best RMSE: {self.best_fitness:.6f}")
        print(f"Best params: {best_params}")
        print("=" * 60)
        
        return best_params, self.optimization_history
    
    def train_best_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train final model with best hyperparameters."""
        if self.best_individual is None:
            raise ValueError("No optimization performed. Call optimize() first.")
        
        best_params = self._decode_individual(self.best_individual)
        print("\n[INFO] Training final model with best hyperparameters...")
        
        model = Sequential([
            Input(shape=(self.n_features,)),
            Dense(best_params['neurons_layer1'], activation='relu'),
            BatchNormalization(),
            Dropout(best_params['dropout_rate']),
            Dense(best_params['neurons_layer2'], activation='relu'),
            BatchNormalization(),
            Dropout(best_params['dropout_rate']),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse', metrics=['mae'])
        
        start_time = time.time()
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=best_params['epochs'],
            batch_size=32,
            callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
            verbose=1
        )
        training_time = time.time() - start_time
        
        test_results = model.evaluate(X_test, y_test, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0).flatten()
        test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        
        results = {
            'model_type': 'GA-ANN',
            'best_params': best_params,
            'test_loss': test_results[0],
            'test_mae': test_results[1],
            'test_rmse': test_rmse,
            'training_time': training_time,
            'optimization_time': self.total_time,
            'total_time': training_time + self.total_time,
            'history': history.history,
            'predictions': {'test': y_pred_test}
        }
        
        print("\n" + "=" * 60)
        print("GA-ANN RESULTS")
        print("=" * 60)
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Test MAE: {test_results[1]:.6f}")
        print(f"Training Time: {training_time:.2f}s")
        print("=" * 60)
        
        return model, results


def optimize_ann_with_ga(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    population_size: int = 20,
    n_generations: int = 15,
    random_seed: int = 42
) -> Tuple[Any, Dict[str, Any], List[Dict]]:
    """Complete GA optimization pipeline for ANN."""
    optimizer = GeneticANNOptimizer(
        X_train, y_train, X_val, y_val,
        population_size=population_size,
        n_generations=n_generations,
        random_seed=random_seed
    )
    
    best_params, history = optimizer.optimize()
    model, results = optimizer.train_best_model(X_test, y_test)
    results['optimization_history'] = history
    
    return model, results, history


if __name__ == "__main__":
    print("Testing GA optimizer...")
    
    np.random.seed(42)
    n_samples, n_features = 500, 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1
    
    X_train, X_val, X_test = X[:350], X[350:425], X[425:]
    y_train, y_val, y_test = y[:350], y[350:425], y[425:]
    
    optimizer = GeneticANNOptimizer(X_train, y_train, X_val, y_val,
        population_size=10, n_generations=5)
    
    best_params, history = optimizer.optimize()
    model, results = optimizer.train_best_model(X_test, y_test)
    
    print("\nGA optimizer test completed!")
