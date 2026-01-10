"""
ga_optimizer.py - Genetic Algorithm Optimizer for ANN Hyperparameters

This module implements a Genetic Algorithm (GA) to optimize the hyperparameters
of the feedforward ANN model, creating a HYBRID CI approach.

Hybrid CI Justification:
========================
This implementation combines TWO distinct CI paradigms:
1. Evolutionary Computation (Genetic Algorithm) - for global search
2. Neural Networks (ANN) - for function approximation

The GA handles the discrete/continuous optimization of hyperparameters,
while the ANN handles the actual prediction task. This is more effective than:
- Pure grid search (exhaustive, computationally expensive)
- Random search (no guided exploration)
- Pure gradient methods (can get stuck in local optima)

Genetic Algorithm Components:
=============================
1. Chromosome: Encoded hyperparameters (learning_rate, neurons, epochs)
2. Population: Collection of candidate solutions
3. Fitness Function: Validation RMSE (lower is better)
4. Selection: Tournament selection
5. Crossover: Blend crossover for continuous, uniform for discrete
6. Mutation: Gaussian mutation for continuous, random reset for discrete

Author: TCI6313 Student
Date: 2026
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass
import warnings

# DEAP - Distributed Evolutionary Algorithms in Python
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    raise ImportError("DEAP is required. Install with: pip install deap")

# TensorFlow for ANN training
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


@dataclass
class HyperparameterBounds:
    """
    Define bounds for hyperparameters to optimize.
    
    This dataclass specifies the search space for the GA.
    Each hyperparameter has a minimum and maximum value.
    
    Attributes:
    -----------
    learning_rate_bounds : Tuple[float, float]
        Min and max learning rate (log scale recommended)
    neurons_layer1_bounds : Tuple[int, int]
        Min and max neurons in first hidden layer
    neurons_layer2_bounds : Tuple[int, int]
        Min and max neurons in second hidden layer
    epochs_bounds : Tuple[int, int]
        Min and max training epochs
    dropout_bounds : Tuple[float, float]
        Min and max dropout rate
    """
    learning_rate_bounds: Tuple[float, float] = (0.0001, 0.01)
    neurons_layer1_bounds: Tuple[int, int] = (32, 128)
    neurons_layer2_bounds: Tuple[int, int] = (16, 64)
    epochs_bounds: Tuple[int, int] = (50, 150)
    dropout_bounds: Tuple[float, float] = (0.1, 0.4)


class GeneticANNOptimizer:
    """
    Genetic Algorithm Optimizer for ANN Hyperparameters.
    
    This class implements a GA to find optimal hyperparameters for the ANN model.
    It's a HYBRID CI approach combining evolutionary computation with neural networks.
    
    The GA evolves a population of hyperparameter configurations, using validation
    RMSE as the fitness function. This global search method can find better
    configurations than manual tuning or grid search.
    
    Attributes:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    n_features : Number of input features
    bounds : HyperparameterBounds
    population_size : Number of individuals in population
    n_generations : Number of generations to evolve
    crossover_prob : Probability of crossover
    mutation_prob : Probability of mutation
    tournament_size : Size of tournament for selection
    
    Example:
    --------
    >>> optimizer = GeneticANNOptimizer(X_train, y_train, X_val, y_val)
    >>> best_params, history = optimizer.optimize()
    >>> print(f"Best learning rate: {best_params['learning_rate']}")
    """
    
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
        """
        Initialize the GA optimizer.
        
        Parameters:
        -----------
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data for fitness evaluation
        bounds : HyperparameterBounds
            Search space bounds
        population_size : int
            Number of individuals in each generation
        n_generations : int
            Number of generations to evolve
        crossover_prob : float
            Probability of crossover between parents
        mutation_prob : float
            Probability of mutation
        tournament_size : int
            Number of individuals in tournament selection
        random_seed : int
            Random seed for reproducibility
        """
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
        
        # Tracking
        self.best_individual = None
        self.best_fitness = float('inf')
        self.optimization_history = []
        self.total_time = None
        
        # Set seeds
        np.random.seed(random_seed)
        random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Setup DEAP
        self._setup_deap()
    
    def _setup_deap(self) -> None:
        """
        Setup DEAP framework for genetic algorithm.
        
        This configures:
        - Fitness class (minimizing RMSE)
        - Individual class (list of hyperparameters)
        - Population initialization
        - Genetic operators
        """
        # Create fitness class (minimize RMSE, so weights=-1.0)
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual
            
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generators for each hyperparameter
        # Chromosome: [learning_rate, neurons_layer1, neurons_layer2, epochs, dropout]
        
        # Learning rate (continuous, log scale)
        self.toolbox.register(
            "attr_lr",
            random.uniform,
            np.log10(self.bounds.learning_rate_bounds[0]),
            np.log10(self.bounds.learning_rate_bounds[1])
        )
        
        # Neurons layer 1 (discrete)
        self.toolbox.register(
            "attr_neurons1",
            random.randint,
            self.bounds.neurons_layer1_bounds[0],
            self.bounds.neurons_layer1_bounds[1]
        )
        
        # Neurons layer 2 (discrete)
        self.toolbox.register(
            "attr_neurons2",
            random.randint,
            self.bounds.neurons_layer2_bounds[0],
            self.bounds.neurons_layer2_bounds[1]
        )
        
        # Epochs (discrete)
        self.toolbox.register(
            "attr_epochs",
            random.randint,
            self.bounds.epochs_bounds[0],
            self.bounds.epochs_bounds[1]
        )
        
        # Dropout rate (continuous)
        self.toolbox.register(
            "attr_dropout",
            random.uniform,
            self.bounds.dropout_bounds[0],
            self.bounds.dropout_bounds[1]
        )
        
        # Individual and population
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (self.toolbox.attr_lr, self.toolbox.attr_neurons1,
             self.toolbox.attr_neurons2, self.toolbox.attr_epochs,
             self.toolbox.attr_dropout),
            n=1
        )
        
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.tournament_size
        )
    
    def _decode_individual(self, individual: List) -> Dict[str, Any]:
        """
        Decode chromosome into hyperparameters.
        
        Parameters:
        -----------
        individual : List
            Encoded chromosome [lr_log, neurons1, neurons2, epochs, dropout]
            
        Returns:
        --------
        Dict[str, Any]
            Decoded hyperparameters
        """
        return {
            'learning_rate': 10 ** individual[0],  # Convert from log scale
            'neurons_layer1': int(round(individual[1])),
            'neurons_layer2': int(round(individual[2])),
            'epochs': int(round(individual[3])),
            'dropout_rate': individual[4]
        }
    
    def _evaluate_individual(self, individual: List) -> Tuple[float]:
        """
        Evaluate fitness of an individual (train ANN and compute validation RMSE).
        
        This is the FITNESS FUNCTION of the GA.
        
        Parameters:
        -----------
        individual : List
            Encoded chromosome
            
        Returns:
        --------
        Tuple[float]
            Validation RMSE (lower is better)
            
        Notes:
        ------
        Fitness = Validation RMSE
        We use validation set (not test set) to prevent overfitting to test data.
        This is critical for proper model evaluation.
        """
        params = self._decode_individual(individual)
        
        try:
            # Build ANN with these hyperparameters
            model = Sequential([
                Input(shape=(self.n_features,)),
                Dense(
                    params['neurons_layer1'],
                    activation='relu'
                ),
                BatchNormalization(),
                Dropout(params['dropout_rate']),
                Dense(
                    params['neurons_layer2'],
                    activation='relu'
                ),
                BatchNormalization(),
                Dropout(params['dropout_rate']),
                Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='mse'
            )
            
            # Train with early stopping (reduced epochs for faster evaluation)
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=params['epochs'],
                batch_size=32,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            # Compute validation RMSE (fitness)
            val_predictions = model.predict(self.X_val, verbose=0).flatten()
            val_rmse = np.sqrt(np.mean((self.y_val - val_predictions) ** 2))
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return (val_rmse,)
            
        except Exception as e:
            print(f"[WARNING] Evaluation failed: {e}")
            return (float('inf'),)  # Return worst fitness on failure
    
    def _mutate_individual(self, individual: List) -> Tuple[List]:
        """
        Mutate an individual's genes.
        
        Mutation introduces genetic diversity and helps explore the search space.
        
        Parameters:
        -----------
        individual : List
            Chromosome to mutate
            
        Returns:
        --------
        Tuple[List]
            Mutated individual
        """
        # Learning rate (log scale) - Gaussian mutation
        if random.random() < 0.3:
            individual[0] += random.gauss(0, 0.3)
            individual[0] = np.clip(
                individual[0],
                np.log10(self.bounds.learning_rate_bounds[0]),
                np.log10(self.bounds.learning_rate_bounds[1])
            )
        
        # Neurons layer 1 - Random adjustment
        if random.random() < 0.3:
            individual[1] += random.randint(-16, 16)
            individual[1] = np.clip(
                individual[1],
                self.bounds.neurons_layer1_bounds[0],
                self.bounds.neurons_layer1_bounds[1]
            )
        
        # Neurons layer 2 - Random adjustment
        if random.random() < 0.3:
            individual[2] += random.randint(-8, 8)
            individual[2] = np.clip(
                individual[2],
                self.bounds.neurons_layer2_bounds[0],
                self.bounds.neurons_layer2_bounds[1]
            )
        
        # Epochs - Random adjustment
        if random.random() < 0.3:
            individual[3] += random.randint(-20, 20)
            individual[3] = np.clip(
                individual[3],
                self.bounds.epochs_bounds[0],
                self.bounds.epochs_bounds[1]
            )
        
        # Dropout rate - Gaussian mutation
        if random.random() < 0.3:
            individual[4] += random.gauss(0, 0.05)
            individual[4] = np.clip(
                individual[4],
                self.bounds.dropout_bounds[0],
                self.bounds.dropout_bounds[1]
            )
        
        return (individual,)
    
    def optimize(self, verbose: bool = True) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Run the genetic algorithm optimization.
        
        This is the main optimization loop that evolves the population
        over multiple generations to find optimal hyperparameters.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        Tuple[Dict[str, Any], List[Dict]]
            Best hyperparameters and optimization history
            
        Notes:
        ------
        GA Process:
        1. Initialize random population
        2. Evaluate fitness of all individuals
        3. Select parents via tournament selection
        4. Apply crossover to create offspring
        5. Apply mutation for diversity
        6. Replace population
        7. Repeat for n_generations
        """
        print("=" * 60)
        print("GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Crossover probability: {self.crossover_prob}")
        print(f"Mutation probability: {self.mutation_prob}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Initialize population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        
        # Hall of fame - keeps best individuals
        hof = tools.HallOfFame(5)
        
        # Run GA with elitism
        if verbose:
            print("\n[INFO] Starting evolution...")
        
        for gen in range(self.n_generations):
            gen_start = time.time()
            
            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Update hall of fame
            hof.update(population)
            
            # Record statistics
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
                print(f"Gen {gen+1:3d} | "
                      f"Min RMSE: {record['min']:.6f} | "
                      f"Avg RMSE: {record['avg']:.6f} | "
                      f"Time: {gen_time:.1f}s")
        
        self.total_time = time.time() - start_time
        
        # Get best individual
        self.best_individual = hof[0]
        self.best_fitness = hof[0].fitness.values[0]
        best_params = self._decode_individual(hof[0])
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total time: {self.total_time:.2f} seconds")
        print(f"Best Validation RMSE: {self.best_fitness:.6f}")
        print(f"Best Hyperparameters:")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
        print("=" * 60)
        
        return best_params, self.optimization_history
    
    def train_best_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a final model using the best hyperparameters found.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
            
        Returns:
        --------
        Tuple[Model, Dict]
            Trained model and evaluation results
        """
        if self.best_individual is None:
            raise ValueError("No optimization performed. Call optimize() first.")
        
        best_params = self._decode_individual(self.best_individual)
        
        print("\n[INFO] Training final model with best hyperparameters...")
        
        # Build final model
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
        
        model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with all validation techniques
        start_time = time.time()
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=best_params['epochs'],
            batch_size=32,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
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
            'predictions': {
                'test': y_pred_test
            }
        }
        
        print("\n" + "=" * 60)
        print("GA-OPTIMIZED ANN RESULTS")
        print("=" * 60)
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Test MAE: {test_results[1]:.6f}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Optimization Time: {self.total_time:.2f} seconds")
        print("=" * 60)
        
        return model, results


def optimize_ann_with_ga(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    population_size: int = 20,
    n_generations: int = 15,
    random_seed: int = 42
) -> Tuple[Any, Dict[str, Any], List[Dict]]:
    """
    Complete GA optimization pipeline for ANN.
    
    This is the main entry point for GA-based ANN optimization.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    X_test, y_test : Test data
    population_size : GA population size
    n_generations : Number of GA generations
    random_seed : Random seed
    
    Returns:
    --------
    Tuple[Model, Dict, List]
        Trained model, results, and optimization history
    """
    # Initialize optimizer
    optimizer = GeneticANNOptimizer(
        X_train, y_train,
        X_val, y_val,
        population_size=population_size,
        n_generations=n_generations,
        random_seed=random_seed
    )
    
    # Run optimization
    best_params, history = optimizer.optimize()
    
    # Train final model
    model, results = optimizer.train_best_model(X_test, y_test)
    
    results['optimization_history'] = history
    
    return model, results, history


if __name__ == "__main__":
    # Example usage and testing
    print("Testing GA optimizer module...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1
    
    # Split data
    X_train, X_val, X_test = X[:350], X[350:425], X[425:]
    y_train, y_val, y_test = y[:350], y[350:425], y[425:]
    
    # Run optimization (reduced generations for testing)
    optimizer = GeneticANNOptimizer(
        X_train, y_train,
        X_val, y_val,
        population_size=10,
        n_generations=5
    )
    
    best_params, history = optimizer.optimize()
    model, results = optimizer.train_best_model(X_test, y_test)
    
    print("\nGA optimizer module test completed successfully!")
