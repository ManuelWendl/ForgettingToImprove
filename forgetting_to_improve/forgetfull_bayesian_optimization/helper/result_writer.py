from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path


def save_results(results: Dict[str, Any], config: Dict[str, Any], results_path: str) -> None:
    """
    Save experiment results to a text file.
    
    Args:
        results: Results dictionary with method names as keys and statistics as values
        config: Configuration dictionary
        results_path: Path to save results
    """
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Bayesian Optimization Experiment Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Objective: {config['objective']}\n")
        f.write(f"Acquisition: {config['acquisition']}\n")
        f.write(f"Number of iterations: {config['n_iter']}\n")
        f.write(f"Number of seeds: {config['n_seeds']}\n")
        f.write(f"Initial points: {config.get('init_points', 0)}\n")
        f.write(f"Methods compared: {', '.join(results.keys())}\n\n")
        
        # Write results for each method
        for method_name, stats in results.items():
            f.write("=" * 80 + "\n")
            f.write(f"Results for Method: {method_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Final Simple Regret Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {stats['simple_regret']['mean'][-1]:.6f}\n")
            f.write(f"Std: {stats['simple_regret']['std'][-1]:.6f}\n")
            f.write(f"Median: {stats['simple_regret']['median'][-1]:.6f}\n")
            f.write(f"Min: {stats['simple_regret']['min'][-1]:.6f}\n")
            f.write(f"Max: {stats['simple_regret']['max'][-1]:.6f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Final Cumulative Regret Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {stats['cumulative_regret']['mean'][-1]:.6f}\n")
            f.write(f"Std: {stats['cumulative_regret']['std'][-1]:.6f}\n")
            f.write(f"Median: {stats['cumulative_regret']['median'][-1]:.6f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Final Best Value Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {stats['best_values']['mean'][-1]:.6f}\n")
            f.write(f"Std: {stats['best_values']['std'][-1]:.6f}\n")
            f.write(f"Median: {stats['best_values']['median'][-1]:.6f}\n")
            f.write(f"Min: {stats['best_values']['min'][-1]:.6f}\n")
            f.write(f"Max: {stats['best_values']['max'][-1]:.6f}\n\n")
        
        # Add comparison section if multiple methods
        if len(results) > 1:
            f.write("=" * 80 + "\n")
            f.write("Method Comparison (Final Simple Regret)\n")
            f.write("=" * 80 + "\n")
            for method_name, stats in results.items():
                f.write(f"{method_name.upper()}: {stats['simple_regret']['mean'][-1]:.6f} ± {stats['simple_regret']['std'][-1]:.6f}\n")
            
            # Find best method by mean simple regret
            best_method = min(results.items(), key=lambda x: x[1]['simple_regret']['mean'][-1])
            f.write(f"\nBest method: {best_method[0].upper()}\n")


def compute_statistics(all_results: List[List[float]], optimal_value: float) -> Dict[str, Any]:
    """
    Compute statistics across multiple random seeds.
    
    Args:
        all_results: List of best observed values for each seed
        optimal_value: Optimal value of the objective
        
    Returns:
        Dictionary containing statistics
    """
    all_simple_regrets = []
    all_cumulative_regrets = []
    
    for best_observed in all_results:
        simple_regret, cumulative_regret = compute_regret(best_observed, optimal_value)
        all_simple_regrets.append(simple_regret)
        all_cumulative_regrets.append(cumulative_regret)
    
    # Convert to arrays for easier manipulation
    all_simple_regrets = np.array(all_simple_regrets)
    all_cumulative_regrets = np.array(all_cumulative_regrets)
    all_best_values = np.array(all_results)
    
    stats = {
        'simple_regret': {
            'mean': np.mean(all_simple_regrets, axis=0),
            'std': np.std(all_simple_regrets, axis=0),
            'median': np.median(all_simple_regrets, axis=0),
            'min': np.min(all_simple_regrets, axis=0),
            'max': np.max(all_simple_regrets, axis=0),
        },
        'cumulative_regret': {
            'mean': np.mean(all_cumulative_regrets, axis=0),
            'std': np.std(all_cumulative_regrets, axis=0),
            'median': np.median(all_cumulative_regrets, axis=0),
            'min': np.min(all_cumulative_regrets, axis=0),
            'max': np.max(all_cumulative_regrets, axis=0),
        },
        'best_values': {
            'mean': np.mean(all_best_values, axis=0),
            'std': np.std(all_best_values, axis=0),
            'median': np.median(all_best_values, axis=0),
            'min': np.min(all_best_values, axis=0),
            'max': np.max(all_best_values, axis=0),
        }
    }
    
    return stats


def compute_regret(best_observed: List[float], optimal_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute simple and cumulative regret.
    
    Args:
        best_observed: List of best observed values at each iteration
        optimal_value: Optimal value of the objective
        
    Returns:
        Tuple of (simple_regret, cumulative_regret)
    """
    best_observed_array = np.array(best_observed)
    simple_regret = optimal_value - best_observed_array
    cumulative_regret = np.cumsum(simple_regret)
    
    return simple_regret, cumulative_regret