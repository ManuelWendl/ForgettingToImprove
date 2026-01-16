import os
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

class ResultsWriter:
    """Write experiment results to text files."""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Default to results folder inside forgetting_to_improve directory
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def write_single_experiment_results(self, config: Dict[str, Any], results: Dict[str, Any], 
                                      method: str, objective: str, noise: str, seed: int):
        """Write results for a single experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"single_experiment_{method}_{objective}_{noise}_seed{seed}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=== Single Experiment Results ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Objective: {objective}\n")
            f.write(f"Noise: {noise}\n")
            f.write(f"Seed: {seed}\n")
            f.write("\n=== Configuration ===\n")
            f.write(f"A_limits: {config['A_limits']}\n")
            f.write(f"X_limits: {config['X_limits']}\n")
            f.write(f"num_samples: {config['num_samples']}\n")
            f.write(f"num_train_samples: {config['num_train_samples']}\n")
            f.write(f"num_A_samples: {config['num_A_samples']}\n")
            f.write(f"fixed_kernel: {config['fixed_kernel']}\n")
            f.write("\n=== Error Metrics ===\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value:.6f}\n")
        
        print(f"Results written to: {filepath}")
        return filepath
    
    def write_comparison_results(self, config: Dict[str, Any], all_results: Dict[str, Dict[str, Any]],
                               comparison_type: str = "method"):
        """Write comparison results for multiple methods or noise types."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{comparison_type}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"=== {comparison_type.title()} Comparison Results ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n=== Configuration ===\n")
            f.write(f"A_limits: {config['A_limits']}\n")
            f.write(f"X_limits: {config['X_limits']}\n")
            f.write(f"num_samples: {config['num_samples']}\n")
            f.write(f"num_train_samples: {config['num_train_samples']}\n")
            f.write(f"num_A_samples: {config['num_A_samples']}\n")
            f.write(f"fixed_kernel: {config['fixed_kernel']}\n")
            
            if config['type'] == 'statistic':
                f.write(f"Seeds: {len(config['seeds'])} seeds ({min(config['seeds'])}-{max(config['seeds'])})\n")
            else:
                f.write(f"Seed: {config['seeds'][0]}\n")
            
            f.write("\n=== Results ===\n")
            
            # Get all unique metrics
            all_metrics = set()
            for results_dict in all_results.values():
                if isinstance(results_dict, dict):
                    for result in results_dict.values():
                        if isinstance(result, dict):
                            all_metrics.update(result.keys())
                else:
                    all_metrics.update(results_dict.keys())
            
            all_metrics = sorted(all_metrics)
            
            # Write header
            f.write(f"{'Variant':<20}")
            for metric in all_metrics:
                f.write(f"{metric:>15}")
            f.write("\n")
            f.write("-" * (20 + 15 * len(all_metrics)) + "\n")
            
            # Write results for each variant
            for variant, results in all_results.items():
                f.write(f"{variant:<20}")
                if isinstance(results, dict) and 'mean' in results:
                    # Statistical results with mean ± std
                    for metric in all_metrics:
                        if metric in results['mean']:
                            mean_val = results['mean'][metric]
                            std_val = results.get('std', {}).get(metric, 0)
                            f.write(f"{mean_val:>7.3f} ± {std_val:<6.3f}")
                        else:
                            f.write(f"{'N/A':>15}")
                else:
                    # Single experiment results
                    for metric in all_metrics:
                        if metric in results:
                            f.write(f"{results[metric]:>15.6f}")
                        else:
                            f.write(f"{'N/A':>15}")
                f.write("\n")
            
            # If statistical experiment, write min values as well
            if config['type'] == 'statistic':
                f.write("\n=== Minimum Values Across Seeds ===\n")
                f.write(f"{'Variant':<20}")
                for metric in all_metrics:
                    f.write(f"{metric:>15}")
                f.write("\n")
                f.write("-" * (20 + 15 * len(all_metrics)) + "\n")
                
                for variant, results in all_results.items():
                    f.write(f"{variant:<20}")
                    if isinstance(results, dict) and 'min' in results:
                        for metric in all_metrics:
                            if metric in results['min']:
                                f.write(f"{results['min'][metric]:>15.6f}")
                            else:
                                f.write(f"{'N/A':>15}")
                    else:
                        for metric in all_metrics:
                            f.write(f"{'N/A':>15}")
                    f.write("\n")
                
                # Add runtime summary if available
                if any('Runtime' in results.get('mean', {}) for results in all_results.values()):
                    f.write("\n=== Runtime Summary (seconds) ===\n")
                    f.write(f"{'Variant':<20}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}\n")
                    f.write("-" * 68 + "\n")
                    for variant, results in all_results.items():
                        if isinstance(results, dict) and 'Runtime' in results.get('mean', {}):
                            mean_runtime = results['mean']['Runtime']
                            std_runtime = results.get('std', {}).get('Runtime', 0)
                            min_runtime = results.get('min', {}).get('Runtime', 0)
                            max_runtime = results.get('max', {}).get('Runtime', 0)
                            f.write(f"{variant:<20}{mean_runtime:>12.4f}{std_runtime:>12.4f}{min_runtime:>12.4f}{max_runtime:>12.4f}\n")
        
        print(f"Comparison results written to: {filepath}")
        return filepath
    
    def aggregate_statistical_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate results from multiple seeds into mean, std, min, and max statistics."""
        if not results_list:
            return {}
        
        # Get all metrics
        all_metrics = set()
        for result in results_list:
            all_metrics.update(result.keys())
        
        aggregated = {'mean': {}, 'std': {}, 'min': {}, 'max': {}}
        
        for metric in all_metrics:
            values = [result.get(metric, 0) for result in results_list if metric in result]
            if values:
                aggregated['mean'][metric] = np.mean(values)
                aggregated['std'][metric] = np.std(values)
                aggregated['min'][metric] = min(values)
                aggregated['max'][metric] = max(values)
        
        return aggregated
