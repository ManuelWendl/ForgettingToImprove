import yaml
from typing import Dict, Any, List, Union
from ..objectives import sin_symmetric_lengthscale_increase, sin_asymmetric_lengthscale_increase, sin_periodic_lengthscale_increase

class ConfigLoader:
    """Load and parse configuration files for GP experiments."""
    
    def __init__(self):
        self.objective_functions = {
            'sin_symmetric_lengthscale_increase': sin_symmetric_lengthscale_increase,
            'sin_asymmetric_lengthscale_increase': sin_asymmetric_lengthscale_increase,
            'sin_periodic_lengthscale_increase': sin_periodic_lengthscale_increase
        }
        
        # Available datasets
        self.datasets = {
            'boston': 'boston',
            'twitter': 'twitter',
            'casp': 'casp'
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return self._validate_and_parse_config(config)
    
    def _parse_limits(self, limits_value):
        """Parse limits that can be 1D tuple or multidimensional list of tuples."""
        if isinstance(limits_value, list):
            if len(limits_value) == 2 and not isinstance(limits_value[0], (list, tuple)):
                # 1D case: [min, max]
                return tuple(limits_value)
            else:
                # Multidimensional case: [[min1, max1], [min2, max2], ...]
                return [tuple(lim) for lim in limits_value]
        else:
            raise ValueError(f"Invalid limits format: {limits_value}")
    
    def _validate_and_parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and parse the loaded configuration."""
        experiment = config.get('experiment', {})
        
        # Parse experiment type
        exp_type = experiment.get('type', 'single')
        if exp_type not in ['single', 'statistic']:
            raise ValueError(f"Invalid experiment type: {exp_type}. Must be 'single' or 'statistic'")
        
        # Parse methods
        methods = self._parse_list_or_single(experiment.get('method', 'none'))
        valid_methods = ['none', 'sequential', 'batch', 'targetSampling', 'sequentialPerKernel', 'batchPerKernel', 'relevance_pursuit', 'warped_gp', 'heteroscedastic_gp']
        for method in methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")
        
        # Parse objectives (can be functions or datasets)
        objectives = self._parse_list_or_single(experiment.get('objective', 'sin_symmetric_lengthscale_increase'))
        objective_funcs = []
        objective_types = []
        
        for obj_name in objectives:
            if obj_name in self.objective_functions:
                objective_funcs.append(self.objective_functions[obj_name])
                objective_types.append('function')
            elif obj_name in self.datasets:
                objective_funcs.append(obj_name)
                objective_types.append('dataset')
            else:
                raise ValueError(f"Unknown objective function or dataset: {obj_name}")
        
        # Parse noise types
        noises = self._parse_list_or_single(experiment.get('noise', 'joint'))
        valid_noises = ['joint', 'epistemic', 'aleatoric']
        for noise in noises:
            if noise not in valid_noises:
                raise ValueError(f"Invalid noise type: {noise}. Must be one of {valid_noises}")
        
        # Parse seed
        seed = experiment.get('seed', 0)
        if exp_type == 'statistic':
            seeds = list(range(0, seed)) if seed > 0 else [0]
        else:
            seeds = [seed]
        
        # Parse limits
        A_limits = self._parse_limits(experiment.get('A_limits', [-3, 3]))
        X_limits = self._parse_limits(experiment.get('X_limits', [-10, 10]))
        
        # Parse kernel configuration
        kernel_config = experiment.get('kernel', None)
        
        # Parse other parameters with defaults
        parsed_config = {
            'type': exp_type,
            'methods': methods,
            'objectives': objectives,
            'objective_funcs': objective_funcs,
            'objective_types': objective_types,
            'noises': noises,
            'seeds': seeds,
            'A_limits': A_limits,
            'X_limits': X_limits,
            'num_samples': experiment.get('num_samples', 1000),
            'num_train_samples': experiment.get('num_train_samples', 20),
            'num_A_samples': experiment.get('num_A_samples', 50),
            'fixed_kernel': experiment.get('fixed_kernel', True),
            'show_plots': experiment.get('show_plots', False),
            'show_hessian': experiment.get('show_hessian', False),
            'calculate_convexity': experiment.get('calculate_convexity', False),
            'feature_subset': experiment.get('feature_subset', None),
            'kernel_config': kernel_config,
            'gp_alpha': experiment.get('gp_alpha', 0.01),
            'per_kernel_removal_strategy': experiment.get('per_kernel_removal_strategy', 'one_per_kernel')
        }
        
        return parsed_config
    
    def _parse_list_or_single(self, value: Union[str, List[str]]) -> List[str]:
        """Parse a value that can be either a single string or list of strings."""
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(f"Expected string or list, got {type(value)}")
