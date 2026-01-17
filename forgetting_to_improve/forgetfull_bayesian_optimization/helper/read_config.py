import yaml
from typing import Dict, Any, List


def read_config(config_path: str) -> Dict[str, Any]:
    """
    Read and parse the YAML configuration file for Bayesian optimization experiments.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing parsed configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config['bayes_opt']


def get_kernel_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract kernel configuration from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of kernel configuration dictionaries
    """
    return config.get('kernel', [])


def get_acquisition_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract acquisition function configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with acquisition function name and parameters
    """
    acquisition = config.get('acquisition', 'qUpperConfidenceBound')
    acquisition_params = config.get('acquisition_params', {})
    
    return {
        'name': acquisition,
        'params': acquisition_params
    }


def get_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract sample optimization configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with optimization parameters
    """
    return config.get('sample_optimization', {})


def get_perturbation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract perturbation configuration for noise corruption.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with perturbation parameters (type, probability, value)
    """
    return config.get('perturbation', {})


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration parameters are present.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required parameters are missing
    """
    required_fields = ['objective', 'n_seeds', 'n_iter']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required configuration field '{field}' is missing")
    
    # Validate type
    if 'type' in config and config['type'] not in ['comparative', 'single']:
        raise ValueError(f"Invalid type: {config['type']}. Must be 'comparative' or 'single'")
