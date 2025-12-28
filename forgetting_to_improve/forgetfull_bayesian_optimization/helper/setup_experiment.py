import torch
from typing import Dict, Any, Callable, Tuple, List
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley, Branin, Hartmann, Rosenbrock, Levy
from botorch.acquisition import (
    qUpperConfidenceBound, 
    qLogExpectedImprovement, 
    qLogNoisyExpectedImprovement
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.kernels.kernel import AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval

from .read_config import get_kernel_config


def get_objective_function(objective_name: str) -> Tuple[Callable, torch.Tensor]:
    """
    Get the objective function and bounds based on the name.
    
    Args:
        objective_name: Name of the objective function
        
    Returns:
        Tuple of (objective_function, bounds)
    """
    objective_map = {
        'botorch_ackley_2d': (Ackley(dim=2, negate=True), torch.tensor([[-10.0] * 2, [30.0] * 2])),
        'botorch_ackley_5d': (Ackley(dim=5, negate=True), torch.tensor([[0.0] * 5, [1.0] * 5])),
        'botorch_branin': (Branin(negate=True), torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
        'botorch_hartmann_6d': (Hartmann(dim=6, negate=True), torch.tensor([[0.0] * 6, [1.0] * 6])),
        'botorch_rosenbrock_2d': (Rosenbrock(dim=2, negate=True), torch.tensor([[0.0] * 2, [1.0] * 2])),
        'botorch_levy_4d': (Levy(dim=4, negate=True), torch.tensor([[0.0] * 4, [1.0] * 4])),
    }
    
    if objective_name not in objective_map:
        raise ValueError(f"Unknown objective function: {objective_name}")
    
    return objective_map[objective_name]


def create_kernel_from_config(kernel_configs: List[Dict[str, Any]], input_dim: int) -> Any:
    """
    Create a GPyTorch kernel from configuration.
    
    Args:
        kernel_configs: List of kernel configuration dictionaries
        input_dim: Input dimensionality
        
    Returns:
        GPyTorch kernel module
    """
    if not kernel_configs:
        # Default: Matern 2.5 kernel
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=input_dim))
    
    kernels = []
    for kernel_config in kernel_configs:
        kernel_type = kernel_config.get('type', 'matern')
        
        if kernel_type == 'matern':
            nu = kernel_config.get('nu', 2.5)
            length_scale = kernel_config.get('length_scale', 1.0)
            kernel = MaternKernel(nu=nu, ard_num_dims=input_dim)
            if length_scale != 1.0:
                kernel.lengthscale = length_scale
            kernels.append(ScaleKernel(kernel))
            
        elif kernel_type == 'rbf':
            length_scale = kernel_config.get('length_scale', 1.0)
            kernel = RBFKernel(ard_num_dims=input_dim)
            if length_scale != 1.0:
                kernel.lengthscale = length_scale
            kernels.append(ScaleKernel(kernel))
            
        elif kernel_type == 'white':
            # White noise kernel is handled via likelihood
            continue
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Return the first kernel or additive kernel if multiple
    if len(kernels) == 1:
        return kernels[0]
    elif len(kernels) > 1:
        return AdditiveKernel(*kernels)
    else:
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=input_dim))


def initialize_model_with_config(train_x: torch.Tensor, train_y: torch.Tensor, 
                                  bounds: torch.Tensor, config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Initialize GP model with configuration settings.
    
    Args:
        train_x: Training inputs
        train_y: Training outputs
        bounds: Problem bounds
        config: Configuration dictionary
        
    Returns:
        Tuple of (mll, model)
    """
    kernel_configs = get_kernel_config(config)
    input_dim = train_x.shape[-1]
    
    # Create covariance module from config
    covar_module = create_kernel_from_config(kernel_configs, input_dim)
    
    # Create likelihood with noise constraint
    noise_level = config.get('alpha', 0.0)
    if noise_level > 0:
        likelihood = GaussianLikelihood(noise_constraint=Interval(0, noise_level * 10.0))
    else:
        likelihood = GaussianLikelihood()
    
    # Create model
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        likelihood=likelihood,
        covar_module=covar_module
    )
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    return mll, model


def get_acquisition_function(acquisition_config: Dict[str, Any]) -> Callable:
    """
    Get the acquisition function from configuration.
    
    Args:
        acquisition_config: Acquisition function configuration
        
    Returns:
        Acquisition function class
    """
    acq_name = acquisition_config['name']
    acq_params = acquisition_config['params']
    
    if acq_name == 'qUpperConfidenceBound':
        beta = acq_params.get('beta', acq_params.get('kappa', 2.0))
        return lambda model, X_baseline: qUpperConfidenceBound(model, beta=beta)
    elif acq_name == 'qLogExpectedImprovement':
        return lambda model, X_baseline: qLogExpectedImprovement(model, best_f=X_baseline.max())
    elif acq_name == 'qLogNoisyExpectedImprovement':
        return lambda model, X_baseline: qLogNoisyExpectedImprovement(model, X_baseline=X_baseline)
    else:
        raise ValueError(f"Unknown acquisition function: {acq_name}")
