import torch
from typing import Dict, Any, Callable, Tuple, List
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley, Branin, Levy, HolderTable, Griewank, Hartmann, AugmentedRosenbrock
from botorch.acquisition import qUpperConfidenceBound, qLogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import GammaPrior
from gpytorch.priors.torch_priors import LogNormalPrior

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
        'botorch_ackley_2d': (Ackley(dim=2, negate=True), torch.tensor([[-10.0] * 2, [30.0] * 2], dtype=torch.float64)),
        'botorch_ackley_6d': (Ackley(dim=4, negate=True), torch.tensor([[-10.0] * 4, [30.0] * 4], dtype=torch.float64)),
        'botorch_branin': (Branin(negate=True), torch.tensor([[-5.0, 10.0], [0.0, 15.0]], dtype=torch.float64)),
        'botorch_griewank': (Griewank(negate=True), torch.tensor([[-50.0] * 2, [20.0] * 2], dtype=torch.float64)),
        'botorch_levy': (Levy(dim=2, negate=True), torch.tensor([[-10.0] * 2, [10.0] * 2], dtype=torch.float64)),
        'botorch_holder_table': (HolderTable(negate=True), torch.tensor([[-10.0] * 2, [10.0] * 2], dtype=torch.float64)),
        'botorch_hartmann_6d': (Hartmann(dim=6, negate=True), torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.float64)),
        'botorch_augmented_rosenbrock': (AugmentedRosenbrock(dim=4, negate=True), torch.tensor([[-5.0,-5.0, 0, 0], [10.0,10.0,1.0, 1.0]], dtype=torch.float64)),
    }
    
    if objective_name not in objective_map:
        raise ValueError(f"Unknown objective function: {objective_name}")
    
    return objective_map[objective_name]


def create_kernel_from_config(kernel_configs: List[Dict[str, Any]], input_dim: int,
                              train_x=None, train_y=None) -> Any:
    """
    Create a GPyTorch kernel from configuration.

    Args:
        kernel_configs: List of kernel configuration dictionaries
        input_dim: Input dimensionality
        train_x, train_y: Optional current BO data. Only used by the 'jacobian'
            (last-layer linearised-Laplace) kernel, whose MLP is (re)trained on
            them. Ignored by matern.

    Returns:
        GPyTorch kernel module
    """
    if not kernel_configs:
        # Default: Matern 2.5 kernel with LogNormal(0,1) prior for lengthscale
        kernel = MaternKernel(
            nu=2.5, 
            ard_num_dims=input_dim,
            lengthscale_prior=LogNormalPrior(0, 1),
            lengthscale_constraint=GreaterThan(1e-4)
        )
        return kernel
    
    kernels = []
    for kernel_config in kernel_configs:
        kernel_type = kernel_config.get('type', 'matern')
        
        if kernel_type == 'matern':
            nu = kernel_config.get('nu', 2.5)
            length_scale = kernel_config.get('length_scale', 1.0)
            # LogNormal(0,1) prior for lengthscale
            kernel = MaternKernel(
                nu=nu, 
                ard_num_dims=input_dim,
                lengthscale_prior=LogNormalPrior(0, 1),
                lengthscale_constraint=GreaterThan(1e-4)
            )
            if length_scale != 1.0:
                kernel.lengthscale = length_scale
            kernels.append(kernel)
            
        elif kernel_type == 'jacobian':
            # Last-layer linearised-Laplace (Jacobian) kernel: k(x,x') = phi(x) Sigma phi(x')^T.
            # The MLP is retrained on the CURRENT BO data every time the model is
            # re-initialised, so the features adapt as the loop acquires points.
            from ...forgetting_to_improve.helper.nn_kernel import build_jacobian_kernel
            if train_x is None:
                raise ValueError("jacobian kernel requires train_x/train_y to fit its MLP")
            jac, _mlp = build_jacobian_kernel(
                kernel_config, input_dim,
                train_x=train_x.detach().cpu().numpy(),
                train_y=train_y.detach().cpu().numpy().ravel(),
            )
            # Jacobian features have arbitrary amplitude (no lengthscale), so the
            # outputscale is the free scale -- fitted by fit_gpytorch_mll.
            jac = ScaleKernel(jac).to(train_x.dtype)
            kernels.append(jac)

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
        # Fallback with LogNormal(0,1) prior for lengthscale
        kernel = MaternKernel(
            nu=2.5, 
            ard_num_dims=input_dim,
            lengthscale_prior=LogNormalPrior(0, 1),
            lengthscale_constraint=GreaterThan(1e-4)
        )
        return kernel


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
    
    # Create covariance module from config. train_x/train_y are forwarded so the
    # jacobian kernel can (re)fit its MLP on the current data; no-op for matern.
    covar_module = create_kernel_from_config(kernel_configs, input_dim, train_x, train_y)
    
    # Likelihood: Gamma(2, 0.05) noise prior, noise bounded by 10 * alpha
    noise_level = config.get('alpha', 0.0)
    if noise_level > 0:
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(2, 0.05),
            noise_constraint=Interval(1e-6, noise_level * 10.0)
        )
    else:
        # Noise-free objective: keep a small noise for numerical stability
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(2, 0.05),
            noise_constraint=Interval(1e-6, 1e-2)
        )
    
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
    else:
        raise ValueError(f"Unknown acquisition function: {acq_name}")
