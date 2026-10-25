import gpytorch
from typing import  List


def create_kernel_from_config(kernel_config: List[dict], n_dims: int, fixed_kernel: bool = True,
                              device: str = 'cpu', train_x=None, train_y=None):
    """Create a GPyTorch kernel from configuration.

    train_x/train_y are optional and only used by the 'jacobian' kernel to train
    the underlying MLP (theta*); passing them is a no-op for all other kernels.
    """
    if kernel_config is None:
        # Default configuration
        if n_dims == 1:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        else:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
            )
        return kernel, None
    
    kernels = []
    white_noise_level = None
    jac_mlp = None
    jac_finetune = None

    for config in kernel_config:
        kernel_type = config.get('type', 'rbf').lower()
        
        if kernel_type == 'rbf':
            length_scale = config.get('length_scale', 1.0)
            if n_dims == 1:
                k = gpytorch.kernels.RBFKernel()
            else:
                k = gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
            
            if fixed_kernel and 'length_scale' in config:
                k.lengthscale = length_scale
                k.raw_lengthscale.requires_grad = False
            
            kernels.append(k)
        
        elif kernel_type == 'matern':
            nu = config.get('nu', 2.5)
            length_scale = config.get('length_scale', 1.0)
            if n_dims == 1:
                k = gpytorch.kernels.MaternKernel(nu=nu)
            else:
                k = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=n_dims)
            
            if fixed_kernel and 'length_scale' in config:
                k.lengthscale = length_scale
                k.raw_lengthscale.requires_grad = False
            
            kernels.append(k)
        
        elif kernel_type == 'periodic':
            length_scale = config.get('length_scale', 1.0)
            period_length = config.get('period_length', 1.0)
            if n_dims == 1:
                k = gpytorch.kernels.PeriodicKernel()
            else:
                k = gpytorch.kernels.PeriodicKernel(ard_num_dims=n_dims)

            if fixed_kernel:
                if 'length_scale' in config:
                    k.lengthscale = length_scale
                    k.raw_lengthscale.requires_grad = False
                if 'period_length' in config:
                    k.period_length = period_length
                    k.raw_period_length.requires_grad = False

            kernels.append(k)

        elif kernel_type == 'linear' or kernel_type == 'dot_product':
            k = gpytorch.kernels.LinearKernel()
            kernels.append(k)

        elif kernel_type == 'jacobian':
            # Linearised-Laplace (Jacobian) kernel; see helper/nn_kernel (mirrors laplax).
            from .nn_kernel import build_jacobian_kernel
            k, jac_mlp = build_jacobian_kernel(config, n_dims, device=device,
                                               train_x=train_x, train_y=train_y)
            # Optional per-removal re-fitting of the MLP on the reduced dataset.
            # 'finetune' continues training from theta*, 'scratch' re-initialises
            # the weights first so the features are re-learned independently.
            mode = str(config.get('retrain_mode', 'finetune')).lower()
            if mode not in ('finetune', 'scratch'):
                raise ValueError(
                    f"jacobian kernel: unknown retrain_mode '{mode}'. "
                    "Options: 'finetune', 'scratch'"
                )
            default_epochs = config.get('epochs', 500) if mode == 'scratch' else 0
            jac_finetune = {
                'mode': mode,
                'epochs': config.get('finetune_epochs', default_epochs),
                'lr': config.get('finetune_lr', config.get('lr', 1e-2)),
            }
            kernels.append(k)
        
        elif kernel_type == 'white':
            # White noise kernel - homoscedastic noise model
            # In GPyTorch, white noise is handled by the likelihood, not as a kernel
            noise_level = config.get('noise_level', 0.1)
            white_noise_level = noise_level
            # Don't add to kernels list - will be set in likelihood
        
        elif kernel_type == 'scale':
            # ScaleKernel wraps another kernel
            continue
        
        else:
            print(f"Warning: Unknown kernel type '{kernel_type}', skipping")
    
    # Combine kernels
    if len(kernels) == 0:
        # Default to RBF
        kernel = gpytorch.kernels.RBFKernel()
    elif len(kernels) == 1:
        kernel = kernels[0]
    else:
        # Sum kernels
        kernel = kernels[0]
        for k in kernels[1:]:
            kernel = kernel + k
    
    # Wrap in ScaleKernel (outputscale acts as 1/tau for the jacobian kernel)
    kernel = gpytorch.kernels.ScaleKernel(kernel)

    # Expose the frozen MLP so callers can build the matching MLPMean (linearisation centre),
    # plus any per-removal fine-tuning settings read by the sequential optimizer.
    if jac_mlp is not None:
        kernel.jac_mlp = jac_mlp
        kernel.jac_finetune = jac_finetune

    return kernel, white_noise_level