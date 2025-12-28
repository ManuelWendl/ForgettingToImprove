import gpytorch
from typing import  List


def create_kernel_from_config(kernel_config: List[dict], n_dims: int, fixed_kernel: bool = True, device: str = 'cpu'):
    """Create a GPyTorch kernel from configuration."""
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
        
        elif kernel_type == 'linear' or kernel_type == 'dot_product':
            k = gpytorch.kernels.LinearKernel()
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
    
    # Wrap in ScaleKernel
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    
    return kernel, white_noise_level