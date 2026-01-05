from botorch import fit_gpytorch_mll
from ..forgetting_to_improve.optimize import sequentially_optimize_samples, batch_optimize_samples


def filter_samples(
        model,
        mll,
        train_x,
        train_obj,
        target_region,
        algorithm='joint',
        method='sequential',
        min_samples=5,
        initialize_model=None,
        global_bounds=None,
    ):
    """
    Filter training samples based on their influence on predictions in the target region.
    
    This function implements forgetting-based sample selection methods (joint, epistemic).
    Baseline methods (relevance_pursuit, warped_gp, heteroscedastic_gp) are handled 
    separately in baseline_models.py.
    
    Args:
        model: GP model
        mll: Marginal log likelihood
        train_x: Training inputs
        train_obj: Training targets
        target_region: Target region for optimization
        algorithm: Algorithm to use ('joint' or 'epistemic')
        method: Method for optimization ('sequential' or 'batch')
        min_samples: Minimum number of samples to keep
        initialize_model: Function to reinitialize model
        global_bounds: Global bounds for the problem
        
    Returns:
        Tuple of (filtered_x, filtered_y, deleted_samples)
    """
    if algorithm not in ['joint', 'epistemic']:
        # Return unchanged for other algorithms
        return train_x, train_obj, []
    
    if method == 'sequential':
        x_samples, y_samples, deleted_x_samples = train_x, train_obj, []
        for i in range(train_x.shape[0] - min_samples):
            mll, model = initialize_model(x_samples, y_samples, global_bounds)
            try:
                # After first fit, fix noise parameter for subsequent fits
                if i > 0:
                    for param in mll.likelihood.parameters():
                        param.requires_grad = False
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"Error fitting model during filtering: {e}")
            
            if target_region.samples.shape[0] > 2:
                target_region.update(model)
            
            # Store initial count to check if a sample was removed
            initial_count = x_samples.shape[0]
            
            x_samples, y_samples, deleted_x_samples = sequentially_optimize_samples(
                model=model,
                likelihood=mll.likelihood,
                x_samples=x_samples,
                y_samples=y_samples.squeeze(-1),
                x_test=target_region.samples,
                max_iter=1,
                noise_type=algorithm,
                show_hessian=False
            )
            y_samples = y_samples.unsqueeze(-1)
            
            # Break if no sample was removed (no negative influence found)
            if x_samples.shape[0] == initial_count:
                print(f"No more samples with negative influence found. Stopping at {x_samples.shape[0]} samples.")
                break

        print(f"Filtered to {x_samples.shape[0]} samples")
    
    elif method == 'batch':
        x_samples, y_samples, deleted_x_samples = batch_optimize_samples(
            model=model,
            x_samples=train_x,
            y_samples=train_obj.squeeze(-1),
            x_test=target_region.samples,
            noise_type=algorithm,
        )
        y_samples = y_samples.unsqueeze(-1)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return x_samples, y_samples, deleted_x_samples