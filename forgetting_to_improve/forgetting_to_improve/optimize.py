import torch
import gpytorch
from typing import Tuple, Optional, List, Dict
from itertools import combinations
from .uncertainty import (
    calculate_uncertainty_influence,
    calculate_aleatoric_hessian,
    calculate_epistemic_hessian,
)
from .curvature_bounds import calculate_restricted_curvature_bounds
from .helper.plot_gp import visualize_hessian, plot_optimal_comparison


def _finetune_kernel_mlp(model, x_samples, y_samples):
    """Re-fit the NN (jacobian) kernel's MLP on the current retained set.

    ``jac_finetune['mode']`` selects between continuing from theta* ('finetune')
    and re-initialising the weights before training ('scratch'), the latter
    making the re-fit independent of the features learned on the removed points.

    No-op unless the covar_module carries a jacobian MLP (``jac_mlp``) and a
    positive ``jac_finetune['epochs']`` -- so this is safe for RBF/matern kernels.
    Since JacobianKernel and MLPMean share this MLP object and read it live, an
    in-place update adapts both the kernel geometry and the GP mean after removal.
    """
    covar = model.covar_module
    mlp = getattr(covar, 'jac_mlp', None)
    ft = getattr(covar, 'jac_finetune', None)
    if mlp is None or not ft or ft.get('epochs', 0) <= 0:
        return

    epochs, lr = ft['epochs'], ft['lr']
    if ft.get('mode', 'finetune') == 'scratch':
        from .helper.nn_kernel import reset_parameters
        reset_parameters(mlp)
    x = x_samples if x_samples.ndim == 2 else x_samples.unsqueeze(-1)
    mdtype = next(mlp.parameters()).dtype
    x = x.to(mdtype)
    y = y_samples.reshape(-1, 1).to(mdtype)

    for p in mlp.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(mlp(x), y)
        loss.backward()
        opt.step()
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    # Refresh the detached param snapshot used by the full-jacobian feature path.
    base = getattr(covar, 'base_kernel', covar)
    if hasattr(base, 'params'):
        base.params = {k: v.detach() for k, v in mlp.named_parameters()}


def select_worst_sample(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    noise_type: str = 'joint'
) -> Tuple[Optional[int], torch.Tensor, torch.Tensor]:
    """
    Select the sample to potentially remove.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        noise_type: Type of uncertainty to consider ('joint', 'epistemic', 'aleatoric')
        
    Returns:
        min_index: Index of worst sample (or None if no negative influence)
        epistemic_influence: Epistemic uncertainty influence
        aleatoric_influence: Aleatoric uncertainty influence
    """
    epistemic_influence, aleatoric_influence, factor = (
        calculate_uncertainty_influence(model, x_samples, y_samples, x_test)
    )
    
    if noise_type == 'joint':
        total = epistemic_influence + aleatoric_influence * factor
    elif noise_type == 'epistemic':
       total = epistemic_influence
    elif noise_type == 'aleatoric':
        total = aleatoric_influence
    else:
        raise ValueError("Invalid noise_type. Choose from 'joint', 'epistemic', or 'aleatoric'.")
    
    min_index = int(total.argmin())
    if total[min_index] < 0:
        return min_index, epistemic_influence, aleatoric_influence
    
    return None, epistemic_influence, aleatoric_influence


def sequentially_optimize_samples(
    model: gpytorch.models.GP,
    likelihood: gpytorch.likelihoods.Likelihood,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    max_iter: int = 20,
    noise_type: str = 'joint',
    show_hessian: bool = False,
    compare_optimal: bool = False,
    calculate_convexity: bool = False,
    A_limits: Tuple = None
) -> Tuple[torch.Tensor, torch.Tensor, List, Dict]:
    """
    Optimize the training samples by removing those that negatively impact the GP model.
    
    Args:
        model: GPyTorch GP model
        likelihood: GPyTorch likelihood
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        max_iter: Maximum iterations
        noise_type: Type of uncertainty to consider
        show_hessian: Whether to visualize Hessian (only for 1D)
        calculate_convexity: Whether to compute curvature bound m
        A_limits: Region limits for curvature bound calculation (required if calculate_convexity=True)
        
    Returns:
        x_samples: Optimized training inputs
        y_samples: Optimized training targets
        deleted_x_samples: List of removed samples
        curvature_history: Dictionary with keys 'm' (diagonal curvature of the
            point removed at that step), 'lambda_min'/'lambda_max' (spectral
            bounds of the submatrix restricted to that step's *and all later*
            removals -- R_k = T ∩ S^(k), see calculate_restricted_curvature_bounds),
            and 'n_samples'. Empty lists if calculate_convexity=False.
    """
    deleted_x_samples = []
    deleted_y_samples = []
    curvature_history = {'m': [], 'lambda_min': [], 'lambda_max': [], 'n_samples': []}

    # Store original full dataset for optimal comparison
    if compare_optimal:
        original_x = x_samples.clone()
        original_y = y_samples.clone()
        fti_test_uncertainty = []
        optimal_test_uncertainty = []

    # Compute initial curvature bounds if requested
    if calculate_convexity:
        if A_limits is None:
            raise ValueError("A_limits must be provided when calculate_convexity=True")

    # Curvature bounds need the *full* eventual removal set T (R_k = T ∩ S^(k)
    # at every step k, see Section M / Lemma 5), which is only known once the
    # greedy removal loop below has run to completion. So the loop below only
    # records *which original sample* is removed at each step (removal_order,
    # via orig_idx tracking each row's position in the initial x_samples/
    # y_samples); the actual H_R(S^(k)) submatrices are computed afterwards
    # in a second, retrospective pass that replays this exact removal order.
    x_samples_0 = x_samples.clone()
    y_samples_0 = y_samples.clone()
    orig_idx = torch.arange(len(x_samples), device=x_samples.device)
    removal_order = []

    for iteration in range(max_iter):
        index_to_remove, epistemic, aleatoric = select_worst_sample(
            model, x_samples, y_samples, x_test, noise_type=noise_type
        )

        if show_hessian and x_samples.shape[1] == 1:
            # Only show hessian for 1D case
            H = calculate_aleatoric_hessian(y_samples, model, x_samples)
            visualize_hessian(H, aleatoric, epistemic, index_to_remove,
                              title=f"Hessian Iteration {iteration + 1}_{noise_type}")

        if index_to_remove is None:
            break

        deleted_x_samples.append(x_samples[index_to_remove].clone())
        deleted_y_samples.append(y_samples[index_to_remove].clone())
        removal_order.append(int(orig_idx[index_to_remove].item()))

        # Remove sample
        mask = torch.ones(len(x_samples), dtype=torch.bool, device=x_samples.device)
        mask[index_to_remove] = False
        x_samples = x_samples[mask]
        y_samples = y_samples[mask]
        orig_idx = orig_idx[mask]

        if len(x_samples) <= 1:
            print("Cannot remove more samples (would leave less than 2 samples)")
            break

        if compare_optimal:
            # Store FTI uncertainty after this removal (before refitting)
            # We need to refit the model to the current subset first
            model.set_train_data(x_samples, y_samples, strict=False)
            model.eval()
            likelihood.eval()
            
            with torch.no_grad():
                sigma_x_test = model(x_test).variance.mean().item()
            fti_test_uncertainty.append(sigma_x_test)
            
            # Find optimal subset of current size by exhaustive search
            n_current = len(x_samples)
            n_total = len(original_x)
            
            print(f"\n=== Iteration {iteration + 1}: Finding optimal subset of size {n_current} ===")
            
            best_uncertainty = float('inf')
            
            # Try all possible subsets of size n_current
            n_combinations = 0
            for subset_indices in combinations(range(n_total), n_current):
                n_combinations += 1
                subset_indices = list(subset_indices)
                subset_x = original_x[subset_indices]
                subset_y = original_y[subset_indices]
                
                # Evaluate model with this subset (use current hyperparameters, don't refit)
                model.set_train_data(subset_x, subset_y, strict=False)
                model.eval()
                likelihood.eval()
                
                with torch.no_grad():
                    subset_uncertainty = model(x_test).variance.mean().item()
                
                if subset_uncertainty < best_uncertainty:
                    best_uncertainty = subset_uncertainty
                        
            
            optimal_test_uncertainty.append(best_uncertainty)
            
        # For the NN (jacobian) kernel: fine-tune the MLP on the reduced dataset
        # for a few epochs so the kernel adapts after this removal (no-op otherwise).
        _finetune_kernel_mlp(model, x_samples, y_samples)

        # Refit model (just set training data, keep eval mode since we don't reoptimize hyperparameters)
        model.set_train_data(x_samples, y_samples, strict=False)
        model.train()
        likelihood.train()

    # Retrospective Pass 2: now that the full removal order T is known, replay
    # it from the original dataset and, at each step k, compute curvature
    # restricted to R_k = T ∩ S^(k) -- this step's removal *and* every later
    # one still present -- rather than only the diagonal entry of the single
    # point picked at step k (see calculate_restricted_curvature_bounds).
    # The greedy removal decisions themselves are unaffected: they were
    # already made above using the cheap first-order influence measures.
    if calculate_convexity and removal_order:
        x_r = x_samples_0
        y_r = y_samples_0
        idx_r = orig_idx.new_tensor(range(len(x_samples_0)))

        for k, removed_orig_idx in enumerate(removal_order):
            current_pos = int((idx_r == removed_orig_idx).nonzero(as_tuple=True)[0].item())
            future_orig = set(removal_order[k:])
            removal_indices = [i for i, oi in enumerate(idx_r.tolist()) if oi in future_orig]

            try:
                model.set_train_data(x_r, y_r, strict=False)
                m_diag, lambda_min, lambda_max = calculate_restricted_curvature_bounds(
                    model, x_r, y_r, x_test, removal_indices, current_index=current_pos
                )
                curvature_history['m'].append(m_diag)
                curvature_history['lambda_min'].append(lambda_min)
                curvature_history['lambda_max'].append(lambda_max)
                curvature_history['n_samples'].append(len(x_r))
                sample_coords = x_r[current_pos].detach().cpu().numpy()
                print(f"Removal step {k + 1}/{len(removal_order)}: sample at x={sample_coords}, "
                      f"H[i,i]={m_diag:.6e}, restricted to {len(removal_indices)} future removal(s): "
                      f"lambda_min={lambda_min:.6e}, lambda_max={lambda_max:.6e}, n_samples={len(x_r)}")
            except Exception as e:
                print(f"Warning: Could not compute restricted curvature bounds at step {k + 1}: {e}")

            mask = torch.ones(len(x_r), dtype=torch.bool, device=x_r.device)
            mask[current_pos] = False
            x_r = x_r[mask]
            y_r = y_r[mask]
            idx_r = idx_r[mask]

        # Pass 2's last set_train_data call left the model on the
        # second-to-last subset; restore it to the actual final state
        # returned below.
        model.set_train_data(x_samples, y_samples, strict=False)
        model.train()
        likelihood.train()

    # Summary of comparison
    if compare_optimal:
        print("\n=== Final Comparison: FTI vs Optimal Subset Selection ===")
        print(f"{'Subset Size':<15} {'FTI Uncertainty':<20} {'Optimal Uncertainty':<20} {'Difference':<15}")
        print("-" * 70)
        
        # Build list of subset sizes (decreasing)
        n_samples_list = []
        for i in range(len(fti_test_uncertainty)):
            subset_size = len(original_x) - i - 1
            n_samples_list.append(subset_size)
            diff = fti_test_uncertainty[i] - optimal_test_uncertainty[i]
            print(f"{subset_size:<15} {fti_test_uncertainty[i]:<20.6f} {optimal_test_uncertainty[i]:<20.6f} {diff:<15.6f}")
        
        # Create comparison plot
        plot_optimal_comparison(
            fti_uncertainty=fti_test_uncertainty,
            optimal_uncertainty=optimal_test_uncertainty,
            n_samples_list=n_samples_list,
            title="FTI_vs_Optimal_Subset_Selection"
        )
    
    return x_samples, y_samples, deleted_x_samples, curvature_history


def batch_optimize_samples(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    noise_type: str = 'joint'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch optimization of training samples.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        noise_type: Type of uncertainty to consider
        
    Returns:
        x_samples: Optimized training inputs
        y_samples: Optimized training targets
    """
    epistemic, aleatoric, factor = calculate_uncertainty_influence(
        model, x_samples, y_samples, x_test
    )
    
    if noise_type == 'joint':
        total = epistemic + aleatoric * factor
    elif noise_type == 'epistemic':
        total = epistemic
    elif noise_type == 'aleatoric':
        total = aleatoric
    else:
        raise ValueError("Invalid noise_type. Choose from 'joint', 'epistemic', or 'aleatoric'.")
    
    negative_indices = torch.where(total < 0)[0]
    if len(negative_indices) > 0:
        print(f"Removing {len(negative_indices)} samples with negative influence")
        mask = torch.ones(len(x_samples), dtype=torch.bool, device=x_samples.device)
        mask[negative_indices] = False
        x_deleted = x_samples[~mask]
        y_deleted = y_samples[~mask]
        x_samples = x_samples[mask]
        y_samples = y_samples[mask]
        
    else:
        print("No samples with negative influence found")
        x_deleted = torch.tensor([], device=x_samples.device)
        y_deleted = torch.tensor([], device=y_samples.device)
    
    print(f"Final number of samples: {len(x_samples)}")
    return x_samples, y_samples, x_deleted, y_deleted