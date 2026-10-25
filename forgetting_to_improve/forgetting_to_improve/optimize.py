import torch
import gpytorch
from typing import Tuple, Optional, List, Dict
from itertools import combinations
from .uncertainty import (
    calculate_uncertainty_influence,
    solve_map_noise,
    calculate_aleatoric_hessian,
    calculate_epistemic_hessian,
)
from .curvature_bounds import (
    calculate_restricted_curvature_bounds, compute_J, compute_upper_bound_U,
    check_gradient_dominance,
)
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


def _set_noise_clamped(likelihood, noise: float):
    """Set likelihood.noise, clamped into its constraint (warns if clamped)."""
    constraint = likelihood.noise_covar.raw_noise_constraint
    lo, hi = float(constraint.lower_bound), float(constraint.upper_bound)
    clamped = min(max(noise, lo * (1 + 1e-6)), hi * (1 - 1e-6))
    if clamped != noise:
        print(f"Warning: MAP noise {noise:.4e} outside constraint [{lo:.1e}, {hi:.1e}], "
              f"clamped to {clamped:.4e}; removal criterion != dJ/dh at this step")
    likelihood.noise = clamped


def select_worst_sample(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    noise_type: str = 'joint',
    removal_threshold: float = 0.0,
    return_criterion: bool = False,
    use_prior_a: bool = False
):
    """
    Select the sample to potentially remove.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        noise_type: Type of uncertainty to consider ('joint', 'epistemic', 'aleatoric',
            'aleatoric_scaled' = factor * aleatoric, i.e. joint without the epistemic term)
        removal_threshold: Only remove if the removal gain -influence exceeds
            this value (0 recovers the plain "influence < 0" criterion)
        return_criterion: Additionally return the per-sample criterion (n_train,)
        use_prior_a: See calculate_aleatoric_uncertainty_influence
        
    Returns:
        min_index: Index of worst sample (or None if no gain > removal_threshold)
        epistemic_influence: Epistemic uncertainty influence
        aleatoric_influence: Aleatoric uncertainty influence
    """
    epistemic_influence, aleatoric_influence, factor = (
        calculate_uncertainty_influence(model, x_samples, y_samples, x_test,
                                        use_prior_a=use_prior_a)
    )
    
    if noise_type == 'joint':
        total = epistemic_influence + aleatoric_influence * factor
    elif noise_type == 'epistemic':
       total = epistemic_influence
    elif noise_type == 'aleatoric':
        total = aleatoric_influence
    elif noise_type == 'aleatoric_scaled':
        # Joint criterion without the epistemic term: same factor on the
        # aleatoric influence, so joint = epistemic + this.
        total = aleatoric_influence * factor
    else:
        raise ValueError("Invalid noise_type. Choose from 'joint', 'epistemic', 'aleatoric' or 'aleatoric_scaled'.")
    
    min_index = int(total.argmin())
    if -total[min_index] <= removal_threshold:
        min_index = None
    if return_criterion:
        return min_index, epistemic_influence, aleatoric_influence, total
    return min_index, epistemic_influence, aleatoric_influence


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
    A_limits: Tuple = None,
    removal_threshold: float = 0.0,
    map_noise: bool = False
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
        removal_threshold: Minimum removal gain -influence required to remove a sample
        map_noise: Before each greedy step, set the likelihood noise to the MAP-II
            value for the current subset (and use the prior's actual a in the
            aleatoric term), so the removal criterion equals dJ/dh_i of the J
            used in the curvature analysis. The original noise is restored
            afterwards.
        
    Returns:
        x_samples: Optimized training inputs
        y_samples: Optimized training targets
        deleted_x_samples: List of removed samples
        curvature_history: Dictionary with keys 'm' (diagonal curvature of the
            point removed at that step), 'lambda_min'/'lambda_max' (spectral
            bounds of the submatrix restricted to that step's *and all later*
            removals -- R_k = T ∩ S^(k), see calculate_restricted_curvature_bounds),
            'J'/'g' (objective J(S^(k)) and g_k = dJ/dh_i of the sample removed
            at step k, both at the MAP-II noise), 'criterion' (the removal
            criterion value of that sample, at the model's likelihood.noise),
            'rho' (per-step rho_k = 2 M_R [J(S^(k)) - J(S^(k+1))] / g_k^2 with
            M_R = max_k -lambda_min(H_{R_k}); rho_hat = min_k rho_k), 'M_R',
            and 'n_samples'. Also 'm_R' (= min_k -lambda_max(H_{R_k})), 'U'
            (U_q = min{J(S), sum_{i<=q} g_(i)^(0) + m_R q/2}, q = 0..|T|),
            'J_bar' (greedy gain J(S) - J(S^(q)), q = 0..|T|) and
            'n_samples_q' (|S| - q), and the gradient-dominance assumption
            per step k = 0..q-1: 'grad_inf' (||u^(k)+v^(k)||_inf),
            'assumption_lhs', 'assumption_rhs', 'assumption_met'. Empty lists if calculate_convexity=False.
    """
    deleted_x_samples = []
    deleted_y_samples = []
    curvature_history = {'m': [], 'lambda_min': [], 'lambda_max': [], 'n_samples': [],
                         'J': [], 'g': [], 'criterion': [], 'rho': [],
                         'U': [], 'J_bar': [], 'n_samples_q': [], 'grad_inf': [],
                         'assumption_lhs': [], 'assumption_rhs': [], 'assumption_met': []}

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
    removal_criterion = []

    original_noise = model.likelihood.noise.detach().clone() if map_noise else None

    for iteration in range(max_iter):
        if map_noise:
            _set_noise_clamped(model.likelihood, solve_map_noise(model, x_samples, y_samples))

        index_to_remove, epistemic, aleatoric, criterion = select_worst_sample(
            model, x_samples, y_samples, x_test, noise_type=noise_type,
            removal_threshold=removal_threshold, return_criterion=True,
            use_prior_a=map_noise
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
        removal_criterion.append(criterion[index_to_remove].item())

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
    # g_k for rho_k is dJ/dh_i at the MAP-II noise (same function as J and
    # H_R), not the removal criterion, which is evaluated at the model's
    # likelihood.noise; the latter is kept in 'criterion' for comparison.
    if calculate_convexity and removal_order:
        x_r = x_samples_0
        y_r = y_samples_0
        idx_r = orig_idx.new_tensor(range(len(x_samples_0)))
        grad_Jbar_0 = None  # full grad_h Jbar at S (step 0), for U_q

        for k, removed_orig_idx in enumerate(removal_order):
            current_pos = int((idx_r == removed_orig_idx).nonzero(as_tuple=True)[0].item())
            future_orig = set(removal_order[k:])
            removal_indices = [i for i, oi in enumerate(idx_r.tolist()) if oi in future_orig]

            try:
                model.set_train_data(x_r, y_r, strict=False)
                m_diag, lambda_min, lambda_max, J_k, g_k, grad_k = calculate_restricted_curvature_bounds(
                    model, x_r, y_r, x_test, removal_indices, current_index=current_pos,
                    return_full_grad=True
                )
                if k == 0:
                    grad_Jbar_0 = grad_k
                curvature_history['grad_inf'].append(grad_k.abs().max().item())
                curvature_history['m'].append(m_diag)
                curvature_history['lambda_min'].append(lambda_min)
                curvature_history['lambda_max'].append(lambda_max)
                curvature_history['n_samples'].append(len(x_r))
                curvature_history['J'].append(J_k)
                curvature_history['g'].append(g_k)
                curvature_history['criterion'].append(removal_criterion[k])
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

        # rho_k needs J(S^(k+1)), i.e. also J on the final retained set.
        # Only meaningful if every step succeeded (lists aligned with steps).
        if len(curvature_history['J']) == len(removal_order):
            try:
                model.set_train_data(x_r, y_r, strict=False)
                J_seq = curvature_history['J'] + [compute_J(model, x_r, y_r, x_test)]
                M_R = max(-lm for lm in curvature_history['lambda_min'])
                curvature_history['M_R'] = M_R
                curvature_history['rho'] = [
                    2.0 * M_R * (J_seq[k] - J_seq[k + 1]) / max(g ** 2, 1e-30)
                    for k, g in enumerate(curvature_history['g'])
                ]
                rho = curvature_history['rho']
                k_min = min(range(len(rho)), key=rho.__getitem__)
                print(f"M_R={M_R:.6e}, rho_hat=min_k rho_k={rho[k_min]:.6e} at step {k_min + 1}")
            except Exception as e:
                print(f"Warning: Could not compute rho_k: {e}")

            # Upper bound U_q on the best q-removal gain Jbar(T*), against the
            # greedy gain Jbar(T_q) = J(S) - J(S^(q)). J(S) in the min is the
            # mean predictive variance (J + mean k(x,x)), which is >= 0.
            try:
                m_R = min(-lm for lm in curvature_history['lambda_max'])
                with torch.no_grad():
                    prior_var = model.covar_module(x_test, diag=True).mean().item()
                q_max = len(removal_order)
                curvature_history['m_R'] = m_R
                curvature_history['U'] = compute_upper_bound_U(
                    J_seq[0] + prior_var, grad_Jbar_0, m_R, q_max)
                curvature_history['J_bar'] = [J_seq[0] - J_q for J_q in J_seq]
                curvature_history['n_samples_q'] = [len(x_samples_0) - q for q in range(q_max + 1)]
                lhs, rhs, met = check_gradient_dominance(
                    curvature_history['grad_inf'], curvature_history['J_bar'],
                    curvature_history['U'][-1], m_R)
                curvature_history['assumption_lhs'] = lhs
                curvature_history['assumption_rhs'] = rhs
                curvature_history['assumption_met'] = met
                print(f"Gradient-dominance assumption met at {sum(met)}/{len(met)} steps "
                      f"(m_R={m_R:.3e}): {met}")
                print(f"m_R={m_R:.6e}, U_|T|={curvature_history['U'][-1]:.6e}, "
                      f"greedy Jbar(T)={curvature_history['J_bar'][-1]:.6e}")
            except Exception as e:
                print(f"Warning: Could not compute U_q: {e}")

        # Pass 2's last set_train_data call left the model on the
        # second-to-last subset; restore it to the actual final state
        # returned below.
        model.set_train_data(x_samples, y_samples, strict=False)
        model.train()
        likelihood.train()

    if map_noise:
        model.likelihood.noise = original_noise

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
    elif noise_type == 'aleatoric_scaled':
        total = aleatoric * factor
    else:
        raise ValueError("Invalid noise_type. Choose from 'joint', 'epistemic', 'aleatoric' or 'aleatoric_scaled'.")
    
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