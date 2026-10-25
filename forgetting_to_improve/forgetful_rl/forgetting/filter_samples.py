import torch
import numpy as np
from typing import Tuple, Optional, Dict
import sys
sys.path.append('../../../')

import gpytorch

from forgetting_to_improve.forgetting_to_improve.optimize import sequentially_optimize_samples, batch_optimize_samples
from ..data.buffer import ReplayBuffer
from .target_region import TargetRegion


def filter_replay_buffer(
    replay_buffer: ReplayBuffer,
    model,
    likelihood,
    target_region: TargetRegion,
    max_iter: int = 20,
    noise_type: str = 'joint',
    show_hessian: bool = False,
    compare_optimal: bool = False,
    batch_optimize: bool = True,
    max_samples: int = 500,  # Maximum samples to avoid OOM
    store_data_removal: bool = False,
) -> Tuple[ReplayBuffer, Dict, Optional[Dict]]:
    """
    Filter the replay buffer by removing samples that negatively impact the dynamics model.

    For MultitaskGP models, processes each output dimension separately and removes samples
    that are detrimental for ANY dimension.

    Reward is NOT one of those dimensions -- it is excluded entirely (not filtered on
    at all), and is expected to no longer be part of what the dynamics model predicts
    in the first place: reward is often (near-)constant per env step (e.g. CartPole
    balance reward=1/step) or otherwise poorly explained by a rank-deficient linear
    kernel (e.g. PETS cartpole's dense reward), which made the curvature-based
    influence score negative for ~95-100% of samples regardless of noise prior/init
    (diagnosed empirically on both, see chat), and even a bounded fixed-fraction
    workaround tried here previously never converged to "nothing needs removing" the
    way the state-delta dimensions do, perpetually capping the buffer size instead.
    The actual fix lives upstream: see `algorithm/mbpo.py`'s `learn_reward` /
    `reward_fn` wiring and `model/gp_model.py`'s `learn_reward` flag -- when an
    analytic reward_fn is supplied (`overrides.learned_rewards: false`), the model
    only predicts state deltas, reward is computed analytically from the predicted
    next state, and this function's `model`/`y_samples` never include a reward
    dimension to begin with.

    The filtering is based on:
    1. Target region states (from SAC buffer) define x_test - the region we care about
    2. Real environment data (from replay_buffer) provides training samples
    3. Sequential optimization removes detrimental samples from replay_buffer

    Args:
        replay_buffer: ReplayBuffer containing real environment observations
        model: Trained dynamics model (GP or ensemble)
        likelihood: GPyTorch likelihood (for GP models)
        target_region: TargetRegion object determining which states matter (x_test)
        max_iter: Maximum iterations for sequential optimization
        noise_type: Type of uncertainty ('joint', 'epistemic', 'aleatoric')
        show_hessian: Whether to visualize Hessian (only for 1D)
        compare_optimal: Whether to compare with optimal subset selection
        max_samples: Maximum number of samples to process (subsample if larger to avoid OOM)

    Returns:
        filtered_buffer: New ReplayBuffer with detrimental samples removed
        filter_stats: Dictionary with filtering statistics
    """
    
    # Get all data from replay buffer (real environment observations)
    all_data = replay_buffer.get_all()
    original_size = len(all_data)
    
    # Check if buffer is too large and subsample if needed
    if original_size > max_samples:
        print(f"[Filter] Buffer too large ({original_size}), subsampling to {max_samples} samples")
        # Random subsample
        indices = np.random.choice(original_size, max_samples, replace=False)
        indices = np.sort(indices)  # Keep order
        all_data = all_data[indices]
    
    # Prepare training data: state-action pairs -> next_state deltas
    obs = torch.FloatTensor(all_data.obs)
    actions = torch.FloatTensor(all_data.act)
    next_obs = torch.FloatTensor(all_data.next_obs)

    # Concatenate obs and actions as inputs
    x_samples = torch.cat([obs, actions], dim=1)

    # Target: delta obs only -- reward is excluded entirely (see docstring), matching
    # what the dynamics model actually predicts when learn_reward=False.
    y_samples = next_obs - obs
    
    # Get test points from target region (states we care about)
    target_states = torch.FloatTensor(target_region.get_target_region_states())
    
    # Subsample test points if too large to avoid memory issues
    if len(target_states) > 3000:
        print(f"[Filter] Target region too large ({len(target_states)}), subsampling to {3000} test points")
        test_indices = np.random.choice(len(target_states), 3000, replace=False)
        target_states = target_states[test_indices]
    
    # For x_test, we need state-action pairs. Sample actions from random or use mean action
    # Here we'll create a grid by pairing each target state with mean actions from buffer
    mean_action = torch.FloatTensor(all_data.act).mean(dim=0, keepdim=True)
    x_test = torch.cat([
        target_states, 
        mean_action.repeat(len(target_states), 1)
    ], dim=1)
    
    # Keep everything on GPU (model's current device)
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda:0')
    x_samples = x_samples.to(device)
    y_samples = y_samples.to(device)
    x_test = x_test.to(device)
    
    # Store model state
    model_was_training = model.training
    
    # Store original training data for MultitaskGP models
    original_train_inputs = None
    original_train_targets = None
    is_multitask = hasattr(model, 'num_tasks')
    # Shared-trunk jacobian (last-layer Laplace) model -- see model/jacobian_model.py.
    # Additive: mutually exclusive with is_multitask, existing behaviour for
    # is_multitask / plain single-task models below is untouched.
    is_shared_jacobian = hasattr(model, 'gp_models') and hasattr(model, 'shared_mlp')

    if is_multitask and hasattr(model, 'train_inputs'):
        # Store the original training data
        original_train_inputs = model.train_inputs
        original_train_targets = model.train_targets

    # Store sampled size
    sampled_size = len(x_samples)

    # Process each output dimension separately (state-delta dims only -- reward is
    # excluded entirely, see docstring).
    # For MultitaskGP, we need to create single-task GPs for each dimension.
    filtered_indices_list = []

    if is_shared_jacobian:
        _model_kind = ' (shared-trunk JacobianGPModel)'
    elif is_multitask:
        _model_kind = ' (MultitaskGP)'
    else:
        _model_kind = ''
    print(f"[Filter] Processing {y_samples.shape[1]} output dimensions{_model_kind}")

    for dim in range(y_samples.shape[1]):
        y_dim = y_samples[:, dim]

        # Create a single-task GP for this dimension
        if is_multitask:
            # Create single-task GP
            single_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            single_likelihood.noise_covar.noise = likelihood.task_noises[dim].clone().detach()
            
            class SingleTaskGP(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super().__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.LinearKernel()
                    )
                
                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
            # Create and initialize single-task GP
            dim_model = SingleTaskGP(x_samples, y_dim, single_likelihood).to(device)
            dim_model.eval()
            single_likelihood.eval()

        elif is_shared_jacobian:
            # Reuse this dimension's own GP (shared trunk + its own
            # JacobianKernel/MLPMeanIndexed) directly, instead of rebuilding a
            # fresh kernel like the MultitaskGP branch above does -- so the
            # removal criterion scores against the exact same model that's
            # actually predicting. Its JacobianKernel carries no
            # `jac_finetune` (see build_shared_jacobian_kernels), so
            # optimize.py's per-removal MLP finetune hook (used only by
            # sequentially_optimize_samples, not batch_optimize_samples) is
            # automatically a no-op here -- the shared trunk is only ever
            # refit once, jointly across every dimension, after this loop
            # (see "full fine-tuning" call below), never per-dimension
            # mid-loop, which would desync one dimension's features from
            # another's.
            dim_model = model.gp_models[dim]
            dim_model.set_train_data(x_samples, y_dim, strict=False)
            dim_model.eval()

        else:
            # Use the existing model
            dim_model = model
            # Update training data
            if hasattr(dim_model, 'set_train_data'):
                dim_model.set_train_data(x_samples, y_dim, strict=False)
        
        # Run sequential optimization for this dimension
        if batch_optimize:
            x_filtered, y_filtered, deleted_samples, _ = batch_optimize_samples(
                model=dim_model,
                x_samples=x_samples.clone(),
                y_samples=y_dim.clone(),
                x_test=x_test,
            )
        else:
            x_filtered, y_filtered, deleted_samples, _ = sequentially_optimize_samples(
                model=dim_model,
                x_samples=x_samples.clone(),
                y_samples=y_dim.clone(),
                x_test=x_test,
                max_iter=max_iter,
                noise_type=noise_type,
                show_hessian=show_hessian,
                compare_optimal=compare_optimal,
            )
        # Find which indices were kept
        kept_mask = torch.ones(len(x_samples), dtype=torch.bool, device=x_samples.device)
        for deleted_x in deleted_samples:
            # Find matching samples in original
            matches = torch.all(torch.isclose(x_samples, deleted_x.unsqueeze(0), atol=1e-6), dim=1)
            kept_mask = kept_mask & ~matches
        
        filtered_indices_list.append(kept_mask)
        
        print(f"[Filter] Dimension {dim}: Kept {kept_mask.sum().item()}/{sampled_size} samples")
    
    # Combine: keep only samples that are kept in ALL (non-reward) dimensions
    # (OR we could use ANY to be more aggressive: remove if bad in any dim)
    # Let's use ANY - remove if detrimental in ANY dimension
    final_kept_mask = torch.ones(len(x_samples), dtype=torch.bool, device=x_samples.device)
    for kept_mask in filtered_indices_list:
        final_kept_mask = final_kept_mask & kept_mask

    print(f"[Filter] Final: Kept {final_kept_mask.sum().item()}/{sampled_size} samples "
          f"(intersection across state-delta dims, reward excluded)")

    # Shared-trunk jacobian model: the "full fine-tuning" variant -- one
    # single, joint trunk+head refit on the samples retained across ALL
    # dimensions, run exactly once here (never per-dimension inside the loop
    # above). See JacobianGPModel.finetune_shared_trunk for why deferring to
    # this combined mask (rather than refitting per-dimension mid-loop, the
    # way the single-output optimize._finetune_kernel_mlp hook works) is what
    # keeps every output dimension's features consistent with one another.
    if is_shared_jacobian and hasattr(model, 'finetune_shared_trunk'):
        kept_idx = torch.where(final_kept_mask)[0]
        if len(kept_idx) >= 2:
            y_finetune = y_samples[kept_idx]
            # y_samples is ALWAYS state-deltas only -- reward is deliberately
            # excluded from the removal/scoring criterion above (see module
            # docstring). But the shared trunk's own head predicts
            # [state-deltas, reward] when the model has learn_reward=True
            # (same convention as JacobianGPModel.train_model), so a joint
            # refit against state-deltas alone would target-shape-mismatch
            # against the trunk's actual output width -- reattach the
            # matching reward column here (from the same retained rows) so
            # every head the trunk has stays supervised, rather than going
            # stale under the trunk features this refit is about to shift.
            if getattr(model, 'learn_reward', False):
                kept_idx_np = kept_idx.cpu().numpy()
                reward_kept = torch.as_tensor(
                    all_data.rewards[kept_idx_np], dtype=y_finetune.dtype, device=y_finetune.device
                ).reshape(-1, 1)
                y_finetune = torch.cat([y_finetune, reward_kept], dim=-1)
            print(f"[Filter] Shared-trunk jacobian model: joint fine-tuning MLP on "
                  f"{len(kept_idx)} retained samples")
            model.finetune_shared_trunk(x_samples[kept_idx], y_finetune)

    # Restore original training data for MultitaskGP
    if is_multitask and original_train_inputs is not None:
        print("[Filter] Restoring MultitaskGP training data")
        # Set the training data back
        model.set_train_data(original_train_inputs, original_train_targets, strict=False)
    
    # Restore training/eval mode
    if model_was_training:
        model.train()
        if likelihood is not None:
            likelihood.train()
    else:
        model.eval()
        if likelihood is not None:
            likelihood.eval()
    
    filtered_size = final_kept_mask.sum().item()
    removed_size = sampled_size - filtered_size
    
    # Prepare removal_data if requested
    removal_data = None
    if store_data_removal:
        # build masks on original all_data (which may have been subsampled)
        kept_mask_np = final_kept_mask.cpu().numpy().astype(bool)
        removed_mask_np = ~kept_mask_np
        removal_data = {
            'kept': {
                'obs': all_data.obs[kept_mask_np],
                'act': all_data.act[kept_mask_np],
                'next_obs': all_data.next_obs[kept_mask_np],
                'rewards': all_data.rewards[kept_mask_np],
                'terminateds': all_data.terminateds[kept_mask_np],
                'truncateds': all_data.truncateds[kept_mask_np],
            },
            'removed': {
                'obs': all_data.obs[removed_mask_np],
                'act': all_data.act[removed_mask_np],
                'next_obs': all_data.next_obs[removed_mask_np],
                'rewards': all_data.rewards[removed_mask_np],
                'terminateds': all_data.terminateds[removed_mask_np],
                'truncateds': all_data.truncateds[removed_mask_np],
            }
        }
    
    # Create new filtered buffer with kept samples
    # Extract shapes from the existing buffer arrays
    obs_shape = replay_buffer.obs.shape[1:]
    action_shape = replay_buffer.action.shape[1:]
    
    filtered_buffer = ReplayBuffer(
        capacity=replay_buffer.capacity,
        obs_shape=obs_shape,
        action_shape=action_shape,
        obs_type=replay_buffer.obs.dtype,
        action_type=replay_buffer.action.dtype,
        reward_type=replay_buffer.reward.dtype,
        rng=replay_buffer._rng,
    )
    
    # Add filtered samples to new buffer
    kept_indices = torch.where(final_kept_mask)[0].cpu().numpy()
    for idx in kept_indices:
        filtered_buffer.add(
            obs=all_data.obs[idx],
            action=all_data.act[idx],
            next_obs=all_data.next_obs[idx],
            reward=all_data.rewards[idx],
            terminated=all_data.terminateds[idx],
            truncated=all_data.truncateds[idx],
        )
    
    # Collect statistics
    filter_stats = {
        'removal_ratio': removed_size / sampled_size if sampled_size > 0 else 0.0,
        'target_region_size': len(x_test),
        'was_subsampled': original_size > max_samples,
        'original_buffer_size': original_size,
        'sampled_size': sampled_size,
    }
    
    return filtered_buffer, filter_stats, removal_data