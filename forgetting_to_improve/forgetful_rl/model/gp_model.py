import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import gc
import gpytorch
from torch import nn as nn

from ..helper import types
from .util import to_tensor, Normalizer


class MultitaskGPModel(gpytorch.models.ExactGP):
    """Multi-task GP model for dynamics prediction."""
    
    def __init__(self, train_x, train_y, likelihood, num_tasks, kernel_name: str = "matern"):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        # Choose kernel based on configuration
        kernel_name = kernel_name.lower()
        if kernel_name == "matern":
            base_kernel = gpytorch.kernels.MaternKernel()
        elif kernel_name == "rbf" or kernel_name == "gaussian":
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel_name == "linear":
            base_kernel = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unknown kernel_name: {kernel_name}")
                
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=num_tasks, rank=1
        )
        self.num_tasks = num_tasks

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPModel(nn.Module):
    """Gaussian Process dynamics model compatible with ensemble model interface."""
    
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        termination_fn: types.TermFnType,
        deterministic: bool = False,
        target_is_delta: bool = True,
        normalize: bool = True,
        max_train_size: int = 1000,
        kernel_type: str = "matern",
        noise_prior_shape: float = 2.0,
        noise_prior_rate: float = 0.1,
        learn_reward: bool = True,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device(device)
        self.termination_fn = termination_fn
        self.deterministic = deterministic
        self.target_is_delta = target_is_delta
        self.max_train_size = max_train_size
        # When False, the GP only predicts state deltas (out_size == obs_size, no
        # trailing reward column) -- reward is instead computed analytically from the
        # predicted next state by the caller's reward_fn (see algorithm/mbpo.py). This
        # keeps reward -- which can be (near-)constant or otherwise poorly explained by
        # a linear kernel, see the forgetful_rl filtering diagnostics -- out of both the
        # model's own output space and the forgetting filter entirely, rather than
        # trying to special-case it there.
        self.learn_reward = learn_reward
        self.kernel_type = kernel_type
        self.noise_prior_shape = noise_prior_shape
        self.noise_prior_rate = noise_prior_rate
        
        # GP components (initialized during training)
        self.gp_model = None
        self.likelihood = None
        
        # Normalizer
        self.input_normalizer = Normalizer(in_size, self.device) if normalize else None
        self.output_normalizer = Normalizer(out_size, self.device) if normalize else None
        
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through GP model."""
        if self.gp_model is None:
            raise RuntimeError("GP model not trained yet. Call train_model first.")
        
        x = x.float()
        self.gp_model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.gp_model(x))
            mean = predictions.mean
            
            if self.deterministic:
                return mean, None
            else:
                variance = predictions.variance
                return mean, torch.log(variance)

    def reset(self, initial_obs_batch: np.ndarray, return_as_np: bool = True):
        """Reset model state for rollout."""
        model_state = {"obs": initial_obs_batch}
        return model_state

    def step(
        self,
        action: np.ndarray,
        model_state: Dict[str, Any],
        sample: bool = True,
        model_updates: Optional[int] = None,
        rng: torch.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step dynamics model forward."""
        if self.gp_model is None:
            raise RuntimeError("GP model not trained yet. Call train_model first.")
        
        self.gp_model.eval()
        self.likelihood.eval()
        
        obs = model_state["obs"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            obs_t = to_tensor(obs).to(self.device).float()
            act_t = to_tensor(action).to(self.device).float()

            model_in = torch.cat([obs_t, act_t], dim=-1)
            if self.input_normalizer:
                model_in = self.input_normalizer.normalize(model_in)

            predictions = self.likelihood(self.gp_model(model_in))
            mean = predictions.mean
            
            if self.deterministic or not sample:
                preds = mean
            else:
                # Sample from GP posterior
                preds = predictions.rsample(torch.Size([1])).squeeze(0)

            # Denormalize if needed
            if self.output_normalizer:
                preds = self.output_normalizer.denormalize(preds)
            
            preds_np = preds.cpu().numpy()
            if self.learn_reward:
                next_obs = preds_np[:, :-1]
                rewards = preds_np[:, -1:]
            else:
                # All output columns are state deltas; caller (mbpo.py) computes
                # reward analytically from (action, next_obs) via reward_fn.
                next_obs = preds_np
                rewards = None
            if self.target_is_delta:
                next_obs = next_obs + obs

            dones = self.termination_fn(action, next_obs)

            next_model_state = {"obs": next_obs}
            return next_obs, rewards, dones, next_model_state

    def eval_score(self, batch):
        """Evaluate model on validation batch."""
        if self.gp_model is None:
            return torch.tensor(float('inf'))
        
        self.gp_model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            obs, action, next_obs, reward, _, _ = batch.astuple()
            t_obs = (next_obs - obs) if self.target_is_delta else next_obs
            if self.learn_reward:
                target = torch.cat([to_tensor(t_obs), to_tensor(reward).unsqueeze(-1)], dim=-1).to(self.device).float()
            else:
                target = to_tensor(t_obs).to(self.device).float()
            model_in = torch.cat([to_tensor(obs), to_tensor(action)], dim=-1).to(self.device).float()
            
            if self.input_normalizer:
                model_in = self.input_normalizer.normalize(model_in)
            
            if self.output_normalizer:
                target = self.output_normalizer.normalize(target)

            predictions = self.likelihood(self.gp_model(model_in))
            mean = predictions.mean
            
            mse = torch.mean((mean - target) ** 2)
            return mse

    def train_model(
        self,
        replay_buffer,
        optimizer,
        batch_size,
        num_epochs,
        patience,
        improvement_threshold,
        val_ratio=0.2,
        model_lr=0.01,
        model_wd=0.0,
    ):
        """Train GP model on replay buffer data."""
        
        # Get all data
        all_data = replay_buffer.get_all()
        
        # Subsample if dataset is too large
        if len(all_data) > self.max_train_size:
            indices = np.random.choice(len(all_data), self.max_train_size, replace=False)
            all_data = all_data[indices]
        
        val_size = int(len(all_data) * val_ratio)
        train_data = all_data[:-val_size]
        val_data = all_data[-val_size:]

        # Prepare training data
        obs_batch, act_batch = train_data.obs, train_data.act
        next_obs_batch, reward_batch = train_data.next_obs, train_data.rewards
        
        # Compute targets
        t_obs = (next_obs_batch - obs_batch) if self.target_is_delta else next_obs_batch
        if self.learn_reward:
            targets = np.concatenate([t_obs, reward_batch.reshape(-1, 1)], axis=-1)
        else:
            targets = t_obs
        
        # Convert to tensors
        train_x = torch.cat([to_tensor(obs_batch), to_tensor(act_batch)], dim=-1).to(self.device).float()
        train_y = to_tensor(targets).to(self.device).float()
        
        # Update normalizers
        if self.input_normalizer:
            self.input_normalizer.update_stats(train_x.cpu().numpy())
            train_x = self.input_normalizer.normalize(train_x)
        
        if self.output_normalizer:
            self.output_normalizer.update_stats(train_y.cpu().numpy())
            train_y = self.output_normalizer.normalize(train_y)
        
        # Initialize GP model
        # --- safe-recreation: free any previous GP/likelihood to avoid GPU memory buildup ---
        try:
            if getattr(self, "gp_model", None) is not None:
                del self.gp_model
            if getattr(self, "likelihood", None) is not None:
                del self.likelihood
            gc.collect()
            if "cuda" in str(self.device).lower():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.out_size)
        # Set Gamma prior on noise using configured shape/rate
        gamma_prior = gpytorch.priors.GammaPrior(self.noise_prior_shape, self.noise_prior_rate)
        # Register prior for the likelihood noise
        try:
            # Preferred registration: attribute name 'noise'
            self.likelihood.register_prior("noise_prior", gamma_prior, lambda m: m.noise)
        except Exception:
            # Fallback: try setting prior on the underlying noise covar if available
            try:
                self.likelihood.noise_covar.register_prior("noise_prior", gamma_prior, "noise")
            except Exception:
                pass
        self.gp_model = MultitaskGPModel(train_x, train_y, self.likelihood, num_tasks=self.out_size, kernel_name=self.kernel_type)
        
        self.gp_model.to(self.device)
        self.likelihood.to(self.device)
        
        # Set to training mode
        self.gp_model.train()
        self.likelihood.train()
        
        # GP models manage their own optimizer (mbpo.py's externally-built `model_optimizer`
        # is None here, since dynamics_model.parameters() is empty before self.gp_model exists
        # -- see mbpo.py's "Create optimizer only if model has parameters" comment). The GP's
        # own params/likelihood are also recreated fresh above every call, so the optimizer
        # must be rebuilt here regardless -- but it now honors model_lr (from
        # cfg.overrides.model_lr, threaded through explicitly) instead of a hardcoded 0.01.
        optimizer = torch.optim.Adam(
            list(self.gp_model.parameters()) + list(self.likelihood.parameters()),
            lr=model_lr,
            weight_decay=model_wd,
        )

        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        
        # Training loop
        best_val_score = float('inf')
        epochs_since_update = 0
        training_losses = []
        
        max_iter = num_epochs if num_epochs else 100
        
        for epoch in range(max_iter):
            optimizer.zero_grad()
            output = self.gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            # Validation
            if (epoch + 1) % 10 == 0:
                current_val_score = self.eval_score(val_data)
                improvement = (best_val_score - current_val_score) / abs(best_val_score + 1e-8)
                
                if improvement > improvement_threshold:
                    best_val_score = min(best_val_score, current_val_score.item())
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1
                
                if epochs_since_update >= patience // 10:
                    break
        
        return np.mean(training_losses)