"""Shared-trunk jacobian (last-layer Laplace) dynamics model for forgetful_rl.

Alternative to ``model/gp_model.py`` (GPModel with matern/rbf/linear
kernels); see the class docstring below for how it plugs into
``algorithm/mbpo.py`` and ``forgetting/filter_samples.py``.

One MLP (a shared trunk with an ``out_size``-wide head) predicts every output
dimension of the dynamics model; each output dimension gets its own gpytorch
``ExactGP`` (JacobianKernel + MLPMeanIndexed) for GP-conditioned predictive
mean/variance, all sharing that one trunk's frozen features -- see
``forgetting_to_improve.forgetting_to_improve.helper.nn_kernel.shared_trunk``
for why this is mathematically exact (block-diagonal GGN across outputs) and
cheaper than one independent MLP per dimension.
"""
import gc
import sys
from typing import Any, Dict, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from torch import nn as nn

sys.path.append('../../../')
from forgetting_to_improve.forgetting_to_improve.helper.nn_kernel import (
    build_shared_jacobian_kernels,
    finetune_shared_mlp,
    reset_parameters,
)

from ..helper import types
from .util import to_tensor, Normalizer


class _SharedTrunkExactGP(gpytorch.models.ExactGP):
    """Single-output ExactGP wrapping one (mean, kernel) pair returned by
    ``build_shared_jacobian_kernels`` -- one instance per output dimension,
    all sharing the same underlying MLP trunk (see ``mean_module.model`` /
    ``covar_module.base_kernel.model``, the same object across every
    dimension's GP)."""

    def __init__(self, train_x, train_y, likelihood, mean_module, jac_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(jac_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class JacobianGPModel(nn.Module):
    """Dynamics model backed by a shared-trunk multi-output last-layer
    Laplace (jacobian) approximation.

    Same interface as ``GPModel`` (``forward``/``reset``/``step``/
    ``eval_score``/``train_model``); select it with
    ``dynamics_model=jacobian_gp_model``.

    mbpo.py passes ``learn_reward`` to models whose class name ends in
    "GPModel", and runs forgetting when the model has ``.gp_model`` and
    ``.likelihood`` (here: ``self`` and the per-dimension likelihoods, set by
    train_model). ``forgetting/filter_samples.py`` recognises this model by
    ``.gp_models``/``.shared_mlp``.
    """

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
        hidden: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        last_layer: bool = True,
        prior_prec: float = 1.0,
        epochs: int = 500,
        lr: float = 1e-2,
        retrain_mode: str = "finetune",
        finetune_epochs: int = 200,
        finetune_lr: float = 1e-2,
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
        # learn_reward=False: the model predicts the state only (see GPModel)
        self.learn_reward = learn_reward

        # Shared-trunk jacobian kernel config (same keys as the `kernel: [{type: jacobian, ...}]` block)
        self.jacobian_config = dict(
            hidden=hidden, depth=depth, activation=activation, last_layer=last_layer,
            prior_prec=prior_prec, epochs=epochs, lr=lr,
        )
        mode = str(retrain_mode).lower()
        if mode not in ("finetune", "scratch"):
            raise ValueError(
                f"JacobianGPModel: unknown retrain_mode '{mode}'. Options: 'finetune', 'scratch'"
            )
        # Joint trunk refit after forgetting (finetune_shared_trunk); `epochs` above
        # is used for the initial fit in train_model
        self.jacobian_finetune = dict(mode=mode, epochs=finetune_epochs, lr=finetune_lr)

        # Set by train_model()
        self.shared_mlp = None
        self.gp_models: Optional[nn.ModuleList] = None
        self.likelihoods: Optional[nn.ModuleList] = None
        self.gp_model = None
        self.likelihood = None

        self.input_normalizer = Normalizer(in_size, self.device) if normalize else None
        self.output_normalizer = Normalizer(out_size, self.device) if normalize else None

        self.to(self.device)

    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack independent per-dimension GP predictions into (batch, out_size)."""
        means, variances = [], []
        for gp, lik in zip(self.gp_models, self.likelihoods):
            gp.eval()
            lik.eval()
            pred = lik(gp(x))
            means.append(pred.mean)
            variances.append(pred.variance)
        return torch.stack(means, dim=-1), torch.stack(variances, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass, mirroring GPModel.forward's (mean, log_variance) contract."""
        if self.gp_models is None:
            raise RuntimeError("JacobianGPModel not trained yet. Call train_model first.")
        x = x.float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, variance = self._predict(x)
            if self.deterministic:
                return mean, None
            return mean, torch.log(variance)

    def reset(self, initial_obs_batch: np.ndarray, return_as_np: bool = True):
        """Reset model state for rollout."""
        return {"obs": initial_obs_batch}

    def step(
        self,
        action: np.ndarray,
        model_state: Dict[str, Any],
        sample: bool = True,
        model_updates: Optional[int] = None,
        rng: torch.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step dynamics model forward, mirroring GPModel.step exactly."""
        if self.gp_models is None:
            raise RuntimeError("JacobianGPModel not trained yet. Call train_model first.")

        obs = model_state["obs"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            obs_t = to_tensor(obs).to(self.device).float()
            act_t = to_tensor(action).to(self.device).float()

            model_in = torch.cat([obs_t, act_t], dim=-1)
            if self.input_normalizer:
                model_in = self.input_normalizer.normalize(model_in)

            mean, variance = self._predict(model_in)

            if self.deterministic or not sample:
                preds = mean
            else:
                std = variance.clamp_min(1e-12).sqrt()
                if rng is not None:
                    noise = torch.randn(mean.shape, generator=rng, device=mean.device, dtype=mean.dtype)
                else:
                    noise = torch.randn_like(mean)
                preds = mean + std * noise

            if self.output_normalizer:
                preds = self.output_normalizer.denormalize(preds)

            preds_np = preds.cpu().numpy()
            if self.learn_reward:
                next_obs = preds_np[:, :-1]
                rewards = preds_np[:, -1:]
            else:
                # All output columns are state deltas; caller (mbpo.py) computes
                # reward analytically -- same convention as GPModel.
                next_obs = preds_np
                rewards = None
            if self.target_is_delta:
                next_obs = next_obs + obs

            dones = self.termination_fn(action, next_obs)
            next_model_state = {"obs": next_obs}
            return next_obs, rewards, dones, next_model_state

    def eval_score(self, batch):
        """Evaluate model on a validation batch (mean-squared error)."""
        if self.gp_models is None:
            return torch.tensor(float('inf'))

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

            mean, _ = self._predict(model_in)
            return torch.mean((mean - target) ** 2)

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
        """Train the shared trunk (once, jointly across all output
        dimensions) and build one GP per output dimension on its frozen
        features (last-layer Laplace: fixed features, Bayesian last layer).
        """
        all_data = replay_buffer.get_all()

        if len(all_data) > self.max_train_size:
            indices = np.random.choice(len(all_data), self.max_train_size, replace=False)
            all_data = all_data[indices]

        val_size = max(1, int(len(all_data) * val_ratio))
        train_data = all_data[:-val_size] if len(all_data) > val_size else all_data
        val_data = all_data[-val_size:]

        obs_batch, act_batch = train_data.obs, train_data.act
        next_obs_batch, reward_batch = train_data.next_obs, train_data.rewards

        t_obs = (next_obs_batch - obs_batch) if self.target_is_delta else next_obs_batch
        if self.learn_reward:
            targets = np.concatenate([t_obs, reward_batch.reshape(-1, 1)], axis=-1)
        else:
            targets = t_obs

        train_x = torch.cat([to_tensor(obs_batch), to_tensor(act_batch)], dim=-1).to(self.device).float()
        train_y = to_tensor(targets).to(self.device).float()

        # Validation data, built the same way as train_x/train_y
        val_t = (val_data.next_obs - val_data.obs) if self.target_is_delta else val_data.next_obs
        val_targets = np.concatenate([val_t, val_data.rewards.reshape(-1, 1)], axis=-1) if self.learn_reward else val_t
        val_x = torch.cat([to_tensor(val_data.obs), to_tensor(val_data.act)], dim=-1).to(self.device).float()
        val_y = to_tensor(val_targets).to(self.device).float()

        if self.input_normalizer:
            self.input_normalizer.update_stats(train_x.cpu().numpy())
            train_x = self.input_normalizer.normalize(train_x)
            val_x = self.input_normalizer.normalize(val_x)
        if self.output_normalizer:
            self.output_normalizer.update_stats(train_y.cpu().numpy())
            train_y = self.output_normalizer.normalize(train_y)
            val_y = self.output_normalizer.normalize(val_y)

        # --- safe-recreation: free the previous trunk/GPs before rebuilding ---
        try:
            if getattr(self, "gp_models", None) is not None:
                del self.gp_models, self.likelihoods, self.shared_mlp
            gc.collect()
            if "cuda" in str(self.device).lower():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # ONE shared-trunk MLP fit jointly against every output column,
        # instead of `out_size` independent MLP fits.
        self.shared_mlp, kernels, means = build_shared_jacobian_kernels(
            self.jacobian_config, n_dims=self.in_size, out_dim=self.out_size,
            device=self.device, train_x=train_x, train_y=train_y,
        )

        self.gp_models = nn.ModuleList()
        self.likelihoods = nn.ModuleList()
        training_losses = []

        # Per-dimension GPs use default hyperparameters (no marginal-likelihood fit:
        # an exact-GP fit per output dimension is too slow at replay-buffer sizes).
        # The uncertainty is set by the trunk fit and `prior_prec` (Sigma = I / prior_prec).
        for dim in range(self.out_size):
            lik = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            gp = _SharedTrunkExactGP(train_x, train_y[:, dim], lik, means[dim], kernels[dim]).to(self.device)
            gp.eval()
            lik.eval()

            with torch.no_grad():
                val_mean = lik(gp(val_x)).mean
                dim_val_mse = torch.mean((val_mean - val_y[:, dim]) ** 2).item()
            training_losses.append(dim_val_mse)

            self.gp_models.append(gp)
            self.likelihoods.append(lik)

        # Attributes used by mbpo.py and filter_samples.py to detect a trained GP
        # model. Set via object.__setattr__ so that `self` is not registered as its
        # own submodule (which would recurse in .train()/.eval()/.to()).
        object.__setattr__(self, 'gp_model', self)
        object.__setattr__(self, 'likelihood', self.likelihoods)

        return float(np.mean(training_losses))

    def finetune_shared_trunk(self, x_kept: torch.Tensor, y_kept: torch.Tensor):
        """Refit the whole shared trunk (all output heads) on the samples kept
        after forgetting. Called once with the kept set combined over all
        output dimensions (see forgetting/filter_samples.py), so the features of
        all dimensions stay consistent.
        """
        if self.shared_mlp is None or x_kept.shape[0] < 2:
            return
        ft = self.jacobian_finetune
        if not ft or ft.get('epochs', 0) <= 0:
            return

        if ft.get('mode', 'finetune') == 'scratch':
            reset_parameters(self.shared_mlp)
        finetune_shared_mlp(self.shared_mlp, x_kept, y_kept, epochs=ft['epochs'], lr=ft['lr'])

        # Kernels and means read the trunk directly; refresh the parameter
        # snapshot used when last_layer=False
        for gp in self.gp_models:
            base = getattr(gp.covar_module, 'base_kernel', gp.covar_module)
            if hasattr(base, 'params'):
                base.params = {k: v.detach() for k, v in self.shared_mlp.named_parameters()}
