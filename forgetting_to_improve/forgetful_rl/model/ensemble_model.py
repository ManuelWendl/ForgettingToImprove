import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import copy
import itertools

from ..helper import types
from .util import to_tensor, truncated_normal_init, EnsembleLinearLayer, Normalizer


class EnsembleModel(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        termination_fn: types.TermFnType,
        hid_size: int = 200,
        num_layers: int = 4,
        deterministic: bool = False,
        target_is_delta: bool = True,
        normalize: bool = True,
        learn_logvar_bounds: bool = False,
        propagation_method: str = 'random_model',
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device(device)
        self.termination_fn = termination_fn
        self.deterministic = deterministic
        self.target_is_delta = target_is_delta
        self.num_elites = ensemble_size
        self.propagation_method = propagation_method

        def create_linear(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        layers = []
        curr_in = in_size
        for _ in range(num_layers):
            layers.append(create_linear(curr_in, hid_size))
            layers.append(nn.ReLU())
            curr_in = hid_size
        self.backbone = nn.Sequential(*layers)
        self.output_layer = create_linear(hid_size, out_size if deterministic else 2 * out_size)

        if not deterministic:
            self.min_logvar = nn.Parameter(-10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds)
            self.max_logvar = nn.Parameter(0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds)

        self.input_normalizer = Normalizer(in_size, self.device) if normalize else None
        self.apply(truncated_normal_init)
        self.to(self.device)
        self.elite_models: List[int] = list(range(ensemble_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.float()
        x = x.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        for layer in self.backbone:
            x = layer(x) if isinstance(layer, EnsembleLinearLayer) else layer(x)
        out = self.output_layer(x)
        if self.deterministic:
            return out, None
        mean, logvar = out.chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def reset(self, initial_obs_batch: np.ndarray, return_as_np: bool = True):
        model_state = {"obs": initial_obs_batch}
        if self.propagation_method == "fixed_model":
            indices = torch.randint(0, len(self.elite_models), (initial_obs_batch.shape[0],), device=self.device)
            model_state["propagation_indices"] = torch.tensor(self.elite_models, device=self.device)[indices]
        else:
            model_state["propagation_indices"] = None
        return model_state

    def step(
        self,
        action: np.ndarray,
        model_state: Dict[str, Any],
        sample: bool = True,
        model_updates: Optional[int] = None,
        rng: torch.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        self.eval()
        obs = model_state["obs"]
        with torch.no_grad():
            obs_t = to_tensor(obs).to(self.device).float()
            act_t = to_tensor(action).to(self.device).float()

            model_in = torch.cat([obs_t, act_t], dim=-1)
            if self.input_normalizer:
                model_in = self.input_normalizer.normalize(model_in)

            means, logvars = self.forward(model_in)

            batch_size = obs.shape[0]
            model_idx = torch.randint(0, len(self.elite_models), (batch_size,), device=self.device)
            chosen_model_indices = torch.tensor(self.elite_models, device=self.device)[model_idx]
            batch_range = torch.arange(batch_size, device=self.device)

            if self.propagation_method == "fixed_model" and model_state["propagation_indices"] is not None:
                chosen_model_indices = model_state["propagation_indices"]
            else:
                model_idx = torch.randint(0, len(self.elite_models), (batch_size,), device=self.device)
                chosen_model_indices = torch.tensor(self.elite_models, device=self.device)[model_idx]

            mean = means[chosen_model_indices, batch_range]
            if self.deterministic or not sample:
                preds = mean
            else:
                logvar = logvars[chosen_model_indices, batch_range]
                preds = torch.normal(mean, torch.exp(0.5 * logvar), generator=rng)

            preds_np = preds.cpu().numpy()
            next_obs = preds_np[:, :-1]
            if self.target_is_delta:
                next_obs += obs

            rewards = preds_np[:, -1:]
            dones = self.termination_fn(action, next_obs)

            next_model_state = {"obs": next_obs}
            return next_obs, rewards, dones, next_model_state

    def eval_score(self, batch):
        self.eval()
        with torch.no_grad():
            obs, action, next_obs, reward, _, _ = batch.astuple()
            t_obs = (next_obs - obs) if self.target_is_delta else next_obs
            target = torch.cat([to_tensor(t_obs), to_tensor(reward).unsqueeze(-1)], dim=-1).to(self.device).float()
            model_in = torch.cat([to_tensor(obs), to_tensor(action)], dim=-1).to(self.device).float()
            if self.input_normalizer:
                model_in = self.input_normalizer.normalize(model_in)

            mean, _ = self.forward(model_in)
            target_ext = target.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
            return F.mse_loss(mean, target_ext, reduction="none").mean(dim=(1, 2))

    def train_model(self,
                    replay_buffer,
                    optimizer,
                    batch_size,
                    num_epochs,
                    patience,
                    improvement_threshold,
                    val_ratio=0.2):

        all_data = replay_buffer.get_all()
        val_size = int(len(all_data) * val_ratio)
        train_data = all_data[:-val_size]
        val_data = all_data[-val_size:]

        best_val_score = self.eval_score(val_data)
        best_weights = copy.deepcopy(self.state_dict())
        epochs_since_update = 0
        training_losses = []

        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        for _ in epoch_iter:
            self.train()
            idxs = np.random.permutation(len(train_data))
            batch_losses = []

            for i in range(0, len(train_data), batch_size):
                batch_idxs = idxs[i : i + batch_size]
                batch = train_data[batch_idxs]

                obs_batch, act_batch = batch.obs, batch.act
                if self.input_normalizer:
                    self.input_normalizer.update_stats(np.concatenate([obs_batch, act_batch], axis=-1))

                optimizer.zero_grad()
                loss = self._calculate_loss(batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            total_avg_loss = np.mean(batch_losses)
            training_losses.append(total_avg_loss)

            current_val_score = self.eval_score(val_data)
            improvement = (best_val_score - current_val_score) / torch.abs(best_val_score)

            if improvement.max() > improvement_threshold:
                best_val_score = torch.minimum(best_val_score, current_val_score)
                best_weights = copy.deepcopy(self.state_dict())
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if epochs_since_update >= patience:
                break

        self.load_state_dict(best_weights)
        sorted_indices = np.argsort(best_val_score.cpu().numpy())
        self.elite_models = sorted_indices[:self.num_elites].tolist()

        return np.mean(training_losses)

    def _calculate_loss(self, batch):
        obs, action, next_obs, reward, _, _ = batch.astuple()
        t_obs = (next_obs - obs) if self.target_is_delta else next_obs
        target = torch.cat([to_tensor(t_obs), to_tensor(reward).unsqueeze(-1)], dim=-1).to(self.device).float()
        model_in = torch.cat([to_tensor(obs), to_tensor(action)], dim=-1).to(self.device).float()
        if self.input_normalizer:
            model_in = self.input_normalizer.normalize(model_in)

        mean, logvar = self.forward(model_in)
        target_ext = target.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        inv_var = torch.exp(-logvar)
        mse_loss = torch.pow(mean - target_ext, 2) * inv_var
        return (mse_loss + logvar).mean((1, 2)).sum()