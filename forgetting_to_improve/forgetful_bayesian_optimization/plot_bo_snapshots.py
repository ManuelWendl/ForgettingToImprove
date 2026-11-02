"""Illustration of forgetful BO on a 1D problem (snapshot figure in the paper).

Runs a short BO episode on a rough 1D Levy function and plots the GP before
and after each forgetting step: removed samples, target region A and the next
query before/after removal.

Usage (from the repository root):
    python -m forgetting_to_improve.forgetful_bayesian_optimization.plot_bo_snapshots

Output: figures/forgetful_bo_snapshots_middle_is_3.pdf and
figures/forgetful_bo_all_snapshots_row.pdf (pass --use-nn for the NN kernel).
"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Levy
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import standardize, unnormalize
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.priors.torch_priors import LogNormalPrior
from seaborn import axes_style
from tueplots import bundles, figsizes

from forgetting_to_improve.forgetful_bayesian_optimization.target_region import TargetRegion
from forgetting_to_improve.forgetting_to_improve.optimize import select_worst_sample
from forgetting_to_improve.forgetting_to_improve.helper.nn_kernel import build_jacobian_kernel, MLPMean

warnings.filterwarnings("ignore")

PALETTE = {
	"active": "#5F4690",
	"posterior": "#1D6996",
	"uncertainty": "#38A6A5",
	"target": "#73AF48",
	"removed": "#CC503E",
	"truth": "#0F8554",
}


def get_demo_problem():
	"""Return a rough 1D BoTorch test function for expressive visualization."""
	problem = Levy(dim=1, negate=True, dtype=torch.float64)
	bounds = problem.bounds.to(dtype=torch.float64)
	name = "Levy 1D (rough)"

	def objective(X: torch.Tensor) -> torch.Tensor:
		return problem(X)

	return name, objective, bounds


def create_bo_style_likelihood(alpha: float = 0.0) -> GaussianLikelihood:
	"""Match likelihood prior/constraints from BO setup_experiment.initialize_model_with_config."""
	if alpha > 0:
		return GaussianLikelihood(
			noise_prior=GammaPrior(2, 200),
			noise_constraint=Interval(1e-6, alpha * 10.0),
		)
	return GaussianLikelihood(
		noise_prior=GammaPrior(2, 200),
		noise_constraint=Interval(1e-6, 1e-2),
	)


def create_bo_style_covar_module(input_dim: int) -> MaternKernel:
	"""Match BO kernel defaults (Matern-2.5 with LogNormal lengthscale prior)."""
	return MaternKernel(
		nu=2.5,
		ard_num_dims=input_dim,
		lengthscale_prior=LogNormalPrior(0, 1),
		lengthscale_constraint=GreaterThan(1e-4),
	)


def fit_internal_gp(train_x: torch.Tensor, train_y_raw: torch.Tensor, use_nn: bool = False):
	"""Fit the GP used for forgetful BO decisions on standardized targets."""
	y_std = standardize(train_y_raw)
	likelihood = create_bo_style_likelihood(alpha=0.0)
	if use_nn:
		# Build last-layer Jacobian (linearised-Laplace) kernel and mean.
		nn_config = dict(hidden=64, depth=2, last_layer=True, prior_prec=1.0, epochs=200, lr=1e-2, activation="relu")
		kernel, mlp = build_jacobian_kernel(nn_config, n_dims=train_x.shape[-1], device="cpu", train_x=train_x.to(dtype=torch.float32), train_y=y_std.to(dtype=torch.float32))
		mean = MLPMean(mlp)
		model = SingleTaskGP(train_x, y_std, likelihood=likelihood, covar_module=kernel, mean_module=mean)
	else:
		covar_module = create_bo_style_covar_module(input_dim=train_x.shape[-1])
		model = SingleTaskGP(train_x, y_std, likelihood=likelihood, covar_module=covar_module)
	mll = ExactMarginalLogLikelihood(model.likelihood, model)
	fit_gpytorch_mll(mll)
	model.eval()
	model.likelihood.eval()
	return model, mll, y_std


def fit_visual_gp(train_x: torch.Tensor, train_y_raw: torch.Tensor, use_nn: bool = False):
	"""Fit a GP in raw output space for pretty plotting."""
	likelihood = create_bo_style_likelihood(alpha=0.0)
	if use_nn:
		nn_config = dict(hidden=64, depth=2, last_layer=True, prior_prec=1.0, epochs=200, lr=1e-2, activation="relu")
		kernel, mlp = build_jacobian_kernel(nn_config, n_dims=train_x.shape[-1], device="cpu", train_x=train_x.to(dtype=torch.float32), train_y=train_y_raw.to(dtype=torch.float32))
		mean = MLPMean(mlp)
		model = SingleTaskGP(
			train_x,
			train_y_raw,
			likelihood=likelihood,
			covar_module=kernel,
			mean_module=mean,
			outcome_transform=Standardize(m=1),
		)
	else:
		covar_module = create_bo_style_covar_module(input_dim=train_x.shape[-1])
		model = SingleTaskGP(
			train_x,
			train_y_raw,
			likelihood=likelihood,
			covar_module=covar_module,
			outcome_transform=Standardize(m=1),
		)
	mll = ExactMarginalLogLikelihood(model.likelihood, model)
	fit_gpytorch_mll(mll)
	model.eval()
	model.likelihood.eval()
	return model


def propose_next_point(
	model,
	bounds: torch.Tensor,
	existing_x: torch.Tensor,
	beta: float = 2.2,
	target_bounds=None,
):
	"""Simple 1D UCB maximization over a dense grid for the dummy BO loop."""
	grid = torch.linspace(bounds[0, 0], bounds[1, 0], 512, dtype=torch.float64).unsqueeze(-1)
	with torch.no_grad():
		posterior = model.posterior(grid)
		mean = posterior.mean.squeeze(-1)
		std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
		ucb_base = mean + beta * std

	ucb_region = ucb_base.clone()
	if target_bounds:
		region_mask = torch.zeros(grid.shape[0], dtype=torch.bool)
		for lower, upper in target_bounds:
			lower_val = float(lower[0])
			upper_val = float(upper[0])
			region_mask |= (grid[:, 0] >= lower_val) & (grid[:, 0] <= upper_val)
		if torch.any(region_mask):
			ucb_region[~region_mask] = -torch.inf

	min_dist = torch.cdist(grid, existing_x).min(dim=1).values
	ucb_filtered = ucb_region.clone()
	ucb_filtered[min_dist < 0.01 * (bounds[1, 0] - bounds[0, 0])] = -torch.inf

	if not torch.isfinite(ucb_filtered).any():
		ucb_filtered = ucb_region
	if not torch.isfinite(ucb_filtered).any():
		ucb_filtered = ucb_base

	best_idx = int(torch.argmax(ucb_filtered))
	return grid[best_idx].unsqueeze(0)


def evaluate_observation(objective, x_new: torch.Tensor, episode: int, noise_scale: float = 0.1):
	"""Deterministic dummy observations with a few injected outliers for forgetting."""
	base_value = objective(x_new).reshape(-1, 1)
	noise = noise_scale * torch.randn_like(base_value)
	return base_value + noise


def build_episode_history(objective, bounds: torch.Tensor, episode_to_visualize: int, seed: int, use_nn: bool = False):
	"""Run a small BO loop until the requested episode and return the dataset."""
	torch.manual_seed(seed)
	train_x = draw_sobol_samples(bounds=bounds, n=5, q=1, seed=seed).squeeze(1).to(dtype=torch.float64)
	print(f"Initial samples (episode 1): {train_x.squeeze(-1).tolist()}")
	train_x = torch.cat([train_x, torch.tensor([[0.037], [0.056], [0.081], [0.95], [0.6]], dtype=torch.float64)], dim=0)  # Add a central point for better visualization of removals.
	train_y = objective(train_x).reshape(-1, 1) 

	for episode in range(1, episode_to_visualize):
		model = fit_visual_gp(train_x, train_y, use_nn=use_nn)
		x_new = propose_next_point(model, bounds, train_x)
		y_new = evaluate_observation(objective, x_new, episode)
		train_x = torch.cat([train_x, x_new], dim=0)
		train_y = torch.cat([train_y, y_new], dim=0)

	return train_x, train_y


def get_plotting_state(
	train_x: torch.Tensor,
	train_y: torch.Tensor,
	removed_x: list,
	removed_y: list,
	target_bounds,
	remove_idx,
	bounds: torch.Tensor,
	next_x: torch.Tensor | None = None,
	next_y: torch.Tensor | None = None,
	pre_removal_next_x: torch.Tensor | None = None,
	pre_removal_next_y: torch.Tensor | None = None,
	is_next_episode: bool = False,
    use_nn: bool = False,
):
	grid = torch.linspace(bounds[0, 0], bounds[1, 0], 700, dtype=torch.float64).unsqueeze(-1)
	visual_model = fit_visual_gp(train_x, train_y, use_nn=use_nn)
	with torch.no_grad():
		posterior = visual_model.posterior(grid)
		mean = posterior.mean.squeeze(-1)
		std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)

	return {
		"train_x": train_x.clone(),
		"train_y": train_y.clone(),
		"removed_x": [x.clone() for x in removed_x],
		"removed_y": [y.clone() for y in removed_y],
		"remove_idx": remove_idx,
		"next_x": None if next_x is None else next_x.clone(),
		"next_y": None if next_y is None else next_y.clone(),
		"pre_removal_next_x": None if pre_removal_next_x is None else pre_removal_next_x.clone(),
		"pre_removal_next_y": None if pre_removal_next_y is None else pre_removal_next_y.clone(),
		"is_next_episode": is_next_episode,
		"target_bounds": [(lb.clone(), ub.clone()) for lb, ub in target_bounds],
		"grid": grid.squeeze(-1),
		"mean": mean,
		"std": std,
	}


def collect_forgetting_snapshots(
	train_x: torch.Tensor,
	train_y: torch.Tensor,
	objective,
	bounds: torch.Tensor,
	episode: int,
	seed: int,
	min_samples: int,
    use_nn: bool = False,
):
	"""Mirror the forgetful BO filtering loop and record every removal step."""
	target_region = TargetRegion(bounds, num_initial_points=250, seed=seed, iteration=episode)
	current_x = train_x.clone()
	current_y = train_y.clone()
	removed_x = []
	removed_y = []
	snapshots = []

	for _ in range(current_x.shape[0] - min_samples):
		decision_model, _, y_std = fit_internal_gp(current_x, current_y, use_nn=use_nn)
		if target_region.samples.shape[0] > 2:
			target_region.update(decision_model)

		remove_idx, _, _ = select_worst_sample(
			decision_model,
			current_x,
			y_std.squeeze(-1),
			target_region.samples,
			noise_type="joint",
		)

		if remove_idx is None:
			break
		
		target_bounds = target_region.get_bounds()
		pre_removal_next_x = None
		pre_removal_next_y = None
		if len(snapshots) == 0:
			pre_removal_model = fit_visual_gp(current_x, current_y, use_nn=use_nn)
			pre_removal_next_x = propose_next_point(
				pre_removal_model,
				bounds,
				current_x,
				target_bounds=target_bounds,
			)
			pre_removal_next_y = objective(pre_removal_next_x).reshape(-1, 1)

		snapshots.append(
			get_plotting_state(
				train_x=current_x,
				train_y=current_y,
				removed_x=removed_x,
				removed_y=removed_y,
				target_bounds=target_bounds,
				remove_idx=remove_idx,
				bounds=bounds,
				pre_removal_next_x=pre_removal_next_x,
				pre_removal_next_y=pre_removal_next_y,
				use_nn=use_nn,
			)
		)

		removed_x.append(current_x[remove_idx].clone())
		removed_y.append(current_y[remove_idx].clone())

		mask = torch.ones(current_x.shape[0], dtype=torch.bool)
		mask[remove_idx] = False
		current_x = current_x[mask]
		current_y = current_y[mask]

		if current_x.shape[0] <= min_samples:
			break

	# Add final panel: the next BO proposal after the last removal step.
	if current_x.shape[0] >= 2:
		# Recompute target region from the post-removal dataset for the final panel.
		final_target_region = TargetRegion(bounds, num_initial_points=200, seed=seed, iteration=episode)
		final_decision_model, _, _ = fit_internal_gp(current_x, current_y, use_nn=use_nn)
		if final_target_region.samples.shape[0] > 2:
			final_target_region.update(final_decision_model)

		visual_model = fit_visual_gp(current_x, current_y, use_nn=use_nn)
		target_bounds = final_target_region.get_bounds()
		next_x = propose_next_point(
			visual_model,
			bounds,
			current_x,
			target_bounds=target_bounds,
		)
		next_y = evaluate_observation(objective, next_x, episode + 1)
		snapshots.append(
			get_plotting_state(
				train_x=current_x,
				train_y=current_y,
				removed_x=removed_x,
				removed_y=removed_y,
				target_bounds=target_bounds,
				remove_idx=None,
				bounds=bounds,
				next_x=next_x,
				next_y=next_y,
				is_next_episode=True,
				use_nn=use_nn,
			)
		)

	return snapshots


def _apply_plot_gp_style() -> None:
	"""Apply the same visual theme used in helper/plot_gp.py."""
	theme = bundles.neurips2024()
	plt.rcParams.update(axes_style("white"))
	plt.rcParams.update(theme)
	plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
	plt.rcParams.update({"legend.frameon": False})


def _draw_snapshot_panel(ax, snapshot, panel_number: int, objective, bounds: torch.Tensor):
	"""Draw a single removal snapshot panel with plot_gp-like styling."""
	grid_x = torch.linspace(bounds[0, 0], bounds[1, 0], 500, dtype=torch.float64).unsqueeze(-1)
	with torch.no_grad():
		truth_y = objective(grid_x).squeeze(-1)

	for lower, upper in snapshot["target_bounds"]:
		ax.axvspan(
			float(lower[0]),
			float(upper[0]),
			color=PALETTE["target"],
			alpha=0.18,
			zorder=0,
		)

	ax.plot(grid_x.squeeze(-1), truth_y, color="#666666", linewidth=1.5, alpha=0.8, label=r"$f^*$")
	ax.plot(snapshot["grid"], snapshot["mean"], color=PALETTE["posterior"], linewidth=1.8, label=r"$\mu_{\mathrm{GP}}(x)$")
	ax.fill_between(
		snapshot["grid"].numpy(),
		(snapshot["mean"] - 2.2 * snapshot["std"]).numpy(),
		(snapshot["mean"] + 2.2 * snapshot["std"]).numpy(),
		color=PALETTE["uncertainty"],
		alpha=0.28,
		linewidth=0,
		label=r"$\pm 2.567\sigma_{\mathrm{GP}}(x)$",
	)

	active_x = snapshot["train_x"].squeeze(-1)
	active_y = snapshot["train_y"].squeeze(-1)
	ax.scatter(
		active_x,
		active_y,
		s=25,
		color=PALETTE["active"],
		edgecolors=PALETTE["active"],
		linewidth=0.1,
		zorder=4,
		label="Samples",
	)

	if snapshot["removed_x"]:
		removed_x = torch.stack(snapshot["removed_x"]).squeeze(-1)
		removed_y = torch.stack(snapshot["removed_y"]).squeeze(-1)
		ax.scatter(
			removed_x,
			removed_y,
			s=70,
			marker="x",
			linewidth=1.6,
			color=PALETTE["removed"],
			zorder=5,
		)

	remove_idx = snapshot["remove_idx"]
	if remove_idx is not None:
		ax.scatter(
			[snapshot["train_x"][remove_idx, 0]],
			[snapshot["train_y"][remove_idx, 0]],
			s=70,
			marker="x",
			linewidth=1.6,
			color=PALETTE["removed"],
			zorder=6,
		)

	if snapshot.get("pre_removal_next_x") is not None and snapshot.get("pre_removal_next_y") is not None:
		ax.scatter(
			[snapshot["pre_removal_next_x"][0, 0]],
			[snapshot["pre_removal_next_y"][0, 0]],
			s=50,
			marker="*",
			linewidth=0.0,
			edgecolor="white",
			color="#000000",
			zorder=7,
		)
	elif snapshot.get("is_next_episode", False) and snapshot.get("next_x") is not None:
		ax.scatter(
			[snapshot["next_x"][0, 0]],
			[snapshot["next_y"][0, 0]],
			s=50,
			marker="*",
			linewidth=0.0,
			edgecolor="white",
			color="#EDAD08",
			zorder=7,
		)

	ax.set_xlabel("$x$")
	ax.set_ylabel("$f(x)$")
	ax.set_ylim([-11,1])
	ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
	for spine in ax.spines.values():
		spine.set_linewidth(1.25)

	if snapshot.get("is_next_episode", False):
		ax.set_title(f"Snapshot {panel_number}: next episode", fontsize=9)
	else:
		ax.set_title(f"Snapshot {panel_number}: removal", fontsize=9)


def plot_snapshots(problem_name: str, objective, bounds: torch.Tensor, snapshots, save_dir: str | None):
	"""Create two figures:
	1) all snapshots side-by-side,
	2) 3-panel figure with snapshot 3 in the middle panel.
	"""
	if len(snapshots) < 4:
		raise RuntimeError("Need at least four snapshots for the requested 4-panel and combined 3-panel figures.")

	_apply_plot_gp_style()

	if save_dir is None:
		save_dir = os.path.join(os.path.dirname(__file__), "custom_figure_folder")
	os.makedirs(save_dir, exist_ok=True)

	legend_handles = [
		Line2D([0], [0], color="#666666", lw=1.5, alpha=0.8, label=r"$f^*$"),
		Line2D([0], [0], color=PALETTE["posterior"], lw=1.8, label=r"$\mu_{\mathrm{GP}}(x)$"),
		Line2D([0], [0], color=PALETTE["uncertainty"], lw=6, alpha=0.28, label=r"$\pm 2.567\sigma_{\mathrm{GP}}(x)$"),
		Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["active"], markeredgecolor=PALETTE["active"], markersize=5, label="Samples"),
		Line2D([0], [0], marker="x", color=PALETTE["removed"], lw=0, markersize=7, markeredgewidth=1.6, label="Removed"),
		Line2D([0], [0], marker="*", color="#000000", lw=0, markersize=9, markeredgecolor="white", label="Next before removal"),
		Line2D([0], [0], marker="*", color="#EDAD08", lw=0, markersize=9, markeredgecolor="white", label="Next sample"),
		Line2D([0], [0], color=PALETTE["target"], lw=8, alpha=0.18, label=r"$\mathcal{A}$"),
	]

	# Figure 1: all snapshots next to each other.
	all_count = len(snapshots)
	fig1_size = figsizes.iclr2023(nrows=1, ncols=max(2, all_count), height_to_width_ratio=1.0)["figure.figsize"]
	fig1, axes1 = plt.subplots(1, all_count, figsize=fig1_size, sharey=True)
	if all_count == 1:
		axes1 = [axes1]
	for idx, ax in enumerate(axes1):
		_draw_snapshot_panel(ax, snapshots[idx], idx + 1, objective, bounds)
	fig1.legend(legend_handles, [h.get_label() for h in legend_handles], loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
	fig1.tight_layout()
	file_path_1 = os.path.join(save_dir, "forgetful_bo_all_snapshots_row.pdf")
	fig1.savefig(file_path_1, dpi=300, bbox_inches="tight")
	plt.close(fig1)

	# Figure 2: 3 panels, with snapshot 3 in the middle.
	fig2_size = figsizes.iclr2023(nrows=1, ncols=2, height_to_width_ratio=0.9)["figure.figsize"]
	fig2, axes2 = plt.subplots(1, 3, figsize=fig2_size, sharey=True)
	_draw_snapshot_panel(axes2[0], snapshots[0], 1, objective, bounds)
	_draw_snapshot_panel(axes2[1], snapshots[2], 3, objective, bounds)
	_draw_snapshot_panel(axes2[2], snapshots[3], 4, objective, bounds)
	fig2.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)
	fig2.tight_layout(rect=[0, 0, 1, 0.9])
	file_path_2 = os.path.join(save_dir, "forgetful_bo_snapshots_middle_is_3.pdf")
	fig2.savefig(file_path_2, dpi=300, bbox_inches="tight")
	plt.close(fig2)

	print(f"Saved 2 snapshot figure(s) to: {save_dir}")


def main():
	parser = argparse.ArgumentParser(description="Dummy visualization of forgetful BO sample removals in a 1D problem.")
	parser.add_argument("--episode", type=int, default=2, help="Episode to visualize.")
	parser.add_argument("--seed", type=int, default=13, help="Random seed for the dummy BO run.")
	parser.add_argument("--min-samples", type=int, default=8, help="Minimum number of samples to keep after forgetting.")
	parser.add_argument(
		"--save-dir",
		type=str,
		default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"),
		help="Directory to save pairwise snapshot figures.",
	)
	parser.add_argument("--use-nn", action="store_true", help="Use last-layer Jacobian (Laplace) NN kernel for GP models.")
	args = parser.parse_args()

	problem_name, objective_orig, bounds_orig = get_demo_problem()

	# Normalize inputs to [0, 1] exactly as the main BO opti_loop does.
	bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

	def objective(x_norm: torch.Tensor) -> torch.Tensor:
		return objective_orig(unnormalize(x_norm, bounds_orig))

	train_x, train_y = build_episode_history(objective, bounds, args.episode, args.seed, use_nn=args.use_nn)
	snapshots = collect_forgetting_snapshots(
		train_x=train_x,
		train_y=train_y,
		objective=objective,
		bounds=bounds,
		episode=args.episode,
		seed=args.seed,
		min_samples=args.min_samples,
		use_nn=args.use_nn,
	)

	if not snapshots:
		raise RuntimeError("No forgetting steps were generated for the dummy visualization.")

	plot_snapshots(problem_name, objective, bounds, snapshots, args.save_dir)


if __name__ == "__main__":
	main()
