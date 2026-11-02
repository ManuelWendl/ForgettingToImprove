"""Utilities for running AutoML hyperparameter benchmarks from CSV files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple


class CsvObjective:
    """Objective wrapper that looks up accuracy values from a CSV table.

    The dataset is expected to contain a column ``data_id`` identifying the
    task, one column ``accuracy`` giving the performance, and a set of
    numeric columns representing hyperparameters.  When the object is called
    with a point ``x`` it finds the nearest row (Euclidean distance) in the
    hyperparameter space and returns the corresponding accuracy.  The
    attribute ``_optimal_value`` is set to ``-max_accuracy`` so that the
    existing code in ``main.py`` (which negates this quantity) computes the
    correct optimal reward.
    """

    def __init__(self, df: pd.DataFrame, param_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.param_cols = param_cols
        # set optimal value to negative of maximum accuracy so that
        # ``-obj._optimal_value`` equals max accuracy
        self._optimal_value = -float(self.df['accuracy'].max())
        
        # Store bounds for normalization: prevents scale mismatch in distance computation
        data = self.df[self.param_cols].values
        self.param_mins = data.min(axis=0)
        self.param_maxs = data.max(axis=0)
        self.param_ranges = self.param_maxs - self.param_mins
        # Avoid division by zero for constant parameters
        self.param_ranges = np.where(self.param_ranges < 1e-10, 1.0, self.param_ranges)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x may be a batch of points; we evaluate each row separately
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        data = self.df[self.param_cols].values
        # Normalize both query points and data to [0, 1] for fair distance computation
        data_normalized = (data - self.param_mins) / self.param_ranges
        
        outs = []
        for xi in x_np:
            # find nearest neighbour in normalized space
            xi_normalized = (xi - self.param_mins) / self.param_ranges
            d2 = ((data_normalized - xi_normalized) ** 2).sum(axis=1)
            idx = int(d2.argmin())
            outs.append(self.df.iloc[idx]['accuracy'])
        return torch.tensor(outs, dtype=torch.float64)


class SwitchingCsvObjective:
    """Objective wrapper for a sequence of tasks (data_ids).

    This class keeps the full DataFrame and a schedule listing which
    ``data_id`` should be used at each BO iteration.  The underlying
    evaluation behaves exactly like ``CsvObjective`` but is restricted to
    the subset corresponding to the ``current_task``.  The method
    ``set_iteration`` updates the current task and recomputes the
    ``_optimal_value`` so that statistics can be computed correctly when the
    target changes over time.

    Note
    ----
    In the continuous CSV benchmark the optimisation loop detects changes
    of ``current_task`` and clears its training data before beginning a new
    task (always re‑initialising with two points).  Kernel/likelihood
    hyperparameters learned on the previous task are carried over, but no
    past samples are retained.  This keeps the BO process focused on the
    active dataset while still leveraging previously learned lengthscales
    and noise levels.
    """

    def __init__(self, full_df: pd.DataFrame, param_cols: List[str], schedule: List):
        self.full_df = full_df.reset_index(drop=True)
        self.param_cols = param_cols
        self.schedule = schedule
        self.current_task = None
        self.current_df = None
        self._optimal_value = None
        self._iteration = None
        
        # Compute global bounds for normalization across all tasks
        # This ensures consistent distance metrics when switching between tasks
        data = self.full_df[self.param_cols].values
        self.param_mins = data.min(axis=0)
        self.param_maxs = data.max(axis=0)
        self.param_ranges = self.param_maxs - self.param_mins
        # Avoid division by zero for constant parameters
        self.param_ranges = np.where(self.param_ranges < 1e-10, 1.0, self.param_ranges)

    def set_iteration(self, iteration: int):
        """Set the active iteration and update the current task accordingly."""
        if iteration < 0 or iteration >= len(self.schedule):
            raise IndexError(f"Iteration {iteration} out of range for schedule length {len(self.schedule)}")
        task = self.schedule[iteration]
        if task != self.current_task:
            # update only when task changes
            self.current_task = task
            self.current_df = self.full_df[self.full_df['data_id'] == task].reset_index(drop=True)
            self._optimal_value = -float(self.current_df['accuracy'].max())
        self._iteration = iteration

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # delegate to a temporary CsvObjective built on ``current_df``
        if self.current_df is None:
            raise RuntimeError("Current task not set; call set_iteration first")
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        data = self.current_df[self.param_cols].values
        # Normalize both query points and data to [0, 1] for fair distance computation
        data_normalized = (data - self.param_mins) / self.param_ranges
        
        outs = []
        for xi in x_np:
            xi_normalized = (xi - self.param_mins) / self.param_ranges
            d2 = ((data_normalized - xi_normalized) ** 2).sum(axis=1)
            idx = int(d2.argmin())
            outs.append(self.current_df.iloc[idx]['accuracy'])
        return torch.tensor(outs, dtype=torch.float64)

    # expose optimal value property to mimic CsvObjective
    @property
    def optimal_value(self):
        return self._optimal_value


def load_csv_benchmark(csv_path: str) -> pd.DataFrame:
    """Load the AutoML benchmark CSV file into a DataFrame.

    This helper simply wraps ``pandas.read_csv`` so that callers can be
    mocked easily in tests.  We enable ``skipinitialspace`` because the
    original OpenML export has stray spaces after commas which otherwise
    confuse the parser.
    """
    return pd.read_csv(csv_path, skipinitialspace=True)


def get_param_columns(df: pd.DataFrame) -> List[str]:
    """Return the names of columns that should be treated as hyperparameters.

    We drop the known metadata columns (``data_id``, ``accuracy`` and a small
    set of performance/quality measures) and treat all remaining numeric
    columns as parameters.
    """
    exclude = {
        'data_id', 'accuracy', 'auc', 'brier', 'runtime', 'scimark',
        'MajorityClassPercentage', 'MajorityClassSize', 'NumberOfClasses',
        'NumberOfInstances', 'NumberOfFeatures',
        'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures'
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def compute_bounds(df: pd.DataFrame, param_cols: List[str]) -> torch.Tensor:
    """Compute search bounds from the data for the given parameter columns.

    Returns a tensor of shape ``(2, d)`` giving min/max for each dimension.
    """
    mins = df[param_cols].min().values.astype(float)
    maxs = df[param_cols].max().values.astype(float)
    return torch.tensor([mins, maxs], dtype=torch.float64)