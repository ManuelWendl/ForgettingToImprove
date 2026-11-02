"""Load MBPO runs for plotting from local Hydra run folders.

Every run is a dict with keys
    config     -- the run's (resolved) Hydra config as a plain dict
    metrics    -- DataFrame of all logged rows (one row per log call)
    seed       -- the run's seed
    forgetful  -- whether forgetful RL (replay-buffer filtering) was enabled
    removal    -- list of DataFrames, one per filtering event, with columns
                  obs, action, next_obs, reward, removed (only for runs with
                  overrides.store_data_removal=true; otherwise empty)

Local runs are the folders Hydra creates under `root_dir` (default ./exp):
each contains .hydra/config.yaml, results.jsonl and, with
store_data_removal=true, data_removal/*.npz.
"""
import json
import pathlib
from typing import Callable, Dict, Iterator, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def get_cfg(cfg: Dict, dotted_key: str, default=None):
    """cfg['a']['b'] for dotted_key 'a.b' (default if any level is missing)."""
    node = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_local_runs(root: str, where: Optional[Callable[[Dict], bool]] = None) -> Iterator[Dict]:
    """Yield every run below `root` (searched recursively) whose config passes `where`."""
    for jsonl in sorted(pathlib.Path(root).rglob("results.jsonl")):
        run_dir = jsonl.parent
        cfg_path = run_dir / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            print(f"Skipping {run_dir}: no .hydra/config.yaml")
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        if where is not None and not where(cfg):
            continue
        with open(jsonl) as f:
            metrics = pd.DataFrame([json.loads(line) for line in f if line.strip()])
        removal = []
        for npz in sorted((run_dir / "data_removal").glob("*.npz")):
            z = np.load(npz)
            removal.append(pd.DataFrame({
                "obs": list(z["obs"]),
                "action": list(z["act"]),
                "next_obs": list(z["next_obs"]),
                "reward": np.asarray(z["reward"]).ravel(),
                "removed": z["removed"],
            }))
        yield {
            "config": cfg,
            "metrics": metrics,
            "seed": cfg.get("seed"),
            "forgetful": bool(get_cfg(cfg, "overrides.forgetful_rl", False)),
            "removal": removal,
            "path": str(run_dir),
        }


def evaluation_frame(run: Dict) -> pd.DataFrame:
    """Evaluation and forgetful_rl metrics of one run, indexed by env_step,
    with `seed` and `forgetful` columns (rows with no such metric dropped)."""
    data = run["metrics"].copy()
    if "env_step" in data.columns:
        data = data.set_index("env_step")
    elif "_step" in data.columns:
        data = data.rename(columns={"_step": "env_step"}).set_index("env_step")
    cols = [c for c in data.columns if str(c).startswith("eval") or str(c).startswith("forgetful_rl/")]
    cols = [c for c in cols if c != "forgetful_rl/data_removal"]
    data = data[cols].copy().dropna(how="all")
    data["seed"] = run["seed"]
    data["forgetful"] = run["forgetful"]
    return data
