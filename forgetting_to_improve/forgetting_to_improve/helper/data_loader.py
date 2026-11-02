import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Global cache for datasets
_DATASET_CACHE = {}

def _normalize_data(x: np.ndarray, y: np.ndarray, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize features and target using StandardScaler (z-score normalization).
    
    Parameters
    ----------
    x : np.ndarray
        Input features
    y : np.ndarray
        Target values
    dataset_name : str
        Name of the dataset (unused, kept for the log output of the callers)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Normalized x and y arrays
    """
    # Normalize features (X)
    x_scaler = StandardScaler()
    x_normalized = x_scaler.fit_transform(x)
    
    # Normalize target (y)
    y_scaler = StandardScaler()
    y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    print("Data normalized:")
    print(f"  Original X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Normalized X range: [{x_normalized.min():.3f}, {x_normalized.max():.3f}]")
    print(f"  Original y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Normalized y range: [{y_normalized.min():.3f}, {y_normalized.max():.3f}]")
    
    return x_normalized, y_normalized

def load_boston() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Boston housing dataset (bundled BostonHousing.csv)."""
    if 'boston' in _DATASET_CACHE:
        return _DATASET_CACHE['boston']

    df = pd.read_csv(Path(__file__).parent / "BostonHousing.csv")
    y = df['medv'].values
    x = df.drop('medv', axis=1).values
    print(f"Boston dataset loaded: x={x.shape}, y={y.shape}")

    # Normalize the data before caching
    x_normalized, y_normalized = _normalize_data(x, y, 'boston')
    _DATASET_CACHE['boston'] = (x_normalized, y_normalized)
    return x_normalized, y_normalized


def load_twitter() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Twitter flash crash dataset (bundled copy of the BoTorch tutorial data).
    
    This dataset contains Dow Jones Industrial Average (DJIA) prices on April 23, 2013,
    when a fake tweet about an explosion at the White House caused a brief market crash.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x: Time indices (normalized, 1D)
        y: DJIA prices (normalized, 1D)
    """
    # Check if already cached
    if 'twitter' in _DATASET_CACHE:
        return _DATASET_CACHE['twitter']
    
    # BoTorch tutorials/data/twitter_flash_crash.csv
    data = pd.read_csv(Path(__file__).parent / "twitter_flash_crash.csv", index_col=0)
    
    # Convert timestamp to numeric (minutes since start)
    data["Time"] = pd.to_datetime(data["Time"])
    time_numeric = (data["Time"] - data["Time"].min()).dt.total_seconds() / 60.0
    
    # Extract features and target
    x = time_numeric.values.reshape(-1, 1)  # Time in minutes (1D)
    y = data["Price"].values  # DJIA price
    
    print(f"Twitter flash crash dataset loaded: x={x.shape}, y={y.shape}")
    print(f"Original time range: {data['Time'].min()} to {data['Time'].max()}")
    print(f"Time range in minutes: [{x.min():.1f}, {x.max():.1f}]")
    print(f"Price range: [{y.min():.1f}, {y.max():.1f}]")
    
    # Normalize the data before caching
    x_normalized, y_normalized = _normalize_data(x, y, 'twitter')
    
    # Cache the normalized dataset
    _DATASET_CACHE['twitter'] = (x_normalized, y_normalized)
    return x_normalized, y_normalized

def load_yacht() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Yacht Hydrodynamics dataset.
    
    This dataset contains variations in hull geometry coefficients and Froude number
    to predict residuary resistance per unit weight of displacement.
    
    Features:
    1. Longitudinal position of the center of buoyancy (adimensional)
    2. Prismatic coefficient (adimensional)
    3. Length-displacement ratio (adimensional)
    4. Beam-draught ratio (adimensional)
    5. Length-beam ratio (adimensional)
    6. Froude number (adimensional)
    
    Target: Residuary resistance per unit weight of displacement (adimensional)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x: 6 hull geometry features (normalized)
        y: Residuary resistance values (normalized)
    """
    # Check if already cached
    if 'yacht' in _DATASET_CACHE:
        return _DATASET_CACHE['yacht']
    
    # Load from local data file
    try:
        # Get the path to the data file relative to this module
        data_path = Path(__file__).parent / "yacht_hydrodynamics.data"
        print(f"Loading Yacht Hydrodynamics dataset from: {data_path}")
        # Load space-separated values
        data = np.loadtxt(data_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load yacht_hydrodynamics.data: {e}. "
            "Please ensure the file exists at forgetting_to_improve/forgetting_to_improve/helper/yacht_hydrodynamics.data"
        )
    
    # Extract features (columns 0-5) and target (column 6)
    x = data[:, :6]  # 6 features
    y = data[:, 6]   # Target: Residuary resistance
    
    print(f"Yacht Hydrodynamics dataset loaded: x={x.shape}, y={y.shape}")
    print(f"Feature ranges:")
    feature_names = ['LCB', 'PC', 'LDR', 'BDR', 'LBR', 'Froude']
    for i, name in enumerate(feature_names):
        print(f"  {name}: [{x[:, i].min():.3f}, {x[:, i].max():.3f}]")
    print(f"Residuary resistance range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Normalize the data before caching
    x_normalized, y_normalized = _normalize_data(x, y, 'yacht')
    
    # Cache the normalized dataset
    _DATASET_CACHE['yacht'] = (x_normalized, y_normalized)
    return x_normalized, y_normalized


def load_casp() -> Tuple[np.ndarray, np.ndarray]:
    """Load the CASP (Critical Assessment of protein Structure Prediction) dataset.
    
    This dataset contains physicochemical properties of protein tertiary structures
    and their root-mean-square deviation (RMSD) from experimentally determined structures.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x: 9 physicochemical features (F1-F9) (normalized)
        y: RMSD values (normalized)
    """
    # Check if already cached
    if 'casp' in _DATASET_CACHE:
        return _DATASET_CACHE['casp']
    
    # Load from local CSV file
    try:
        # Get the path to the CSV file relative to this module
        csv_path = Path(__file__).parent / "CASP.csv"
        print(f"Loading CASP dataset from: {csv_path}")
        data = pd.read_csv(csv_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load CASP.csv: {e}. "
            "Please ensure the file exists at forgetting_to_improve/forgetting_to_improve/helper/CASP.csv"
        )
    
    # Extract features (F1-F9) and target (RMSD)
    feature_cols = [f'F{i}' for i in range(1, 10)]  # F1, F2, ..., F9
    x = data[feature_cols].values  # 9 features
    y = data['RMSD'].values  # Target: Root Mean Square Deviation
    
    print(f"CASP dataset loaded: x={x.shape}, y={y.shape}")
    print(f"Feature ranges:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: [{data[col].min():.2f}, {data[col].max():.2f}]")
    print(f"RMSD range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Normalize the data before caching
    x_normalized, y_normalized = _normalize_data(x, y, 'casp')
    
    # Cache the normalized dataset
    _DATASET_CACHE['casp'] = (x_normalized, y_normalized)
    return x_normalized, y_normalized


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset by name.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x and y data arrays (x can be multidimensional)
    """
    datasets = {
        'boston': load_boston,
        'twitter': load_twitter,
        'casp': load_casp,
        'yacht': load_yacht,
        # Add more datasets here as needed
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(datasets.keys())}")
    
    return datasets[dataset_name]()
