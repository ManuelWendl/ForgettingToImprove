import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Global cache for datasets
_DATASET_CACHE = {}
# Global cache for normalization parameters
_NORMALIZATION_PARAMS = {}

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
        Name of dataset for storing normalization parameters
        
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
    
    # Store normalization parameters for potential denormalization
    _NORMALIZATION_PARAMS[dataset_name] = {
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'x_mean': x_scaler.mean_,
        'x_std': x_scaler.scale_,
        'y_mean': y_scaler.mean_[0],
        'y_std': y_scaler.scale_[0]
    }
    
    print("Data normalized:")
    print(f"  Original X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Normalized X range: [{x_normalized.min():.3f}, {x_normalized.max():.3f}]")
    print(f"  Original y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Normalized y range: [{y_normalized.min():.3f}, {y_normalized.max():.3f}]")
    
    return x_normalized, y_normalized

def load_boston() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Boston housing dataset with caching and fallback options."""
    
    # Check if already cached
    if 'boston' in _DATASET_CACHE:
        print("Using cached Boston dataset")
        return _DATASET_CACHE['boston']
    
    # Try multiple data sources in order of preference
    data_sources = [
        ("GitHub CSV", lambda: _load_boston_csv()),
        ("CMU Original", lambda: _load_boston_cmu()),
    ]
    
    for source_name, loader_func in data_sources:
        try:
            print(f"Attempting to load Boston dataset from: {source_name}")
            x, y = loader_func()
            print(f"Successfully loaded Boston dataset from {source_name}")
            print(f"Dataset shape: x={x.shape}, y={y.shape}")
            
            # Normalize the data before caching
            x_normalized, y_normalized = _normalize_data(x, y, 'boston')
            
            # Cache the normalized dataset
            _DATASET_CACHE['boston'] = (x_normalized, y_normalized)
            return x_normalized, y_normalized
            
        except Exception as e:
            print(f"Failed to load from {source_name}: {e}")
            continue
    
    raise RuntimeError("All data sources failed to load Boston dataset")


def load_twitter() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Twitter flash crash dataset from botorch GitHub repository.
    
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
        print("Using cached Twitter flash crash dataset")
        return _DATASET_CACHE['twitter']
    
    # Load from botorch GitHub repository (tutorials data not included in pip package)
    try:
        url = "https://raw.githubusercontent.com/pytorch/botorch/main/tutorials/data/twitter_flash_crash.csv"
        print(f"Loading Twitter flash crash data from botorch GitHub: {url}")
        data = pd.read_csv(url, index_col=0)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load twitter_flash_crash.csv from GitHub: {e}. "
            "Please check your internet connection."
        )
    
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
        print("Using cached CASP dataset")
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


def _load_boston_csv() -> Tuple[np.ndarray, np.ndarray]:
    """Load Boston dataset from GitHub CSV."""
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    y = df['medv'].values
    x = df.drop('medv', axis=1).values
    print(f"GitHub CSV loaded: x.shape={x.shape}, y.shape={y.shape}")
    return x, y

def _load_boston_cmu() -> Tuple[np.ndarray, np.ndarray]:
    """Load Boston dataset from CMU using the official sklearn-recommended parsing.
    
    The CMU format has 14 values per observation split across 2 lines.
    Following sklearn's recommended parsing code.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    
    # Use sklearn's recommended parsing (from their deprecation message)
    x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]
    
    print(f"CMU Boston loaded: x.shape={x.shape}, y.shape={y.shape}")
    return x, y


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
        # Add more datasets here as needed
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(datasets.keys())}")
    
    return datasets[dataset_name]()

def clear_dataset_cache():
    """Clear the dataset cache to force reloading."""
    global _DATASET_CACHE, _NORMALIZATION_PARAMS
    _DATASET_CACHE.clear()
    _NORMALIZATION_PARAMS.clear()
    print("Dataset cache and normalization parameters cleared")

def get_cached_datasets():
    """Get list of currently cached datasets."""
    return list(_DATASET_CACHE.keys())

def get_normalization_params(dataset_name: str) -> Dict[str, Any]:
    """
    Get normalization parameters for a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing normalization parameters
    """
    if dataset_name not in _NORMALIZATION_PARAMS:
        raise ValueError(f"No normalization parameters found for dataset: {dataset_name}")
    return _NORMALIZATION_PARAMS[dataset_name]

def denormalize_predictions(y_pred: np.ndarray, y_std: Optional[np.ndarray], dataset_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Denormalize predictions back to original scale.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Normalized predictions
    y_std : np.ndarray, optional
        Normalized standard deviations
    dataset_name : str
        Name of dataset to get normalization parameters
        
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        Denormalized predictions and standard deviations
    """
    params = get_normalization_params(dataset_name)
    y_scaler = params['y_scaler']
    
    # Denormalize predictions
    y_pred_denorm = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Denormalize standard deviations (scale only, no shift for std)
    if y_std is not None:
        y_std_denorm = y_std * params['y_std']
        return y_pred_denorm, y_std_denorm
    
    return y_pred_denorm, None

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset for setting appropriate limits.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing dataset information
    """
    if dataset_name == 'boston':
        return {
            'feature_names': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
            'n_features': 13,
            'is_multidimensional': True
        }
    elif dataset_name == 'twitter':
        return {
            'feature_names': ['Time'],
            'n_features': 1,
            'is_multidimensional': False,
            'description': 'Twitter flash crash - DJIA prices on April 23, 2013'
        }
    elif dataset_name == 'casp':
        return {
            'feature_names': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
            'n_features': 9,
            'is_multidimensional': True,
            'description': 'CASP - Protein structure prediction physicochemical properties'
        }
    
    raise ValueError(f"No information available for dataset: {dataset_name}")