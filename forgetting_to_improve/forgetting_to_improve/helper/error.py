import numpy as np
from typing import Optional
from scipy import stats
from .plot_gp import plot_calibration_curve

def calculate_prediction_errors(
    x_data: np.ndarray,
    y_prediction: np.ndarray,
    stddev_prediction: np.ndarray,
    y_test: np.ndarray,
    region = None,
    plot: bool = False,
    title: Optional[str] = None
) -> dict:
    """
    Calculate error metrics and calibration error for GP predictions.
    
    Parameters
    ----------
    x_data : np.ndarray
        Input features (can be 1D or 2D)
    y_prediction : np.ndarray
        Predicted values
    stddev_prediction : np.ndarray
        Predicted standard deviations
    y_test : np.ndarray
        True target values
    region : tuple, list of tuples, or None
        Region limits. For 1D: (min, max). For multidimensional: [(min1, max1), (min2, max2), ...]
    plot : bool
        Whether to create plots
    title : str, optional
        Title for plots
    """
    # Filter data to target region if specified
    if region is not None:
        # Handle both 1D and multidimensional regions
        if x_data.ndim == 1 or (x_data.ndim == 2 and x_data.shape[1] == 1):
            # 1D case
            x_flat = x_data.flatten() if x_data.ndim == 2 else x_data
            if isinstance(region[0], (list, tuple)):
                # Multidimensional region specified but data is 1D - use first dimension
                min_x, max_x = region[0]
            else:
                # Simple 1D region
                min_x, max_x = region
            mask = (x_flat >= min_x) & (x_flat <= max_x)
        else:
            # Multidimensional case
            if not isinstance(region[0], (list, tuple)):
                # Same region for all dimensions
                region = [region] * x_data.shape[1]
            
            mask = np.ones(len(x_data), dtype=bool)
            for i, (min_val, max_val) in enumerate(region):
                mask &= (x_data[:, i] >= min_val) & (x_data[:, i] <= max_val)
        
        y_pred_filtered = y_prediction[mask]
        stddev_filtered = stddev_prediction[mask]
        y_true_filtered = y_test[mask]
    else:
        y_pred_filtered = y_prediction
        stddev_filtered = stddev_prediction
        y_true_filtered = y_test
    
    if len(y_pred_filtered) == 0:
        raise ValueError("No data points found in the specified region")
    
    # Calculate basic error metrics
    errors = y_pred_filtered - y_true_filtered
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    
    # Calculate negative log-likelihood for Gaussian: -log p(y|μ,σ) = 0.5*log(2πσ²) + (y-μ)²/(2σ²)
    nll = 0.5 * np.mean(np.log(2 * np.pi * stddev_filtered**2) + (errors**2 / (stddev_filtered**2)))

    # Calculate calibration error
    calibration_error = kuleshov_calibration_error(y_true_filtered, y_pred_filtered, stddev_filtered, plot=plot, title=title)

    # Additional calibration metrics
    ece_variance = ece_regression_levi(y_true_filtered, y_pred_filtered, stddev_filtered, plot=plot, title=title)
    ece_interval = ece_regression_capone(errors, stddev_filtered, plot=plot, title=title)
    sharpness = sharpness_metric(stddev_filtered)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'NLL': nll,
        'CE': calibration_error,
        'V-ECE': ece_variance,
        'I-ECE': ece_interval,
        'Sharpness': sharpness,
    }


def kuleshov_calibration_error(
    y_test: np.ndarray,
    y_pred_mu: np.ndarray,
    y_pred_std: np.ndarray,
    p_levels: np.ndarray = None,
    weights: np.ndarray = None,
    plot: bool = False,
    title: Optional[str] = None
) -> float:
    """
    Two-sided (central-interval) version of the Kuleshov et al. (2018)
    calibration error.

    Checks how well the *central predictive intervals* of nominal mass p
    correspond to empirical coverage.
    """
    if p_levels is None:
        p_levels = np.linspace(0.0, 1.0, 10)
    m = len(p_levels)

    if weights is None:
        weights = np.ones(m) / m
    else:
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()

    z = np.abs((y_test - y_pred_mu) / y_pred_std)  # standardized absolute residuals

    # Empirical coverage for each nominal central interval
    p_hat = np.empty(m)
    for i, p in enumerate(p_levels):
        alpha = (1 - p) / 2.0
        z_thresh = stats.norm.ppf(1 - alpha)  # two-sided z threshold
        p_hat[i] = np.mean(z <= z_thresh)     # fraction inside central interval

    sq_diffs = (p_levels - p_hat) ** 2

    if plot:
        plot_calibration_curve(p_levels, p_hat, title=title + " CE" if title else "CE")

    return float(np.sum(weights * sq_diffs))

def ece_regression_levi(
    y: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    n_bins: int = 10,
    use_rmse: bool = False,
    plot: bool = False,
    title: str = "V-ECE Calibration Curve"
) -> float:
    """
    Expected Calibration Error (ECE) for regression, as in:
      Levi et al., "Evaluating and Calibrating Uncertainty Prediction in Regression Tasks",
      arXiv:1905.11659 / Sensors, 2022.

    Optionally plots the calibration curve.
    """
    errors = np.abs(y - mu)
    
    # Use quantile-based binning instead of uniform binning
    # This ensures each bin has roughly equal number of samples
    bin_edges = np.quantile(std, np.linspace(0, 1, n_bins + 1))
    # Handle case where all std values are the same
    if np.all(bin_edges == bin_edges[0]):
        return float(np.abs(errors.mean() - std.mean()))

    ece = 0.0
    avg_pred_stds = []
    avg_errors = []

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        
        # For the last bin, include the upper boundary
        if i == n_bins - 1:
            in_bin = (std >= lower) & (std <= upper)
        else:
            in_bin = (std >= lower) & (std < upper)
            
        prop_in_bin = in_bin.mean()  # |B_m| / N

        if prop_in_bin > 0 and np.sum(in_bin) > 1:  # Require at least 2 samples per bin
            bin_pred_std = std[in_bin].mean()
            if use_rmse:
                bin_err = np.sqrt(np.mean((y[in_bin] - mu[in_bin]) ** 2))
            else:
                bin_err = np.mean(errors[in_bin])

            avg_pred_stds.append(bin_pred_std)
            avg_errors.append(bin_err)

            # Eq. (8): weighted absolute difference between empirical error and predicted std
            ece += prop_in_bin * np.abs(bin_err - bin_pred_std)
        else:
            avg_pred_stds.append(0)
            avg_errors.append(0)

    return float(ece)/n_bins

def ece_regression_capone(errors: np.ndarray, stddevs: np.ndarray, n_bins: int = 10, plot: bool = False, title: str = "I-ECE") -> float:
    """
    Calculate Expected Calibration Error (ECE) by binning predictions by confidence.
    Optionally plot the calibration curve.
    """
    # Calculate confidence scores (inverse of standard deviation)
    confidences = 1 / (1 + stddevs)
    
    # Bin edges for confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    avg_confidences = []
    accuracies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy (1 - normalized error) for samples in this bin
            bin_errors = np.abs(errors[in_bin])
            bin_stddevs = stddevs[in_bin]
            
            # Accuracy as proportion of errors within 1 standard deviation
            accuracy_in_bin = np.mean(bin_errors <= bin_stddevs)
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            avg_confidences.append(avg_confidence_in_bin)
            accuracies.append(accuracy_in_bin)
            
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece/n_bins


def sharpness_metric(stddevs: np.ndarray) -> float:
    """
    Calculate sharpness metric (average prediction uncertainty).
    Lower values indicate sharper (more confident) predictions.
    """
    return np.mean(stddevs)