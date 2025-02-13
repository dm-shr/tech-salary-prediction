from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


class DistributionDriftChecker:
    """Class for checking distribution drift between two datasets."""

    def __init__(self, target_col: str = "log_salary_from"):
        self.target_col = target_col

    def calculate_drift_metrics(
        self, historical_data: pd.DataFrame, new_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate multiple drift metrics between historical and new data.

        Args:
            historical_data: DataFrame containing historical data
            new_data: DataFrame containing new data

        Returns:
            Dictionary containing different drift metrics
        """
        hist_target = historical_data[self.target_col].values
        new_target = new_data[self.target_col].values

        # Calculate KS test
        ks_statistic, ks_pvalue = stats.ks_2samp(hist_target, new_target)

        # Calculate Jensen-Shannon divergence
        js_divergence = self._calculate_js_divergence(hist_target, new_target)

        # Calculate Population Stability Index (PSI)
        psi_score = self._calculate_psi(hist_target, new_target)

        return {
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "js_divergence": js_divergence,
            "psi": psi_score,
        }

    def check_drift(
        self,
        historical_data: pd.DataFrame,
        new_data: pd.DataFrame,
        threshold_config: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if there's significant drift between distributions.

        Args:
            historical_data: DataFrame containing historical data
            new_data: DataFrame containing new data
            threshold_config: Dictionary containing thresholds for different metrics

        Returns:
            Tuple of (is_drift_detected, metrics_dict)
        """
        metrics = self.calculate_drift_metrics(historical_data, new_data)

        # Check if any metric exceeds its threshold
        is_drift = (
            metrics["ks_pvalue"] < threshold_config.get("ks_pvalue_threshold", 0.05)
            or metrics["js_divergence"] > threshold_config.get("js_divergence_threshold", 0.1)
            or metrics["psi"] > threshold_config.get("psi_threshold", 0.2)
        )

        return is_drift, metrics

    def _calculate_js_divergence(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        # Determine optimal number of bins using Freedman-Diaconis rule
        def get_optimal_bins(data):
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            bin_width = 2 * iqr / (len(data) ** (1 / 3))  # Freedman-Diaconis rule
            data_range = np.ptp(data)
            if bin_width == 0:  # Handle case when IQR is 0
                return min(20, len(np.unique(data)))
            n_bins = int(np.ceil(data_range / bin_width))
            return min(n_bins, 20)  # Cap at 20 bins

        # Combine data to get common binning
        combined = np.concatenate([dist1, dist2])
        n_bins = get_optimal_bins(combined)

        # Use numpy histogram directly instead of KBinsDiscretizer
        hist_range = (np.min(combined), np.max(combined))
        hist1, _ = np.histogram(dist1, bins=n_bins, range=hist_range, density=True)
        hist2, _ = np.histogram(dist2, bins=n_bins, range=hist_range, density=True)

        # Add small constant and normalize
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        m = 0.5 * (hist1 + hist2)
        return 0.5 * (stats.entropy(hist1, m) + stats.entropy(hist2, m))

    def _calculate_psi(self, dist1: np.ndarray, dist2: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Calculate bin edges based on dist1
        bin_edges = np.percentile(dist1, np.linspace(0, 100, bins + 1))
        bin_edges[-1] = float("inf")  # Ensure all values are captured

        # Calculate bin proportions for both distributions
        hist1 = np.histogram(dist1, bins=bin_edges)[0] / len(dist1)
        hist2 = np.histogram(dist2, bins=bin_edges)[0] / len(dist2)

        # Add small constant to avoid division by zero or log(0)
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10

        return np.sum((hist2 - hist1) * np.log(hist2 / hist1))
