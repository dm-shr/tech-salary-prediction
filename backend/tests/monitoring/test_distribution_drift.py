import logging

import numpy as np
import pandas as pd
import pytest
from src.monitoring.distribution_drift import DistributionDriftChecker


@pytest.fixture
def drift_checker():
    return DistributionDriftChecker(target_col="log_salary_from")


@pytest.fixture
def threshold_config():
    return {"ks_pvalue_threshold": 0.05, "js_divergence_threshold": 0.1, "psi_threshold": 0.2}


@pytest.fixture
def sample_historical_data():
    """Create sample historical data with log-normal distribution."""
    np.random.seed(42)
    n_samples = 1000
    return pd.DataFrame(
        {
            "log_salary_from": np.random.normal(loc=11.5, scale=0.3, size=n_samples),
            "other_column": range(n_samples),
        }
    )


@pytest.fixture
def sample_new_data_no_drift():
    """Create sample new data with similar distribution (no drift)."""
    np.random.seed(43)
    n_samples = 200
    return pd.DataFrame(
        {
            "log_salary_from": np.random.normal(loc=11.5, scale=0.3, size=n_samples),
            "other_column": range(n_samples),
        }
    )


@pytest.fixture
def sample_new_data_with_drift():
    """Create sample new data with different distribution (with drift)."""
    np.random.seed(44)
    n_samples = 200
    return pd.DataFrame(
        {
            "log_salary_from": np.random.normal(loc=12.0, scale=0.5, size=n_samples),
            "other_column": range(n_samples),
        }
    )


def test_target_drift_no_drift(
    drift_checker, threshold_config, sample_historical_data, sample_new_data_no_drift, caplog
):
    """Test drift detection with similar distributions."""
    caplog.set_level(logging.INFO)

    is_drift, metrics = drift_checker.check_drift(
        sample_historical_data, sample_new_data_no_drift, threshold_config
    )

    assert not is_drift, "Should not detect drift in similar distributions"
    assert "ks_pvalue" in metrics
    assert "js_divergence" in metrics
    assert "psi" in metrics


def test_target_drift_with_drift(
    drift_checker, threshold_config, sample_historical_data, sample_new_data_with_drift, caplog
):
    """Test drift detection with different distributions."""
    caplog.set_level(logging.INFO)

    is_drift, metrics = drift_checker.check_drift(
        sample_historical_data, sample_new_data_with_drift, threshold_config
    )

    assert is_drift, "Should detect drift in different distributions"
    assert metrics["ks_pvalue"] < threshold_config["ks_pvalue_threshold"]
