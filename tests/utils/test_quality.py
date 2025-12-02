import pytest
import numpy as np
from isgri.utils.quality import QualityMetrics


def test_quality_metrics_init():
    """Test QualityMetrics initialization."""
    qm = QualityMetrics(binsize=1.0, emin=30, emax=300)
    assert qm.lightcurve is None
    assert qm.binsize == 1.0

def test_chi_squared_raw_constant():
    """Test chi-squared for constant lightcurve (should be ~0)."""
    counts = np.ones(100) * 10
    qm = QualityMetrics()
    chi = qm.raw_chi_squared(counts=counts)
    assert chi < 0.1  # Should be very small


def test_chi_squared_raw_poisson():
    """Test chi-squared for Poisson-like data."""
    np.random.seed(42)
    counts = np.random.poisson(10, 100)
    qm = QualityMetrics()
    chi = qm.raw_chi_squared(counts=counts)
    assert 0.5 < chi < 2.0  # Should be ~1 for Poisson


def test_chi_squared_clipped():
    """Test sigma-clipped chi-squared removes outliers."""
    counts = np.ones(100) * 10
    counts[50] = 100  # Add outlier
    
    qm = QualityMetrics()
    chi_raw = qm.raw_chi_squared(counts=counts)
    chi_clipped = qm.sigma_clip_chi_squared(counts=counts, sigma=3.0)
    
    assert chi_clipped < chi_raw  # Clipped should be smaller

