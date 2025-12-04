import pytest
import numpy as np
from isgri.utils.quality import QualityMetrics
from isgri.utils.lightcurve import LightCurve


@pytest.fixture
def mock_lightcurve():
    """Create a simple mock LightCurve for testing."""
    n_events = 1000
    time = np.linspace(0, 100 / 86400, n_events)
    energies = np.random.uniform(30, 300, n_events)
    gtis = np.array([[time[0], time[-1]]])
    dety = np.random.randint(0, 128, n_events)
    detz = np.random.randint(0, 134, n_events)
    weights = np.ones(n_events)
    metadata = {"SWID": "test"}

    return LightCurve(time, energies, gtis, dety, detz, weights, metadata)


def test_quality_metrics_init():
    """Test QualityMetrics initialization."""
    qm = QualityMetrics(binsize=1.0, emin=30, emax=300)

    assert qm.lightcurve is None
    assert qm.binsize == 1.0
    assert qm.emin == 30
    assert qm.emax == 300
    assert qm.local_time is False
    assert qm.module_data is None


def test_quality_metrics_init_with_lightcurve(mock_lightcurve):
    """Test QualityMetrics initialization with LightCurve."""
    qm = QualityMetrics(mock_lightcurve, binsize=1.0, emin=30, emax=300)

    assert qm.lightcurve is not None
    assert isinstance(qm.lightcurve, LightCurve)


def test_quality_metrics_init_invalid_type():
    """Test QualityMetrics raises error for invalid lightcurve type."""
    with pytest.raises(TypeError):
        QualityMetrics(lightcurve="not_a_lightcurve")


def test_compute_counts_without_lightcurve():
    """Test _compute_counts raises error when lightcurve is None."""
    qm = QualityMetrics()

    with pytest.raises(ValueError, match="Lightcurve is not set"):
        qm._compute_counts()


def test_compute_counts_caching(mock_lightcurve):
    """Test that _compute_counts caches results."""
    qm = QualityMetrics(mock_lightcurve, binsize=1.0, emin=30, emax=300)

    data1 = qm._compute_counts()
    data2 = qm._compute_counts()

    assert data1 is data2
    assert "time" in data1
    assert "counts" in data1


def test_chi_squared_constant_lightcurve():
    """Test chi-squared for constant lightcurve."""
    counts = np.ones(100) * 10.0
    qm = QualityMetrics()

    chi = qm.raw_chi_squared(counts=counts)

    assert chi < 0.1


def test_chi_squared_poisson_lightcurve():
    """Test chi-squared for Poisson-distributed data."""
    np.random.seed(42)
    counts = np.random.poisson(10, 1000)
    qm = QualityMetrics()

    chi = qm.raw_chi_squared(counts=counts)

    assert 0.7 < chi < 1.3


def test_chi_squared_variable_lightcurve():
    """Test chi-squared for highly variable lightcurve."""
    counts = np.concatenate([np.ones(50) * 5, np.ones(50) * 50])
    qm = QualityMetrics()

    chi = qm.raw_chi_squared(counts=counts)

    assert chi > 2.0


def test_chi_squared_1d_vs_2d():
    """Test chi-squared handles both 1D and 2D arrays."""
    np.random.seed(42)
    counts_1d = np.random.poisson(10, 100)
    counts_2d = np.random.poisson(10, (8, 100))

    qm = QualityMetrics()
    chi_1d = qm.raw_chi_squared(counts=counts_1d)
    chi_2d = qm.raw_chi_squared(counts=counts_2d)

    assert isinstance(chi_1d, (float, np.floating))
    assert isinstance(chi_2d, (float, np.floating))


def test_chi_squared_return_all():
    """Test chi-squared with return_all=True."""
    np.random.seed(42)
    counts = np.random.poisson(10, (8, 100))
    qm = QualityMetrics()

    chi, dof, no_counts = qm.raw_chi_squared(counts=counts, return_all=True)

    assert chi.shape == (8,)
    assert dof.shape == (8,)
    assert no_counts.shape == (8,)
    assert np.all(dof == 99)
    assert np.all(no_counts > 0)


def test_chi_squared_with_zeros():
    """Test chi-squared handles zero counts."""
    counts = np.array([10, 10, 0, 10, 10])
    qm = QualityMetrics()

    chi = qm.raw_chi_squared(counts=counts)

    assert np.isfinite(chi)
    assert chi < 0.5


def test_sigma_clip_removes_outliers():
    """Test sigma clipping removes outliers."""
    counts = np.ones(100) * 10.0
    counts[50] = 100.0

    qm = QualityMetrics()
    chi_raw = qm.raw_chi_squared(counts=counts)
    chi_clipped = qm.sigma_clip_chi_squared(counts=counts, sigma=3.0)

    assert chi_clipped < chi_raw
    assert chi_clipped < 0.5


def test_sigma_clip_different_thresholds():
    """Test different sigma thresholds."""
    np.random.seed(42)
    counts = np.random.poisson(10, 100)
    counts[50] = 100

    qm = QualityMetrics()
    chi_1sigma = qm.sigma_clip_chi_squared(counts=counts, sigma=1.0)
    chi_3sigma = qm.sigma_clip_chi_squared(counts=counts, sigma=3.0)

    assert chi_1sigma <= chi_3sigma


def test_sigma_clip_2d_arrays():
    """Test sigma clipping works with 2D arrays."""
    np.random.seed(42)
    counts = np.random.poisson(10, (8, 100))
    counts[:, 50] = 100

    qm = QualityMetrics()
    chi_raw = qm.raw_chi_squared(counts=counts)
    chi_clipped = qm.sigma_clip_chi_squared(counts=counts, sigma=3.0)

    assert chi_clipped < chi_raw


def test_gti_chi_squared_filters_correctly(mock_lightcurve):
    """Test GTI filtering applies correctly."""
    qm = QualityMetrics(mock_lightcurve, binsize=1.0, emin=30, emax=300, local_time=False)

    chi = qm.gti_chi_squared()

    assert np.isfinite(chi)
    assert chi > 0


def test_gti_chi_squared_no_overlap():
    """Test GTI chi-squared raises error when no overlap."""
    np.random.seed(42)
    time = np.linspace(0, 100, 100)
    counts = np.random.poisson(10, (8, 100))
    gtis = np.array([[200, 300]])

    qm = QualityMetrics()

    with pytest.raises(ValueError, match="No overlap"):
        qm.gti_chi_squared(time=time, counts=counts, gtis=gtis)


def test_gti_chi_squared_custom_gtis():
    """Test GTI chi-squared with custom GTI array."""
    np.random.seed(42)
    time = np.linspace(0, 100, 100)
    counts = np.random.poisson(10, (8, 100))
    gtis = np.array([[0, 50], [75, 100]])

    qm = QualityMetrics()
    chi = qm.gti_chi_squared(time=time, counts=counts, gtis=gtis)

    assert np.isfinite(chi)
    assert chi > 0


def test_gti_chi_squared_return_all():
    """Test GTI chi-squared with return_all=True."""
    np.random.seed(42)
    time = np.linspace(0, 100, 100)
    counts = np.random.poisson(10, (8, 100))
    gtis = np.array([[0, 100]])

    qm = QualityMetrics()
    chi, dof, no_counts = qm.gti_chi_squared(time=time, counts=counts, gtis=gtis, return_all=True)

    assert chi.shape == (8,)
    assert dof.shape == (8,)
    assert no_counts.shape == (8,)


def test_chi_squared_weighting():
    """Test chi-squared weighting by total counts."""
    counts = np.array(
        [
            np.concatenate([np.ones(50) * 5, np.ones(50) * 50]),
            np.ones(100) * 100,
        ]
    )

    qm = QualityMetrics()
    chi_weighted = qm.raw_chi_squared(counts=counts)
    chi_1, chi_2 = qm.raw_chi_squared(counts=counts, return_all=True)[0]

    # Weighted average closer to high-count module
    assert abs(chi_weighted - chi_2) < abs(chi_weighted - chi_1)


def test_chi_squared_all_nan():
    """Test chi-squared handles all-NaN arrays."""
    counts = np.full(100, np.nan)
    qm = QualityMetrics()

    chi = qm.raw_chi_squared(counts=counts)

    assert np.isnan(chi)


def test_integration_with_real_lightcurve(mock_lightcurve):
    """Test full workflow with real LightCurve."""
    qm = QualityMetrics(mock_lightcurve, binsize=1.0, emin=30, emax=300, local_time=False)

    chi_raw = qm.raw_chi_squared()
    chi_clipped = qm.sigma_clip_chi_squared(sigma=3.0)
    chi_gti = qm.gti_chi_squared()

    assert np.isfinite(chi_raw)
    assert np.isfinite(chi_clipped)
    assert np.isfinite(chi_gti)
    assert qm.module_data is not None
