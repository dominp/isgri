import pytest
import numpy as np
from astropy.io import fits
import tempfile
import os
from isgri.utils.lightcurve import LightCurve


@pytest.fixture
def mock_events_file():
    n_events = 1000

    events = np.zeros(
        n_events,
        dtype=[
            ("TIME", "f8"),
            ("ISGRI_ENERGY", "f4"),
            ("DETY", "i2"),
            ("DETZ", "i2"),
            ("SELECT_FLAG", "i2"),
        ],
    )

    events["TIME"] = np.linspace(0, 100 / 86400, n_events)
    events["ISGRI_ENERGY"] = np.random.uniform(30, 300, n_events)
    events["DETY"] = np.random.randint(0, 128, n_events)
    events["DETZ"] = np.random.randint(0, 134, n_events)
    events["SELECT_FLAG"] = 0

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fits") as f:
        hdu = fits.BinTableHDU(data=events, name="ISGR-EVTS-ALL")
        hdu.header["REVOL"] = 1000
        hdu.header["SWID"] = "100000100010"
        hdu.header["TSTART"] = 0.0
        hdu.header["TSTOP"] = 100.0
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(f.name, overwrite=True)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


def test_lightcurve_load_from_file(mock_events_file):
    """Test loading LightCurve from FITS file."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert isinstance(lc, LightCurve)
    assert len(lc.time) > 0
    assert len(lc.energies) > 0
    assert len(lc.dety) > 0
    assert len(lc.detz) > 0
    assert lc.metadata is not None


def test_lightcurve_metadata(mock_events_file):
    """Test metadata extraction."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert lc.metadata["REVOL"] == 1000
    assert lc.metadata["SWID"] == "100000100010"
    assert lc.metadata["TSTART"] == 0.0
    assert lc.metadata["TSTOP"] == 100.0


def test_lightcurve_rebin_basic(mock_events_file):
    """Test basic rebinning."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    time, counts = lc.rebin(binsize=1.0, emin=30, emax=300)

    assert len(time) > 0
    assert len(counts) > 0
    assert len(time) == len(counts)
    assert np.all(counts >= 0)


def test_lightcurve_rebin_energy_filter(mock_events_file):
    """Test rebinning with energy filtering."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    time1, counts1 = lc.rebin(binsize=1.0, emin=50, emax=100)
    time2, counts2 = lc.rebin(binsize=1.0, emin=30, emax=300)

    assert np.sum(counts1) <= np.sum(counts2)


def test_lightcurve_rebin_by_modules(mock_events_file):
    """Test rebinning by detector modules."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    time, counts = lc.rebin_by_modules(binsize=1.0, emin=30, emax=300)

    assert len(time) > 0
    assert len(counts) == 8  # 8 modules
    for module_counts in counts:
        assert len(module_counts) == len(time)
        assert np.all(module_counts >= 0)


def test_lightcurve_rebin_modules_vs_full_detector(mock_events_file):
    """Test that sum of module lightcurves equals full detector lightcurve."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    time_modules, counts_modules = lc.rebin_by_modules(binsize=1.0, emin=30, emax=300)
    time_full, counts_full = lc.rebin(binsize=1.0, emin=30, emax=300)

    module_sum = np.sum([counts for counts in counts_modules], axis=0)

    assert np.allclose(time_modules, time_full)
    assert np.allclose(module_sum, counts_full, rtol=0.01)


def test_lightcurve_time_conversion_ijd2loc(mock_events_file):
    """Test IJD to local time conversion."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    local = lc.ijd2loc(lc.time[:10])

    assert len(local) == 10
    assert np.all(local >= 0)
    assert np.all(local <= (lc.time[-1] - lc.t0) * 86400)


def test_lightcurve_time_conversion_loc2ijd(mock_events_file):
    """Test local time to IJD conversion."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    local = np.array([0, 10, 50])  # seconds
    ijd = lc.loc2ijd(local)

    assert len(ijd) == 3
    assert np.allclose(ijd, lc.t0 + local / 86400)


def test_lightcurve_time_conversion_roundtrip(mock_events_file):
    """Test time conversion round trip."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    ijd_original = lc.time[:10]
    local = lc.ijd2loc(ijd_original)
    ijd_back = lc.loc2ijd(local)

    assert np.allclose(ijd_original, ijd_back)


def test_lightcurve_cts_method(mock_events_file):
    """Test cts() count extraction."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    counts = lc.cts(0, 100, emin=30, emax=300)

    assert isinstance(counts, (float, np.floating))
    assert counts > 0


def test_lightcurve_cts_time_range(mock_events_file):
    """Test cts() with different time ranges."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    counts_full = lc.cts(0, 100, emin=30, emax=300)
    counts_half = lc.cts(0, 50, emin=30, emax=300)

    assert counts_half <= counts_full
    assert counts_half > 0


def test_lightcurve_gtis(mock_events_file):
    """Test GTI handling."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert lc.gtis is None


def test_lightcurve_pif_default(mock_events_file):
    """Test default PIF behavior (no PIF file)."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    # Without PIF file, all events should have PIF=1
    assert np.all(lc.weights == 1.0)


def test_lightcurve_invalid_file():
    """Test loading from invalid path."""
    with pytest.raises(FileNotFoundError):
        LightCurve.load_data(events_path="/nonexistent/file.fits")


def test_lightcurve_rebin_invalid_energy(mock_events_file):
    """Test rebinning with invalid energy range."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    with pytest.raises(ValueError):
        lc.rebin(binsize=1.0, emin=300, emax=30)


def test_lightcurve_rebin_custom_bins(mock_events_file):
    """Test rebinning with custom bin edges."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    custom_bins = [0, 10, 25, 50, 100]  # seconds
    time, counts = lc.rebin(binsize=custom_bins, emin=30, emax=300)

    assert len(time) == len(custom_bins) - 1
    assert len(counts) == len(custom_bins) - 1
    assert np.all(counts >= 0)


def test_lightcurve_rebin_with_custom_mask(mock_events_file):
    """Test rebinning with custom event mask."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    custom_mask = np.ones(len(lc.time), dtype=bool)  # exclude first half of events
    custom_mask[: len(lc.time) // 2] = False

    time_masked, counts_masked = lc.rebin(binsize=1.0, emin=30, emax=300, custom_mask=custom_mask)
    time_full, counts_full = lc.rebin(binsize=1.0, emin=30, emax=300)

    assert np.sum(counts_masked) < np.sum(counts_full)


def test_lightcurve_module_assignment():
    """Test that events are correctly assigned to detector modules."""
    # Create synthetic events with DIFFERENT counts per module to verify assignment
    module_counts_expected = [10, 25, 50, 75, 100, 125, 150, 200]  # Different for each module
    n_events = sum(module_counts_expected)

    events = np.zeros(
        n_events,
        dtype=[
            ("TIME", "f8"),
            ("ISGRI_ENERGY", "f4"),
            ("DETY", "i2"),
            ("DETZ", "i2"),
            ("SELECT_FLAG", "i2"),
        ],
    )

    # Module layout: 0 1
    #                2 3
    #                4 5
    #                6 7
    module_positions = [
        (16, 32),  # Module 0: DETZ [0-32),   DETY [0-64)
        (16, 96),  # Module 1: DETZ [0-32),   DETY [64-130)
        (48, 32),  # Module 2: DETZ [32-66),  DETY [0-64)
        (48, 96),  # Module 3: DETZ [32-66),  DETY [64-130)
        (80, 32),  # Module 4: DETZ [66-100), DETY [0-64)
        (80, 96),  # Module 5: DETZ [66-100), DETY [64-130)
        (116, 32),  # Module 6: DETZ [100-134), DETY [0-64)
        (116, 96),  # Module 7: DETZ [100-134), DETY [64-130)
    ]

    idx = 0
    for module_no, (detz, dety) in enumerate(module_positions):
        n_events_module = module_counts_expected[module_no]

        events["DETZ"][idx : idx + n_events_module] = detz
        events["DETY"][idx : idx + n_events_module] = dety
        events["TIME"][idx : idx + n_events_module] = np.linspace(0, 10 / 86400, n_events_module)
        events["ISGRI_ENERGY"][idx : idx + n_events_module] = 100  # All same energy
        events["SELECT_FLAG"][idx : idx + n_events_module] = 0

        idx += n_events_module

    # Create LightCurve
    time = events["TIME"]
    energies = events["ISGRI_ENERGY"]
    dety = events["DETY"]
    detz = events["DETZ"]
    gtis = np.array([[time[0], time[-1]]])
    weights = np.ones(n_events)
    metadata = {}

    lc = LightCurve(time, energies, gtis, dety, detz, weights, metadata)

    # Rebin by modules (1 bin covering all time)
    times, counts = lc.rebin_by_modules(binsize=20.0, emin=50, emax=200, local_time=True)

    # Verify each module has the expected count
    for module_no, expected_count in enumerate(module_counts_expected):
        actual_count = counts[module_no][0]
        assert actual_count == expected_count, (
            f"Module {module_no} at position {module_positions[module_no]}: "
            f"expected {expected_count} counts, got {actual_count}"
        )


@pytest.fixture
def mock_pif_file():
    pif_data = np.ones((134, 130), dtype=np.float32)
    pif_data[0:50, :] = 0.8
    pif_data[50:100, :] = 0.4
    pif_data[100:, :] = 0.2

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fits") as f:
        hdu = fits.ImageHDU(data=pif_data, name="ISGR-PIF.-ima")
        hdu.header["SOURCE"] = "TEST_SOURCE"
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(f.name, overwrite=True)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


@pytest.fixture
def mock_events_with_pif_coords():
    n_events = 300
    events = np.zeros(
        n_events,
        dtype=[
            ("TIME", "f8"),
            ("ISGRI_ENERGY", "f4"),
            ("DETY", "i2"),
            ("DETZ", "i2"),
            ("SELECT_FLAG", "i2"),
        ],
    )

    events["TIME"][:100] = np.linspace(0, 100 / 86400, 100)
    events["DETZ"][:100] = 25
    events["DETY"][:100] = 65
    events["ISGRI_ENERGY"][:100] = 100

    events["TIME"][100:200] = np.linspace(0, 100 / 86400, 100)
    events["DETZ"][100:200] = 75
    events["DETY"][100:200] = 65
    events["ISGRI_ENERGY"][100:200] = 100

    events["TIME"][200:] = np.linspace(0, 100 / 86400, 100)
    events["DETZ"][200:] = 115
    events["DETY"][200:] = 65
    events["ISGRI_ENERGY"][200:] = 100

    events["SELECT_FLAG"] = 0

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fits") as f:
        hdu = fits.BinTableHDU(data=events, name="ISGR-EVTS-ALL")
        hdu.header["REVOL"] = 1000
        hdu.header["SWID"] = "100000100010"
        hdu.header["TSTART"] = 0.0
        hdu.header["TSTOP"] = 100.0
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(f.name, overwrite=True)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


def test_lightcurve_pif_loading(mock_events_with_pif_coords, mock_pif_file):
    """Test PIF weights loaded correctly."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    assert not np.all(lc.weights == 1.0)
    assert lc.use_pif == True
    assert lc.pif_threshold == 0.5
    assert np.allclose(lc.weights[:100], 0.8, rtol=0.01)
    assert np.allclose(lc.weights[100:200], 0.4, rtol=0.01)
    assert np.allclose(lc.weights[200:], 0.2, rtol=0.01)


def test_lightcurve_pif_rebin(mock_events_with_pif_coords, mock_pif_file):
    """Test rebinning with PIF threshold."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts = lc.rebin(binsize=200.0, emin=50, emax=200, local_time=True)
    assert np.isclose(counts.sum(), 100 * 0.8, rtol=0.1)


def test_lightcurve_pif_threshold_override(mock_events_with_pif_coords, mock_pif_file):
    """Test overriding PIF threshold."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts_low = lc.rebin(binsize=200.0, emin=50, emax=200, pif_threshold=0.3)
    assert np.isclose(counts_low.sum(), 100 * 0.8 + 100 * 0.4, rtol=0.1)

    time, counts_high = lc.rebin(binsize=200.0, emin=50, emax=200, pif_threshold=0.9)
    assert counts_high.sum() == 0


def test_lightcurve_pif_toggle(mock_events_with_pif_coords, mock_pif_file):
    """Test toggling PIF on/off."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts_pif = lc.rebin(binsize=200.0, emin=50, emax=200)
    time, counts_no_pif = lc.rebin(binsize=200.0, emin=50, emax=200, use_pif=False)

    assert counts_no_pif.sum() == 300
    assert counts_pif.sum() < counts_no_pif.sum()


def test_lightcurve_pif_instance_settings(mock_events_with_pif_coords, mock_pif_file):
    """Test changing instance PIF settings."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts_with = lc.rebin(binsize=200.0, emin=50, emax=200)

    lc.use_pif = False
    time, counts_without = lc.rebin(binsize=200.0, emin=50, emax=200)
    assert counts_without.sum() > counts_with.sum()

    lc.use_pif = True
    lc.pif_threshold = 0.3
    time, counts_new = lc.rebin(binsize=200.0, emin=50, emax=200)
    assert counts_new.sum() > counts_with.sum()


def test_lightcurve_pif_caching(mock_events_with_pif_coords, mock_pif_file):
    """Test weight caching."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    weights1 = lc._get_weights()
    weights2 = lc._get_weights()
    assert weights1 is weights2

    weights3 = lc._get_weights(pif_threshold=0.3)
    assert weights3 is not weights1


def test_lightcurve_pif_get_method(mock_events_with_pif_coords, mock_pif_file):
    """Test get() method with PIF."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, energy, dety, detz, weights = lc.get()
    assert len(time) == 100
    assert np.all(weights >= 0.5)

    time, energy, dety, detz, weights = lc.get(pif_threshold=0.3)
    assert len(time) == 200
    assert np.all(weights >= 0.3)

    time, energy, dety, detz, weights = lc.get(use_pif=False)
    assert len(time) == 300
    assert np.all(weights == 1.0)


def test_lightcurve_pif_rebin_by_modules(mock_events_with_pif_coords, mock_pif_file):
    """Test PIF in rebin_by_modules."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts_pif = lc.rebin_by_modules(binsize=200.0, emin=50, emax=200)
    time, counts_no_pif = lc.rebin_by_modules(binsize=200.0, emin=50, emax=200, use_pif=False)

    total_pif = sum(c.sum() for c in counts_pif)
    total_no_pif = sum(c.sum() for c in counts_no_pif)

    assert total_no_pif == 300
    assert total_pif < total_no_pif


def test_lightcurve_pif_cts(mock_events_with_pif_coords, mock_pif_file):
    """Test PIF in cts method."""
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    counts_pif = lc.cts(0, 200, emin=50, emax=200, local_time=True)
    counts_no_pif = lc.cts(0, 200, emin=50, emax=200, local_time=True, use_pif=False)

    assert counts_no_pif == 300
    assert counts_pif < counts_no_pif


def test_lightcurve_no_pif_file(mock_events_file):
    """Test loading without PIF file."""
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert np.all(lc.weights == 1.0)
    assert lc.use_pif == False

    time, counts = lc.rebin(binsize=1.0, emin=30, emax=300)
    assert len(counts) > 0
