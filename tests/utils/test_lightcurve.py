import pytest
import numpy as np
from astropy.io import fits
import tempfile
import os
import shutil
from pathlib import Path
from isgri.utils.lightcurve import LightCurve
from isgri.config import Config


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
        hdu.header["NAXIS2"] = n_events
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(f.name, overwrite=True)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


@pytest.fixture
def mock_archive_structure(tmp_path, mock_events_file):
    revol_dir = tmp_path / "1000"
    revol_dir.mkdir()

    swid_dir = revol_dir / "100000100010.001"
    swid_dir.mkdir()

    events_file = swid_dir / "isgri_events.fits.gz"
    shutil.copy(mock_events_file, events_file)

    return tmp_path, revol_dir, swid_dir, events_file


@pytest.fixture
def mock_pif_structure(tmp_path):
    source_dir = tmp_path / "Crab"
    source_dir.mkdir()

    revol_dir = source_dir / "1000"
    revol_dir.mkdir()

    pif_file = revol_dir / "100000100010_pif.fits"
    pif_data = np.ones((134, 130), dtype=np.float32) * 0.8
    hdu = fits.ImageHDU(data=pif_data, name="ISGR-PIF.-ima")
    hdu.header["SOURCE"] = "Crab"
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(pif_file)

    return tmp_path, source_dir, revol_dir, pif_file


@pytest.fixture
def mock_config(tmp_path):
    config_file = tmp_path / "test_config.toml"
    return Config(path=config_file)


def test_lightcurve_load_from_file(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert isinstance(lc, LightCurve)
    assert len(lc.time) > 0
    assert len(lc.energies) > 0
    assert len(lc.dety) > 0
    assert len(lc.detz) > 0
    assert lc.metadata is not None


def test_lightcurve_load_with_swid(mock_archive_structure, mock_config, monkeypatch):
    tmp_path, revol_dir, swid_dir, events_file = mock_archive_structure

    mock_config.create_new(archive_path=tmp_path)

    monkeypatch.setattr("isgri.utils.lightcurve.Config", lambda: mock_config)

    lc = LightCurve.load_data(swid="100000100010")

    assert isinstance(lc, LightCurve)
    assert len(lc.time) > 0
    assert lc.metadata["SWID"] == "100000100010"


def test_lightcurve_load_with_swid_and_source(tmp_path, mock_events_file, mock_config, monkeypatch):
    revol_dir = tmp_path / "archive" / "1000"
    revol_dir.mkdir(parents=True)

    swid_dir = revol_dir / "100000100010.001"
    swid_dir.mkdir()

    events_file = swid_dir / "isgri_events.fits.gz"
    shutil.copy(mock_events_file, events_file)

    pif_base = tmp_path / "pif"
    source_dir = pif_base / "Crab"
    source_dir.mkdir(parents=True)

    pif_revol_dir = source_dir / "1000"
    pif_revol_dir.mkdir()

    pif_file = pif_revol_dir / "100000100010_pif.fits"
    pif_data = np.ones((134, 130), dtype=np.float32) * 0.8
    hdu = fits.ImageHDU(data=pif_data, name="ISGR-PIF.-ima")
    hdu.header["SOURCE"] = "Crab"
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(pif_file)

    mock_config.create_new(archive_path=tmp_path / "archive", pif_path=pif_base)

    monkeypatch.setattr("isgri.utils.lightcurve.Config", lambda: mock_config)

    lc = LightCurve.load_data(swid="100000100010", source="Crab", use_pif=True, pif_threshold=0.5)

    assert isinstance(lc, LightCurve)
    assert len(lc.time) > 0
    assert lc.metadata["SWID"] == "100000100010"
    assert not np.all(lc.weights == 1.0)


def test_lightcurve_load_with_source_no_swid(mock_pif_structure, mock_config, monkeypatch):
    pif_path, source_dir, pif_revol_dir, pif_file = mock_pif_structure

    mock_config.create_new(pif_path=pif_path)

    monkeypatch.setattr("isgri.utils.lightcurve.Config", lambda: mock_config)

    with pytest.raises(ValueError):
        LightCurve.load_data(source="Crab")


def test_lightcurve_load_swid_no_config():
    with pytest.raises((ValueError, FileNotFoundError)):
        LightCurve.load_data(swid="100000100010")


def test_lightcurve_metadata(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert lc.metadata["REVOL"] == 1000
    assert lc.metadata["SWID"] == "100000100010"
    assert lc.metadata["TSTART"] == 0.0
    assert lc.metadata["TSTOP"] == 100.0


def test_lightcurve_rebin_basic(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    time, counts = lc.rebin(binsize=1.0, emin=30, emax=300)

    assert len(time) > 0
    assert len(counts) > 0
    assert len(time) == len(counts)
    assert np.all(counts >= 0)


def test_lightcurve_rebin_energy_filter(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    time1, counts1 = lc.rebin(binsize=1.0, emin=50, emax=100)
    time2, counts2 = lc.rebin(binsize=1.0, emin=30, emax=300)

    assert np.sum(counts1) <= np.sum(counts2)


def test_lightcurve_rebin_by_modules(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    time, counts = lc.rebin_by_modules(binsize=1.0, emin=30, emax=300)

    assert len(time) > 0
    assert len(counts) == 8
    for module_counts in counts:
        assert len(module_counts) == len(time)
        assert np.all(module_counts >= 0)


def test_lightcurve_rebin_modules_vs_full_detector(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    time_modules, counts_modules = lc.rebin_by_modules(binsize=1.0, emin=30, emax=300)
    time_full, counts_full = lc.rebin(binsize=1.0, emin=30, emax=300)

    module_sum = np.sum([counts for counts in counts_modules], axis=0)

    assert np.allclose(time_modules, time_full)
    assert np.allclose(module_sum, counts_full, rtol=0.01)


def test_lightcurve_time_conversion_ijd2loc(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    local = lc.ijd2loc(lc.time[:10])

    assert len(local) == 10
    assert np.all(local >= 0)
    assert np.all(local <= (lc.time[-1] - lc.t0) * 86400)


def test_lightcurve_time_conversion_loc2ijd(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    local = np.array([0, 10, 50])
    ijd = lc.loc2ijd(local)

    assert len(ijd) == 3
    assert np.allclose(ijd, lc.t0 + local / 86400)


def test_lightcurve_time_conversion_roundtrip(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    ijd_original = lc.time[:10]
    local = lc.ijd2loc(ijd_original)
    ijd_back = lc.loc2ijd(local)

    assert np.allclose(ijd_original, ijd_back)


def test_lightcurve_cts_method(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    counts = lc.cts(0, 100, emin=30, emax=300)

    assert isinstance(counts, (float, np.floating))
    assert counts > 0


def test_lightcurve_cts_time_range(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    counts_full = lc.cts(0, 100, emin=30, emax=300)
    counts_half = lc.cts(0, 50, emin=30, emax=300)

    assert counts_half <= counts_full
    assert counts_half > 0


def test_lightcurve_gtis(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert lc.gtis is None


def test_lightcurve_pif_default(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert np.all(lc.weights == 1.0)


def test_lightcurve_invalid_file():
    with pytest.raises(FileNotFoundError):
        LightCurve.load_data(events_path="/nonexistent/file.fits")


def test_lightcurve_rebin_invalid_energy(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    with pytest.raises(ValueError):
        lc.rebin(binsize=1.0, emin=300, emax=30)


def test_lightcurve_rebin_custom_bins(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    custom_bins = [0, 10, 25, 50, 100]
    time, counts = lc.rebin(binsize=custom_bins, emin=30, emax=300)

    assert len(time) == len(custom_bins) - 1
    assert len(counts) == len(custom_bins) - 1
    assert np.all(counts >= 0)


def test_lightcurve_rebin_with_custom_mask(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    custom_mask = np.ones(len(lc.time), dtype=bool)
    custom_mask[: len(lc.time) // 2] = False

    time_masked, counts_masked = lc.rebin(binsize=1.0, emin=30, emax=300, custom_mask=custom_mask)
    time_full, counts_full = lc.rebin(binsize=1.0, emin=30, emax=300)

    assert np.sum(counts_masked) < np.sum(counts_full)


def test_lightcurve_module_assignment():
    module_counts_expected = [10, 25, 50, 75, 100, 125, 150, 200]
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

    module_positions = [
        (16, 32),
        (16, 96),
        (48, 32),
        (48, 96),
        (80, 32),
        (80, 96),
        (116, 32),
        (116, 96),
    ]

    idx = 0
    for module_no, (detz, dety) in enumerate(module_positions):
        n_events_module = module_counts_expected[module_no]

        events["DETZ"][idx : idx + n_events_module] = detz
        events["DETY"][idx : idx + n_events_module] = dety
        events["TIME"][idx : idx + n_events_module] = np.linspace(0, 10 / 86400, n_events_module)
        events["ISGRI_ENERGY"][idx : idx + n_events_module] = 100
        events["SELECT_FLAG"][idx : idx + n_events_module] = 0

        idx += n_events_module

    time = events["TIME"]
    energies = events["ISGRI_ENERGY"]
    dety = events["DETY"]
    detz = events["DETZ"]
    gtis = np.array([[time[0], time[-1]]])
    weights = np.ones(n_events)
    metadata = {}

    lc = LightCurve(time, energies, gtis, dety, detz, weights, metadata)

    times, counts = lc.rebin_by_modules(binsize=20.0, emin=50, emax=200, local_time=True)

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
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts = lc.rebin(binsize=200.0, emin=50, emax=200, local_time=True)
    assert np.isclose(counts.sum(), 100 * 0.8, rtol=0.1)


def test_lightcurve_pif_threshold_override(mock_events_with_pif_coords, mock_pif_file):
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts_low = lc.rebin(binsize=200.0, emin=50, emax=200, pif_threshold=0.3)
    assert np.isclose(counts_low.sum(), 100 * 0.8 + 100 * 0.4, rtol=0.1)

    time, counts_high = lc.rebin(binsize=200.0, emin=50, emax=200, pif_threshold=0.9)
    assert counts_high.sum() == 0


def test_lightcurve_pif_toggle(mock_events_with_pif_coords, mock_pif_file):
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    time, counts_pif = lc.rebin(binsize=200.0, emin=50, emax=200)
    time, counts_no_pif = lc.rebin(binsize=200.0, emin=50, emax=200, use_pif=False)

    assert counts_no_pif.sum() == 300
    assert counts_pif.sum() < counts_no_pif.sum()


def test_lightcurve_pif_instance_settings(mock_events_with_pif_coords, mock_pif_file):
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
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    weights1 = lc._get_weights()
    weights2 = lc._get_weights()
    assert weights1 is weights2

    weights3 = lc._get_weights(pif_threshold=0.3)
    assert weights3 is not weights1


def test_lightcurve_pif_get_method(mock_events_with_pif_coords, mock_pif_file):
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
    lc = LightCurve.load_data(
        events_path=mock_events_with_pif_coords, pif_path=mock_pif_file, use_pif=True, pif_threshold=0.5
    )

    counts_pif = lc.cts(0, 200, emin=50, emax=200, local_time=True)
    counts_no_pif = lc.cts(0, 200, emin=50, emax=200, local_time=True, use_pif=False)

    assert counts_no_pif == 300
    assert counts_pif < counts_no_pif


def test_lightcurve_no_pif_file(mock_events_file):
    lc = LightCurve.load_data(events_path=mock_events_file)

    assert np.all(lc.weights == 1.0)
    assert lc.use_pif == False

    time, counts = lc.rebin(binsize=1.0, emin=30, emax=300)
    assert len(counts) > 0
