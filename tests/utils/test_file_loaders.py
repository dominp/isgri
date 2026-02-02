import pytest
import numpy as np
from astropy.io import fits
import tempfile
import os
import shutil
from pathlib import Path
from isgri.utils.file_loaders import load_isgri_events, resolve_event_path, resolve_pif_path, verify_events_file
from isgri.config import Config


@pytest.fixture
def mock_events_file():
    """Create a minimal mock ISGRI events FITS file."""
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
    """Create mock archive directory structure."""
    revol_dir = tmp_path / "1000"
    revol_dir.mkdir()

    swid_dir = revol_dir / "100000100010.001"
    swid_dir.mkdir()

    events_file = swid_dir / "isgri_events.fits.gz"
    shutil.copy(mock_events_file, events_file)

    return tmp_path, revol_dir, swid_dir, events_file


@pytest.fixture
def mock_archive_multiple_versions(tmp_path, mock_events_file):
    """Create archive with multiple SCW versions."""
    revol_dir = tmp_path / "1000"
    revol_dir.mkdir()

    for version in ["001", "002"]:
        swid_dir = revol_dir / f"100000100010.{version}"
        swid_dir.mkdir()
        events_file = swid_dir / "isgri_events.fits.gz"
        shutil.copy(mock_events_file, events_file)

    return tmp_path, revol_dir


@pytest.fixture
def mock_config(tmp_path):
    """Create mock config file."""
    config_file = tmp_path / "test_config.toml"
    return Config(path=config_file)


@pytest.fixture
def mock_pif_file(tmp_path):
    """Create mock PIF file."""
    pif_file = tmp_path / "100000100010_pif.fits"
    hdu = fits.PrimaryHDU(data=np.ones((134, 130)))
    hdul = fits.HDUList([hdu])
    hdul.writeto(pif_file)
    return pif_file


@pytest.fixture
def mock_pif_structure(tmp_path, mock_pif_file):
    """Create mock PIF directory structure."""
    source_dir = tmp_path / "Crab"
    source_dir.mkdir()

    revol_dir = source_dir / "1000"
    revol_dir.mkdir()

    pif_file = revol_dir / "100000100010_pif.fits"
    shutil.copy(mock_pif_file, pif_file)

    return tmp_path, source_dir, revol_dir, pif_file


def test_verify_events_file_valid(mock_events_file):
    result = verify_events_file(mock_events_file)
    assert result == str(mock_events_file)


def test_verify_events_file_missing():
    with pytest.raises(Exception):
        verify_events_file("/nonexistent/file.fits")


def test_verify_events_file_invalid_extension(tmp_path):
    invalid_file = tmp_path / "invalid.fits"
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.writeto(invalid_file)

    with pytest.raises(ValueError, match="ISGR-EVTS-ALL extension not found"):
        verify_events_file(invalid_file)


def test_resolve_event_path_direct_file(mock_events_file):
    path = resolve_event_path(mock_events_file)
    assert path == str(mock_events_file)


def test_resolve_event_path_invalid():
    with pytest.raises(FileNotFoundError):
        resolve_event_path("/nonexistent/path.fits")


def test_resolve_event_path_scw_directory(mock_archive_structure):
    tmp_path, revol_dir, swid_dir, events_file = mock_archive_structure

    path = resolve_event_path(str(swid_dir))
    assert path == str(events_file)


def test_resolve_event_path_no_events_in_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No isgri_events file found"):
        resolve_event_path(str(empty_dir))


def test_resolve_event_path_multiple_events(tmp_path, mock_events_file):
    file1 = tmp_path / "isgri_events_1.fits"
    file2 = tmp_path / "isgri_events_2.fits"
    shutil.copy(mock_events_file, file1)
    shutil.copy(mock_events_file, file2)

    with pytest.raises(FileNotFoundError, match="Multiple isgri_events files found"):
        resolve_event_path(str(tmp_path))


def test_resolve_event_path_with_swid(mock_archive_structure, mock_config):
    tmp_path, revol_dir, swid_dir, events_file = mock_archive_structure

    mock_config.create_new(archive_path=tmp_path)

    path = resolve_event_path(swid="100000100010", config=mock_config)
    assert path == str(events_file)


def test_resolve_event_path_swid_no_config():
    with pytest.raises(ValueError, match="Either events_path must be provided"):
        resolve_event_path(swid="100000100010")


def test_resolve_event_path_config_no_archive_path(mock_config):
    with pytest.raises(ValueError, match="archive_path must be defined"):
        resolve_event_path(swid="100000100010", config=mock_config)


def test_resolve_event_path_multiple_versions(mock_archive_multiple_versions, mock_config):
    tmp_path, revol_dir = mock_archive_multiple_versions

    mock_config.create_new(archive_path=tmp_path)

    with pytest.raises(FileNotFoundError, match="Multiple directories found"):
        resolve_event_path(swid="100000100010", config=mock_config)


def test_resolve_event_path_no_matching_swid(tmp_path, mock_config):
    revol_dir = tmp_path / "1000"
    revol_dir.mkdir()

    mock_config.create_new(archive_path=tmp_path)

    with pytest.raises(FileNotFoundError, match="No directory found for SWID"):
        resolve_event_path(swid="100000100010", config=mock_config)


def test_resolve_event_path_invalid_archive_path(mock_config):
    mock_config.create_new(archive_path=Path("/nonexistent/archive"))

    with pytest.raises(FileNotFoundError, match="Archive path from config is not a directory"):
        resolve_event_path(swid="100000100010", config=mock_config)


def test_load_isgri_events(mock_events_file):
    events, gtis, metadata = load_isgri_events(mock_events_file)

    assert len(events) > 0
    assert "TIME" in events.dtype.names
    assert "ISGRI_ENERGY" in events.dtype.names
    assert "DETY" in events.dtype.names
    assert "DETZ" in events.dtype.names

    assert metadata["REVOL"] == 1000
    assert metadata["SWID"] == "100000100010"
    assert metadata["TSTART"] == 0.0
    assert metadata["TSTOP"] == 100.0
    assert metadata["NoEVTS"] == 1000

    assert gtis is None


def test_load_isgri_events_filters_bad_events():
    n_events = 100

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

    events["SELECT_FLAG"] = 0
    events["SELECT_FLAG"][::2] = 1

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fits") as f:
        hdu = fits.BinTableHDU(data=events, name="ISGR-EVTS-ALL")
        hdu.header["NAXIS2"] = n_events
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(f.name, overwrite=True)
        filepath = f.name

    try:
        loaded_events, _, _ = load_isgri_events(filepath)
        assert len(loaded_events) == n_events // 2
    finally:
        os.unlink(filepath)


def test_resolve_pif_path_direct_file(mock_pif_file):
    path = resolve_pif_path(pif_path=str(mock_pif_file))
    assert path == str(mock_pif_file)


def test_resolve_pif_path_invalid():
    with pytest.raises(FileNotFoundError):
        resolve_pif_path(pif_path="/nonexistent/path.fits")


def test_resolve_pif_path_with_source(mock_pif_structure, mock_config):
    tmp_path, source_dir, revol_dir, pif_file = mock_pif_structure

    mock_config.create_new(pif_path=tmp_path)

    path = resolve_pif_path(source="Crab", swid="100000100010", config=mock_config)
    assert path == str(pif_file)


def test_resolve_pif_path_no_source(mock_pif_structure, mock_config):
    tmp_path, source_dir, revol_dir, pif_file = mock_pif_structure

    mock_config.create_new(pif_path=source_dir)

    path = resolve_pif_path(swid="100000100010", config=mock_config)
    assert path == str(pif_file)


def test_resolve_pif_path_no_swid(tmp_path, mock_config):
    mock_config.create_new(pif_path=tmp_path)

    with pytest.raises(ValueError, match="swid must be provided"):
        resolve_pif_path(config=mock_config)


def test_resolve_pif_path_no_config_no_path():
    with pytest.raises(ValueError, match="Either pif_path must be provided"):
        resolve_pif_path(swid="100000100010")


def test_resolve_pif_path_multiple_files(tmp_path, mock_pif_file, mock_config):
    revol_dir = tmp_path / "1000"
    revol_dir.mkdir()

    pif1 = revol_dir / "100000100010_pif1.fits"
    pif2 = revol_dir / "100000100010_pif2.fits"
    shutil.copy(mock_pif_file, pif1)
    shutil.copy(mock_pif_file, pif2)

    mock_config.create_new(pif_path=tmp_path)

    with pytest.raises(FileNotFoundError, match="Multiple PIF files found"):
        resolve_pif_path(swid="100000100010", config=mock_config)


def test_resolve_pif_path_no_matching_file(tmp_path, mock_config):
    revol_dir = tmp_path / "1000"
    revol_dir.mkdir()

    mock_config.create_new(pif_path=tmp_path)

    with pytest.raises(FileNotFoundError, match="No PIF file found"):
        resolve_pif_path(swid="100000100010", config=mock_config)


def test_resolve_pif_path_invalid_pif_path(mock_config):
    mock_config.create_new(pif_path=Path("/nonexistent/pif"))

    with pytest.raises(FileNotFoundError, match="PIF path from config is not a directory"):
        resolve_pif_path(swid="100000100010", config=mock_config)
