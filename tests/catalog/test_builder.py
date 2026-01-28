import pytest
import numpy as np
from astropy.table import Table
from pathlib import Path
import tempfile
from isgri.catalog.builder import CatalogBuilder, new_catalog_names, new_catalog_dtypes
from isgri.config import Config


# Ensure that the test environment is isolated from user config
@pytest.fixture(autouse=True)
def isolate_config(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "DEFAULT_PATH", tmp_path / "config.toml")
    yield


def create_dummy_event_file(filepath: Path, revol: int, scw_id: str):
    """Create a dummy ISGRI event FITS file with minimal content."""
    from astropy.io import fits

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

    events["TIME"] = np.linspace(0, 100 / 86400, n_events)
    events["ISGRI_ENERGY"] = np.random.uniform(30, 300, n_events)
    events["DETY"] = np.random.randint(0, 128, n_events)
    events["DETZ"] = np.random.randint(0, 134, n_events)
    events["SELECT_FLAG"] = 0

    hdu = fits.BinTableHDU(data=events, name="ISGR-EVTS-ALL")
    hdu.header["REVOL"] = revol
    hdu.header["SWID"] = scw_id.split(".")[0]
    hdu.header["TSTART"] = 0.0
    hdu.header["TSTOP"] = 100.0
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(filepath, overwrite=True)


@pytest.fixture
def temp_archive_dir():
    # Create a temporary directory to act as the archive with dummy SCW files
    # Structure: archive/REVOL/SCW_ID/isgri_events.fits.gz
    # Create some dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "archive"
        archive_path.mkdir()
        # Create dummy revolutions and SCW files
        for rev in range(1000, 1003):
            rev_dir = archive_path / f"{rev:04d}"
            rev_dir.mkdir()
            for scw_num in range(1, 4):
                scw_id = f"{rev:04d}00{scw_num:02d}0010.001"
                scw_dir = rev_dir / scw_id
                scw_dir.mkdir()
                event_file = scw_dir / "isgri_events.fits.gz"
                # Create a fake event file with minimal content
                create_dummy_event_file(event_file, revol=rev, scw_id=scw_id)
        yield archive_path


@pytest.fixture
def temp_catalog_file():
    # Create a temporary catalog file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fits") as f:
        catalog_path = Path(f.name)
        rows = [
            (150, "015000010010", 5000.1, 1000, 5000.2, 10, 10, 10, 10, 150, 1, 1, 1),
            (151, "015000020010", 6000.1, 1100, 6000.2, 20, 20, 20, 20, 160, 1, 1, 1),
        ]
        table = Table(rows=rows, names=new_catalog_names, dtype=new_catalog_dtypes)
        table.write(catalog_path, format="fits", overwrite=True)
        yield catalog_path
    Path(catalog_path).unlink()


@pytest.fixture
def temp_dir_with_no_catalog():
    # Create a temporary directory without a catalog file
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_catalog_path = Path(tmpdir) / "no_catalog"
        dir_catalog_path.mkdir()
        yield dir_catalog_path / "catalog.fits"


@pytest.fixture
def temp_arr_dir():
    # Create a temporary directory to act as lightcurve cache
    with tempfile.TemporaryDirectory() as tmpdir:
        arr_cache_path = Path(tmpdir) / "array_cache"
        arr_cache_path.mkdir()
        yield arr_cache_path


def test_initialize_builder(temp_archive_dir, temp_catalog_file):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        n_cores=2,
    )
    assert builder.archive_path == temp_archive_dir
    assert builder.catalog_path == temp_catalog_file
    assert builder.n_cores == 2
    assert isinstance(builder.catalog, Table)
    assert len(builder.catalog) == 2


def test_initialize_builder_no_catalog(temp_archive_dir, temp_dir_with_no_catalog):
    #
    with pytest.raises(FileNotFoundError):
        builder = CatalogBuilder(
            archive_path=temp_archive_dir,
            catalog_path=None,
            n_cores=2,
        )

    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_dir_with_no_catalog,
        n_cores=2,
    )
    assert builder.archive_path == temp_archive_dir
    assert builder.catalog_path == temp_dir_with_no_catalog
    assert isinstance(builder.catalog, Table)
    assert len(builder.catalog) == 0


def test_find_scws(temp_archive_dir, temp_catalog_file):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        n_cores=2,
    )
    scw_list, scw_paths = builder.find_scws()
    assert len(scw_list) == 9  # 3 revolutions * 3 SCWs each
    assert len(scw_paths) == 9
    for path in scw_paths:
        assert Path(path).exists()
    for scw, path in zip(scw_list, scw_paths):
        assert scw in Path(path).name


def test_find_event_files(temp_archive_dir, temp_catalog_file):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        n_cores=2,
    )
    scw_list, scw_paths = builder.find_scws()
    valid_swids, valid_paths = builder.find_event_files(scw_list, scw_paths)
    assert len(valid_swids) == 9  # All SCWs have event files
    assert len(valid_paths) == 9
    for path in valid_paths:
        assert Path(path).exists()
    for swid, path in zip(valid_swids, valid_paths):
        assert swid in Path(path).parent.name


def assertation_table_data(table_data, scw):
    assert table_data["REVOL"] == int(scw[:4])
    assert table_data["SWID"] == scw
    assert "TSTART" in table_data
    assert "TSTOP" in table_data
    assert "ONTIME" in table_data
    assert "RA_SCX" in table_data
    assert "DEC_SCX" in table_data
    assert "RA_SCZ" in table_data
    assert "DEC_SCZ" in table_data
    assert "NoEVTS" in table_data
    assert "CHI" in table_data
    assert "CUT_CHI" in table_data
    assert "GTI_CHI" in table_data


def test_process_scw(temp_archive_dir, temp_catalog_file):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        n_cores=2,
    )
    scw_list, scw_paths = builder.find_scws()
    for scw, path in zip(scw_list, scw_paths):
        table_data, array_data = builder._process_scw(path)
        assertation_table_data(table_data, scw)
        assert array_data["SWID"] == scw
        assert len(array_data["TIME"]) > 0
        assert len(array_data["COUNTS"]) > 0
        assert array_data["MODULE_COUNTS"].shape[0] == 8
        assert "GTIS" in array_data


def test_process_rev(temp_archive_dir, temp_catalog_file):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        n_cores=2,
    )
    scw_list, scw_paths = builder.find_scws()
    revols = set(scw[:4] for scw in scw_list)
    for revol in revols:
        rev_paths = [path for scw, path in zip(scw_list, scw_paths) if scw.startswith(revol)]
        table_data_list, array_data = builder._process_rev(rev_paths)
        assert len(table_data_list) == len(rev_paths)
        assert array_data.shape[0] == len(rev_paths)
        for table_data, path in zip(table_data_list, rev_paths):
            scw = Path(path).name.split(".")[0]
            assertation_table_data(table_data, scw)


def test_add_table_data(temp_archive_dir, temp_catalog_file):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        n_cores=2,
    )
    scw_list, scw_paths = builder.find_scws()
    revols = set(scw[:4] for scw in scw_list)
    for revol in revols:
        rev_paths = [path for scw, path in zip(scw_list, scw_paths) if scw.startswith(revol)]
        table_data_list, array_data = builder._process_rev(rev_paths)
        initial_len = len(builder.catalog)
        builder._add_catalog_data(table_data_list)
        assert len(builder.catalog) == initial_len + len(table_data_list)
        saved_catalog = Table.read(builder.catalog_path)
        assert len(saved_catalog) == len(builder.catalog)


def test_save_array_data(temp_archive_dir, temp_catalog_file, temp_arr_dir):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        lightcurve_cache=temp_arr_dir,
        n_cores=2,
    )
    arr_data1 = (
        {"SWID": "015000010010", "TIME": np.array([0.0, 1.0])},
        {"SWID": "015000020010", "TIME": np.array([0.0, 1.0])},
    )
    dtype = [("SWID", "U16"), ("TIME", "O")]
    array_data1 = np.empty(len(arr_data1), dtype=dtype)
    for i, d in enumerate(arr_data1):
        array_data1[i] = (d["SWID"], d["TIME"])

    builder._add_array_data("1500", array_data1)
    arr_file1 = temp_arr_dir / "1500.npy"
    assert arr_file1.exists()
    loaded_data1 = np.load(arr_file1, allow_pickle=True)
    assert len(loaded_data1) == 2
    assert loaded_data1[0]["SWID"] == "015000010010"
    assert np.array_equal(loaded_data1[0]["TIME"], np.array([0.0, 1.0]))

    arr_data2 = ({"SWID": "015000030010", "TIME": np.array([0.0, 1.0])},)
    array_data2 = np.empty(len(arr_data2), dtype=dtype)
    for i, d in enumerate(arr_data2):
        array_data2[i] = (d["SWID"], d["TIME"])
    builder._add_array_data("1500", array_data2)
    loaded_data2 = np.load(arr_file1, allow_pickle=True)
    assert len(loaded_data2) == 3
    assert loaded_data2[2]["SWID"] == "015000030010"
    assert np.array_equal(loaded_data2[2]["TIME"], np.array([0.0, 1.0]))


def test_update_catalog(temp_archive_dir, temp_catalog_file, temp_arr_dir):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        lightcurve_cache=temp_arr_dir,
        n_cores=2,
    )
    initial_catalog_len = len(builder.catalog)
    builder.update_catalog()
    updated_catalog = Table.read(builder.catalog_path)
    assert len(updated_catalog) >= initial_catalog_len
    scw_list, _ = builder.find_scws()
    for scw in scw_list:
        assert scw in updated_catalog["SWID"]
    revols = set(scw[:4] for scw in scw_list)
    for revol in revols:
        arr_file = temp_arr_dir / f"{revol}.npy"
        assert arr_file.exists()
        arr_data = np.load(arr_file, allow_pickle=True)
        swids_in_arr = [entry["SWID"] for entry in arr_data]
        swids_in_catalog = updated_catalog["SWID"][updated_catalog["REVOL"] == int(revol)].tolist()
        for swid in swids_in_catalog:
            assert swid in swids_in_arr


def test_update_catalog_no_new_scws(temp_archive_dir, temp_catalog_file, temp_arr_dir):
    builder = CatalogBuilder(
        archive_path=temp_archive_dir,
        catalog_path=temp_catalog_file,
        lightcurve_cache=temp_arr_dir,
        n_cores=2,
    )
    initial_catalog_len = len(builder.catalog)
    builder.update_catalog()  # First update
    catalog_len_after_first_update = len(builder.catalog)
    assert catalog_len_after_first_update >= initial_catalog_len
    builder.update_catalog()
    catalog_len_after_second_update = len(builder.catalog)
    assert catalog_len_after_second_update == catalog_len_after_first_update
