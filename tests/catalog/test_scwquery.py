import pytest
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from pathlib import Path
import tempfile
from isgri.catalog.scwquery import ScwQuery, Filter


@pytest.fixture
def mock_catalog():
    """Create mock SCW catalog"""
    n_scw = 1000

    catalog = Table(
        {
            "SWID": [f"{i:012d}" for i in range(n_scw)],
            "REVOL": np.random.randint(100, 500, n_scw),
            "TSTART": np.linspace(3000, 4000, n_scw),
            "TSTOP": np.linspace(3001, 4001, n_scw),
            "RA_SCX": np.random.uniform(0, 360, n_scw),
            "DEC_SCX": np.random.uniform(-80, 80, n_scw),
            "RA_SCZ": np.random.uniform(0, 360, n_scw),
            "DEC_SCZ": np.random.uniform(-80, 80, n_scw),
            "CHI": np.random.uniform(0.5, 5.0, n_scw),
            "CUT_CHI": np.random.uniform(0.5, 5.0, n_scw),
            "GTI_CHI": np.random.uniform(0.5, 5.0, n_scw),
        }
    )

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fits") as f:
        catalog.write(f.name, overwrite=True)
        yield f.name

    Path(f.name).unlink()


def test_ScwQuery_init(mock_catalog):
    """Test ScwQuery initialization"""
    query = ScwQuery(mock_catalog)

    assert query.catalog_path.exists()
    assert query._catalog is None
    assert query._mask is None
    assert query._filters == []


def test_ScwQuery_lazy_loading(mock_catalog):
    """Test catalog lazy loading"""
    query = ScwQuery(mock_catalog)

    assert query._catalog is None
    catalog = query.catalog
    assert query._catalog is not None
    assert len(catalog) == 1000


def test_ScwQuery_invalid_path():
    """Test initialization with invalid path"""
    query = ScwQuery("/nonexistent/catalog.fits")

    with pytest.raises(FileNotFoundError):
        _ = query.catalog


def test_ScwQuery_missing_columns():
    """Test validation with missing columns"""
    bad_catalog = Table({"SWID": ["test"]})

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fits") as f:
        bad_catalog.write(f.name, overwrite=True)
        query = ScwQuery(f.name)

        with pytest.raises(ValueError, match="missing required columns"):
            _ = query.catalog

        Path(f.name).unlink()


def test_time_filter_tstart(mock_catalog):
    """Test time filtering with tstart"""
    query = ScwQuery(mock_catalog)

    results = query.time(tstart=3500).get()

    assert len(results) < 1000
    assert all(results["TSTOP"] >= 3500)


def test_time_filter_tstop(mock_catalog):
    """Test time filtering with tstop"""
    query = ScwQuery(mock_catalog)

    results = query.time(tstop=3500).get()

    assert len(results) < 1000
    assert all(results["TSTART"] <= 3500)


def test_time_filter_range(mock_catalog):
    """Test time filtering with both tstart and tstop"""
    query = ScwQuery(mock_catalog)

    results = query.time(tstart=3400, tstop=3600).get()

    assert len(results) < 1000
    assert all(results["TSTOP"] >= 3400)
    assert all(results["TSTART"] <= 3600)


def test_time_filter_string_format(mock_catalog):
    """Test time filtering with ISO string"""
    query = ScwQuery(mock_catalog)

    results = query.time(tstart="2008-01-01T00:00:00").get()

    assert len(results) >= 0


def test_quality_filter_max_chi(mock_catalog):
    """Test quality filtering"""
    query = ScwQuery(mock_catalog)

    results = query.quality(max_chi=2.0).get()

    assert len(results) < 1000
    assert all(results["CHI"] <= 2.0)


def test_quality_filter_different_column(mock_catalog):
    """Test quality filtering with different chi column"""
    query = ScwQuery(mock_catalog)

    results = query.quality(max_chi=2.0, chi_type="CUT_CHI").get()

    assert all(results["CUT_CHI"] <= 2.0)


def test_quality_filter_invalid_column(mock_catalog):
    """Test quality filtering with invalid column"""
    query = ScwQuery(mock_catalog)

    with pytest.raises(ValueError, match="not found"):
        query.quality(chi_type="NONEXISTENT")


def test_quality_filter_negative_chi(mock_catalog):
    """Test quality filtering rejects negative chi"""
    query = ScwQuery(mock_catalog)

    with pytest.raises(ValueError, match="must be positive"):
        query.quality(max_chi=-1.0)


def test_position_filter_radius(mock_catalog):
    """Test position filtering with radius"""
    query = ScwQuery(mock_catalog)

    results = query.position(ra=100.0, dec=30.0, radius=10.0).get()

    assert len(results) >= 0


def test_position_filter_fov_full(mock_catalog):
    """Test position filtering with fully coded FOV"""
    query = ScwQuery(mock_catalog)

    results = query.position(ra=100.0, dec=30.0, fov_mode="full").get()

    assert len(results) >= 0


def test_position_filter_fov_any(mock_catalog):
    """Test position filtering with any FOV"""
    query = ScwQuery(mock_catalog)

    results = query.position(ra=100.0, dec=30.0, fov_mode="any").get()

    assert len(results) >= 0


def test_position_filter_max_offset(mock_catalog):
    """Test position filtering with custom max offset"""
    query = ScwQuery(mock_catalog)

    results = query.position(ra=100.0, dec=30.0, max_offset=5.0).get()

    assert len(results) >= 0


def test_position_filter_skycoord(mock_catalog):
    """Test position filtering with SkyCoord"""
    query = ScwQuery(mock_catalog)
    target = SkyCoord(100.0, 30.0, unit="deg")

    results = query.position(target=target, radius=10.0).get()

    assert len(results) >= 0


def test_position_filter_string_coordinates(mock_catalog):
    """Test position filtering with HMS/DMS strings"""
    query = ScwQuery(mock_catalog)

    results = query.position(ra="06h40m00s", dec="+30d00m00s", radius=10.0).get()

    assert len(results) >= 0


def test_position_filter_invalid_radius(mock_catalog):
    """Test position filtering rejects negative radius"""
    query = ScwQuery(mock_catalog)

    with pytest.raises(ValueError, match="must be positive"):
        query.position(ra=100.0, dec=30.0, radius=-5.0)


def test_revolution_filter_single_int(mock_catalog):
    """Test revolution filtering with single integer"""
    query = ScwQuery(mock_catalog)

    results = query.revolution(200).get()

    assert all(results["REVOL"] == 200)


def test_revolution_filter_single_string(mock_catalog):
    """Test revolution filtering with single string"""
    query = ScwQuery(mock_catalog)

    results = query.revolution("0200").get()

    assert all(results["REVOL"] == 200)


def test_revolution_filter_multiple(mock_catalog):
    """Test revolution filtering with multiple revolutions"""
    query = ScwQuery(mock_catalog)

    results = query.revolution([200, 300]).get()

    assert all((results["REVOL"] == 200) | (results["REVOL"] == 300))


def test_revolution_filter_mixed_types(mock_catalog):
    """Test revolution filtering with mixed int and string"""
    query = ScwQuery(mock_catalog)

    results = query.revolution([200, "0300"]).get()

    assert all((results["REVOL"] == 200) | (results["REVOL"] == 300))


def test_revolution_filter_invalid_string(mock_catalog):
    """Test revolution filtering rejects invalid string format"""
    query = ScwQuery(mock_catalog)

    with pytest.raises(ValueError, match="must be 4 digits"):
        query.revolution("200")


def test_chained_filters(mock_catalog):
    """Test multiple chained filters"""
    query = ScwQuery(mock_catalog)

    results = query.time(tstart=3400, tstop=3600).quality(max_chi=2.0).position(ra=100.0, dec=30.0, radius=20.0).get()

    assert len(results) >= 0
    assert all(results["TSTOP"] >= 3400)
    assert all(results["TSTART"] <= 3600)
    assert all(results["CHI"] <= 2.0)


def test_reset_filters(mock_catalog):
    """Test reset clears all filters"""
    query = ScwQuery(mock_catalog)

    query.time(tstart=3500).quality(max_chi=2.0)
    assert len(query._filters) == 2

    query.reset()
    assert len(query._filters) == 0
    assert len(query.get()) == 1000


def test_count_method(mock_catalog):
    """Test count method"""
    query = ScwQuery(mock_catalog)

    count_all = query.count()
    assert count_all == 1000

    count_filtered = query.time(tstart=3500).count()
    assert count_filtered < 1000


def test_filter_replacement(mock_catalog):
    """Test that applying same filter twice replaces the first"""
    query = ScwQuery(mock_catalog)

    query.time(tstart=3400).time(tstart=3600)

    assert len(query._filters) == 1
    results = query.get()
    assert all(results["TSTOP"] >= 3600)


def test_get_offsets(mock_catalog):
    """Test get_offsets adds offset columns"""
    query = ScwQuery(mock_catalog)

    results = query.time(tstart=3400, tstop=3600).get_offsets(ra=100.0, dec=30.0)

    assert "Y_OFFSET" in results.colnames
    assert "Z_OFFSET" in results.colnames
    assert "MAX_OFFSET" in results.colnames
    assert len(results["Y_OFFSET"]) == len(results)


def test_filters_summary(mock_catalog):
    """Test filters_summary property"""
    query = ScwQuery(mock_catalog)

    query.time(tstart=3500).quality(max_chi=2.0).revolution(200)

    summary = query.filters_summary
    assert "time" in summary
    assert "quality" in summary
    assert "revolution" in summary
    assert summary["time"]["tstart"] == 3500
    assert summary["quality"]["max_chi"] == 2.0


def test_repr(mock_catalog):
    """Test string representation"""
    query = ScwQuery(mock_catalog)

    query.time(tstart=3500).quality(max_chi=2.0)
    repr_str = repr(query)

    assert "ScwQuery" in repr_str
    assert "total=100" in repr_str
    assert "time" in repr_str
    assert "quality" in repr_str


def test_empty_result(mock_catalog):
    """Test query with no matches"""
    query = ScwQuery(mock_catalog)

    results = query.time(tstart=10000).get()

    assert len(results) == 0


def test_filter_dataclass():
    """Test Filter dataclass"""
    mask = np.array([True, False, True])
    params = {"test": 123}

    filt = Filter(name="test", mask=mask, params=params)

    assert filt.name == "test"
    assert np.array_equal(filt.mask, mask)
    assert filt.params == params
