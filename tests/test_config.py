import pytest
from pathlib import Path
from isgri.config import Config


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def temp_archive(tmp_path):
    """Create temporary archive directory."""
    archive = tmp_path / "archive"
    archive.mkdir()
    return archive


@pytest.fixture
def temp_catalog(tmp_path):
    """Create temporary catalog file."""
    catalog = tmp_path / "catalog.fits"
    catalog.touch()
    return catalog


@pytest.fixture
def temp_pif(tmp_path):
    """Create temporary PIF directory."""
    pif = tmp_path / "pif"
    pif.mkdir()
    return pif


def test_config_init_default():
    """Test default config initialization."""
    cfg = Config()
    assert cfg.path == Config.DEFAULT_PATH
    assert cfg._config is None


def test_config_init_custom_path(temp_config_dir):
    """Test config with custom path."""
    custom_path = temp_config_dir / "custom.toml"
    cfg = Config(custom_path)
    assert cfg.path == custom_path


def test_config_empty():
    """Test empty config returns None for paths."""
    cfg = Config(Path("/tmp/nonexistent_config.toml"))
    assert cfg.archive_path is None
    assert cfg.catalog_path is None
    assert cfg.pif_path is None


def test_create_new_config(temp_config_dir, temp_archive, temp_catalog, temp_pif):
    """Test creating new config."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.create_new(archive_path=temp_archive, catalog_path=temp_catalog, pif_path=temp_pif)

    assert config_path.exists()
    assert cfg.archive_path == temp_archive
    assert cfg.catalog_path == temp_catalog
    assert cfg.pif_path == temp_pif


def test_set_paths(temp_config_dir, temp_archive, temp_catalog, temp_pif):
    """Test setting paths with set() method."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.set(archive_path=temp_archive, catalog_path=temp_catalog, pif_path=temp_pif)

    assert cfg.archive_path == temp_archive
    assert cfg.catalog_path == temp_catalog
    assert cfg.pif_path == temp_pif

    # Reload and verify persistence
    cfg2 = Config(config_path)
    assert cfg2.archive_path == temp_archive
    assert cfg2.catalog_path == temp_catalog
    assert cfg2.pif_path == temp_pif


def test_set_archive_only(temp_config_dir, temp_archive):
    """Test setting only archive path."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.set(archive_path=temp_archive)

    assert cfg.archive_path == temp_archive
    assert cfg.catalog_path is None
    assert cfg.pif_path is None


def test_set_catalog_only(temp_config_dir, temp_catalog):
    """Test setting only catalog path."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.set(catalog_path=temp_catalog)

    assert cfg.catalog_path == temp_catalog
    assert cfg.archive_path is None
    assert cfg.pif_path is None


def test_set_pif_only(temp_config_dir, temp_pif):
    """Test setting only PIF path."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.set(pif_path=temp_pif)

    assert cfg.pif_path == temp_pif
    assert cfg.archive_path is None
    assert cfg.catalog_path is None


def test_archive_path_not_exists(temp_config_dir):
    """Test archive_path returns path even if doesn't exist."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    nonexistent = Path("/tmp/nonexistent_archive")
    cfg.set(archive_path=nonexistent)

    # Should return path without validation
    assert cfg.archive_path == nonexistent


def test_pif_path_not_exists(temp_config_dir):
    """Test PIF path returns path even if doesn't exist."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    nonexistent = Path("/tmp/nonexistent_pif")
    cfg.set(pif_path=nonexistent)

    # Should return path without validation
    assert cfg.pif_path == nonexistent


def test_config_repr(temp_config_dir, temp_archive, temp_catalog, temp_pif):
    """Test config string representation."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)
    cfg.create_new(archive_path=temp_archive, catalog_path=temp_catalog, pif_path=temp_pif)

    repr_str = repr(cfg)
    assert "Config(" in repr_str
    assert str(config_path) in repr_str
    assert str(temp_archive) in repr_str
    assert str(temp_catalog) in repr_str
    assert str(temp_pif) in repr_str