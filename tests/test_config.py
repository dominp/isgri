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
def temp_summary(tmp_path):
    """Create temporary summary file."""
    summary = tmp_path / "summary.fits"
    summary.touch()
    return summary


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
    assert cfg.summary_path is None


def test_create_new_config(temp_config_dir, temp_archive, temp_summary):
    """Test creating new config."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.create_new(archive_path=temp_archive, summary_path=temp_summary)

    assert config_path.exists()
    assert cfg.archive_path == temp_archive
    assert cfg.summary_path == temp_summary


def test_set_paths(temp_config_dir, temp_archive, temp_summary):
    """Test setting paths with set() method."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.set(archive_path=temp_archive, summary_path=temp_summary)

    assert cfg.archive_path == temp_archive
    assert cfg.summary_path == temp_summary

    # Reload and verify persistence
    cfg2 = Config(config_path)
    assert cfg2.archive_path == temp_archive
    assert cfg2.summary_path == temp_summary


def test_set_archive_only(temp_config_dir, temp_archive):
    """Test setting only archive path."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.set(archive_path=temp_archive)

    assert cfg.archive_path == temp_archive
    assert cfg.summary_path is None


def test_set_summary_only(temp_config_dir, temp_summary):
    """Test setting only summary path."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    cfg.set(summary_path=temp_summary)

    assert cfg.summary_path == temp_summary
    assert cfg.archive_path is None


def test_summary_path_not_exists(temp_config_dir):
    """Test summary_path raises error if file doesn't exist."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    # Set non-existent path
    cfg.set(summary_path=Path("/tmp/nonexistent.fits"))

    # Should raise when accessing property
    with pytest.raises(FileNotFoundError, match="Summary path does not exist"):
        _ = cfg.summary_path


def test_archive_path_not_exists(temp_config_dir):
    """Test archive_path returns path even if doesn't exist."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)

    nonexistent = Path("/tmp/nonexistent_archive")
    cfg.set(archive_path=nonexistent)

    # Should return path without validation
    assert cfg.archive_path == nonexistent


def test_config_repr(temp_config_dir, temp_archive, temp_summary):
    """Test config string representation."""
    config_path = temp_config_dir / "test.toml"
    cfg = Config(config_path)
    cfg.create_new(archive_path=temp_archive, summary_path=temp_summary)

    repr_str = repr(cfg)
    assert "Config(" in repr_str
    assert str(config_path) in repr_str
    assert str(temp_archive) in repr_str
    assert str(temp_summary) in repr_str



