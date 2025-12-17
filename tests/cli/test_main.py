import pytest
from click.testing import CliRunner
from pathlib import Path
from isgri.cli import main
from isgri.config import Config
from astropy.table import Table
import numpy as np


@pytest.fixture
def runner():
    """Create Click CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_catalog(tmp_path):
    """Create mock SCW catalog."""
    catalog_path = tmp_path / "test_catalog.fits"

    n_scw = 100
    data = {
        "SWID": [f"{i:012d}" for i in range(n_scw)],
        "REVOL": np.random.randint(100, 500, n_scw),
        "TSTART": np.linspace(3000, 4000, n_scw),
        "TSTOP": np.linspace(3000.5, 4000.5, n_scw),
        "RA_SCX": np.random.uniform(0, 360, n_scw),
        "DEC_SCX": np.random.uniform(-80, 80, n_scw),
        "RA_SCZ": np.random.uniform(0, 360, n_scw),
        "DEC_SCZ": np.random.uniform(-80, 80, n_scw),
        "CHI": np.random.uniform(0.5, 5.0, n_scw),
        "CUT_CHI": np.random.uniform(0.5, 5.0, n_scw),
        "GTI_CHI": np.random.uniform(0.5, 5.0, n_scw),
    }
    table = Table(data)
    table.write(catalog_path, format="fits", overwrite=True)

    return catalog_path


def test_cli_version(runner):
    """Test --version flag."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_cli_help(runner):
    """Test --help flag."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "ISGRI" in result.output


def test_query_with_catalog(runner, mock_catalog):
    """Test query command with explicit catalog."""
    result = runner.invoke(main, ["query", "--catalog", str(mock_catalog), "--tstart", "3000", "--tstop", "4000"])
    assert result.exit_code == 0
    assert "Found" in result.output
    assert "100" in result.output


def test_query_count(runner, mock_catalog):
    """Test query --count flag."""
    result = runner.invoke(main, ["query", "--catalog", str(mock_catalog),"--tstart", "3000", "--count"])
    assert result.exit_code == 0
    assert result.output.strip() == "100"


def test_query_list_swids(runner, mock_catalog):
    """Test query --list-swids flag."""
    result = runner.invoke(main, ["query", "--catalog", str(mock_catalog), "--tstart", "3000","--list-swids"])
    assert result.exit_code == 0
    lines = result.output.strip().split("\n")
    assert len(lines) == 100
    assert all(len(line) == 12 for line in lines)


def test_query_with_filters(runner, mock_catalog):
    """Test query with time filters."""
    result = runner.invoke(main, ["query", "--catalog", str(mock_catalog), "--tstart", "3000", "--tstop", "3100"])
    assert result.exit_code == 0
    assert "Found" in result.output


def test_query_output_fits(runner, mock_catalog, tmp_path):
    """Test query with FITS output."""
    output_file = tmp_path / "results.fits"
    result = runner.invoke(main, ["query", "--catalog", str(mock_catalog),"--tstart", "3000", "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()
    assert "Saved" in result.output


def test_query_output_csv(runner, mock_catalog, tmp_path):
    """Test query with CSV output."""
    output_file = tmp_path / "results.csv"
    result = runner.invoke(main, ["query", "--catalog", str(mock_catalog),"--tstart", "3000", "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()


def test_query_no_catalog_no_config(runner, tmp_path, monkeypatch):
    """Test query fails when no catalog and no config."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    result = runner.invoke(main, ["query"])
    assert result.exit_code != 0
    assert "No catalog configured" in result.output


def test_query_uses_config(runner, mock_catalog, tmp_path, monkeypatch):
    """Test query uses catalog from config."""
    config_path = tmp_path / "config.toml"
    cfg = Config(config_path)
    cfg.set(catalog_path=mock_catalog)

    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    result = runner.invoke(main, ["query","--tstart", "3000", ])
    assert result.exit_code == 0
    assert "Found 100" in result.output


def test_config_show_empty(runner, tmp_path, monkeypatch):
    """Test config command with empty config."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    assert str(config_path) in result.output
    assert "(not set)" in result.output


def test_config_show_with_values(runner, tmp_path, mock_catalog, monkeypatch):
    """Test config command with configured values."""
    config_path = tmp_path / "config.toml"
    cfg = Config(config_path)
    cfg.set(archive_path=tmp_path, catalog_path=mock_catalog)

    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    assert str(tmp_path) in result.output
    assert str(mock_catalog) in result.output


def test_config_set_archive(runner, tmp_path, monkeypatch):
    """Test config-set --archive."""
    config_path = tmp_path / "test_config.toml"
    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    archive_path = tmp_path / "archive"
    archive_path.mkdir()

    result = runner.invoke(main, ["config-set", "--archive", str(archive_path)])
    assert result.exit_code == 0
    assert "Archive path set" in result.output

    cfg = Config(config_path)
    assert cfg.archive_path == archive_path


def test_config_set_catalog(runner, tmp_path, mock_catalog, monkeypatch):
    """Test config-set --catalog."""
    config_path = tmp_path / "test_config.toml"
    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    result = runner.invoke(main, ["config-set", "--catalog", str(mock_catalog)])
    assert result.exit_code == 0
    assert "Catalog path set" in result.output

    cfg = Config(config_path)
    assert str(cfg.catalog_path) == str(mock_catalog)


def test_config_set_both(runner, tmp_path, mock_catalog, monkeypatch):
    """Test config-set with both options."""
    config_path = tmp_path / "test_config.toml"
    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    archive_path = tmp_path / "archive"
    archive_path.mkdir()

    result = runner.invoke(main, ["config-set", "--archive", str(archive_path), "--catalog", str(mock_catalog)])
    assert result.exit_code == 0
    assert "Archive path set" in result.output
    assert "Catalog path set" in result.output


def test_config_set_no_options(runner):
    """Test config-set with no options fails."""
    result = runner.invoke(main, ["config-set"])
    assert result.exit_code != 0
    assert "Specify at least one option" in result.output


def test_config_set_nonexistent_path_abort(runner, tmp_path, monkeypatch):
    """Test config-set with non-existent path (user aborts)."""
    config_path = tmp_path / "test_config.toml"
    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    result = runner.invoke(
        main, ["config-set", "--archive", "/nonexistent/path"], input="n\n"
    )  # User says 'no' to confirmation

    assert result.exit_code != 0
    assert "Warning" in result.output


def test_config_set_nonexistent_path_confirm(runner, tmp_path, monkeypatch):
    """Test config-set with non-existent path (user confirms)."""
    config_path = tmp_path / "test_config.toml"
    monkeypatch.setattr(Config, "DEFAULT_PATH", config_path)

    result = runner.invoke(
        main, ["config-set", "--archive", "/nonexistent/path"], input="y\n"
    )  # User says 'yes' to confirmation

    assert result.exit_code == 0
    assert "Archive path set" in result.output
