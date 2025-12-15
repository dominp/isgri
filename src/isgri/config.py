import sys
from pathlib import Path
from typing import Optional
from platformdirs import user_config_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w


class Config:
    """
    Configuration manager for ISGRI.

    Manages paths to archive directory and summary catalog. Config is stored
    in platform-specific location (~/.config/isgri/config.toml on Linux).
    Falls back to local isgri_config.toml if global config doesn't exist.

    Parameters
    ----------
    path : Path, optional
        Custom config file path. If not provided, uses platform default.

    Attributes
    ----------
    path : Path
        Path to config file
    archive_path : Path or None
        Path to INTEGRAL archive directory
    summary_path : Path or None
        Path to summary catalog FITS file (validated on access)
    """

    DEFAULT_PATH = Path(user_config_dir("isgri")) / "config.toml"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH
        self._config = None

    @property
    def config(self) -> dict:
        """
        Load and return config dictionary.

        Returns
        -------
        dict
            Configuration dictionary
        """
        if self._config is not None:
            return self._config

        if self.path.exists():
            path = self.path
        elif self.path == self.DEFAULT_PATH and Path("isgri_config.toml").exists():
            print("Config file not found at default path, using local isgri_config.toml instead.", file=sys.stderr)
            path = Path("isgri_config.toml")
        else:
            self._config = {}
            return self._config

        with open(path, "rb") as f:
            self._config = tomllib.load(f)

        return self._config

    @property
    def archive_path(self) -> Optional[Path]:
        """
        Get archive directory path from config.

        Returns
        -------
        Path or None
            Path to archive directory. No validation performed.
        """
        path_str = self.config.get("archive_path")
        if path_str:
            return Path(path_str)
        return None

    @property
    def summary_path(self) -> Optional[Path]:
        """
        Get summary catalog path from config.

        Returns
        -------
        Path or None
            Path to summary FITS file

        Raises
        ------
        FileNotFoundError
            If configured path doesn't exist
        """
        path_str = self.config.get("summary_path")
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Summary path does not exist: {path}")
        return path

    def save(self):
        """Save current config to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            tomli_w.dump(self._config or {}, f)

    def create_new(self, archive_path: Optional[Path] = None, summary_path: Optional[Path] = None):
        """
        Create new config file with given paths.

        Parameters
        ----------
        archive_path : Path, optional
            Path to archive directory
        summary_path : Path, optional
            Path to summary catalog FITS file
        """
        self._config = {}
        if archive_path:
            self._config["archive_path"] = str(archive_path)
        if summary_path:
            self._config["summary_path"] = str(summary_path)
        self.save()

    def set(self, archive_path: Optional[Path] = None, summary_path: Optional[Path] = None):
        """
        Update config paths and save.

        Parameters
        ----------
        archive_path : Path, optional
            New archive directory path
        summary_path : Path, optional
            New summary catalog path
        """
        if archive_path:
            self.config["archive_path"] = str(archive_path)
        if summary_path:
            self.config["summary_path"] = str(summary_path)

        self.save()

    def __repr__(self):
        return f"Config(path={self.path}, archive={self.archive_path}, summary={self.summary_path})"
