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

    Manages paths to archive directory and catalog. Config is stored
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
    catalog_path : Path or None
        Path to catalog FITS file (validated on access)
    pif_path : Path or None
        Path to PIF file
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
    def catalog_path(self) -> Optional[Path]:
        """
        Get catalog path from config.

        Returns
        -------
        Path or None
            Path to catalog FITS file

        Raises
        ------
        FileNotFoundError
            If configured path doesn't exist
        """
        path_str = self.config.get("catalog_path")
        if path_str:
            return Path(path_str)
        return None

    @property
    def pif_path(self) -> Optional[Path]:
        """
        Get PIF path from config.

        Returns
        -------
        Path or None
            Path to PIF file
        """
        path_str = self.config.get("pif_path")
        if path_str:
            return Path(path_str)
        return None

    def save(self):
        """Save current config to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            tomli_w.dump(self._config or {}, f)

    def create_new(
        self, archive_path: Optional[Path] = None, catalog_path: Optional[Path] = None, pif_path: Optional[Path] = None
    ):
        """
        Create new config file with given paths.

        Parameters
        ----------
        archive_path : Path, optional
            Path to archive directory
        catalog_path : Path, optional
            Path to catalog FITS file
        pif_path : Path, optional
            Path to PIF file
        """
        self._config = {}
        if archive_path:
            self._config["archive_path"] = str(archive_path)
        if catalog_path:
            self._config["catalog_path"] = str(catalog_path)
        if pif_path:
            self._config["pif_path"] = str(pif_path)
        self.save()

    def set(
        self, archive_path: Optional[Path] = None, catalog_path: Optional[Path] = None, pif_path: Optional[Path] = None
    ):
        """
        Update config paths and save.

        Parameters
        ----------
        archive_path : Path, optional
            New archive directory path
        catalog_path : Path, optional
            New catalog path
        pif_path : Path, optional
            New PIF path
        """
        if archive_path:
            self.config["archive_path"] = str(archive_path)
        if catalog_path:
            self.config["catalog_path"] = str(catalog_path)
        if pif_path:
            self.config["pif_path"] = str(pif_path)

        self.save()

    def __repr__(self):
        return (
            f"Config(path={self.path}, archive={self.archive_path}, catalog={self.catalog_path}, pif={self.pif_path})"
        )
