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
    DEFAULT_PATH = Path(user_config_dir("isgri")) / "config.toml"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH
        self._config = None

    @property
    def config(self) -> dict:
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
        path_str = self.config.get("archive_path")
        if path_str:
            return Path(path_str)
        return None

    @property
    def summary_path(self) -> Optional[Path]:
        path_str = self.config.get("summary_path")
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Summary path does not exist: {path}")
        return path

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            tomli_w.dump(self._config or {}, f)

    def create_new(self, archive_path: Optional[Path] = None, summary_path: Optional[Path] = None):
        self._config = {}
        if archive_path:
            self._config["archive_path"] = str(archive_path)
        if summary_path:
            self._config["summary_path"] = str(summary_path)
        self.save()

    def set(self, archive_path: Optional[Path] = None, summary_path: Optional[Path] = None):
        if archive_path:
            self.config["archive_path"] = str(archive_path)
        if summary_path:
            self.config["summary_path"] = str(summary_path)

        self.save()

    def __repr__(self):
        return f"Config(path={self.path}, archive={self.archive_path}, summary={self.summary_path})"
