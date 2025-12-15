# ISGRI

Python toolkit for INTEGRAL/ISGRI data analysis.

## Features

### Command Line Interface
Query catalogs directly from the terminal:
- Filter by time, position, quality, revolution
- Export results to FITS/CSV
- List SWIDs for batch processing

### SCW Catalog Query
Query INTEGRAL Science Window catalogs with a fluent Python API:
- Filter by time, position, quality, revolution
- Calculate detector offsets
- Export results to FITS/CSV

### Light Curve Analysis
Extract and analyze ISGRI light curves:
- Event loading with PIF weighting
- Custom time binning
- Module-by-module analysis
- Quality metrics (chi-squared tests)
- Time conversions (IJD to/from UTC)


## Installation

```bash
pip install isgri
```

## Quick Start

### CLI Usage

```bash
# Configure default paths (once)
isgri config-set --catalog ~/data/scw_catalog.fits

# Query by time range
isgri query --tstart 2010-01-01 --tstop 2010-12-31

# Query Crab with quality cut
isgri query --ra 83.63 --dec 22.01 --max-chi 2.0 --fov full

# Get list of SWIDs for processing
isgri query --tstart 3000 --tstop 3100 --list-swids > swids.txt

# Export results
isgri query --tstart 3000 --tstop 3100 --output results.fits
```

### Query SCW Catalog

```python
from isgri.catalog import ScwQuery

# Load catalog
cat = ScwQuery("path_to_catalog.fits")

# Find Crab observations in 2010 with good quality
results = (cat
    .time(tstart="2010-01-01", tstop="2010-12-31")
    .quality(max_chi=2.0)
    .position(ra=83.63, dec=22.01, fov_mode="full")
    .get()
)

print(f"Found {len(results)} observations")
```

### Analyze Light Curves

```python
from isgri.utils import LightCurve, QualityMetrics

# Load events with PIF weighting
lc = LightCurve.load_data(
    events_path="isgri_events.fits",
    pif_path="source_model.fits",
    pif_threshold=0.5
)

# Create 1-second binned light curve
time, counts = lc.rebin(binsize=1.0, emin=20, emax=100)

# Compute quality metrics
qm = QualityMetrics(lc, binsize=1.0, emin=20, emax=100)
chi = qm.raw_chi_squared()
print(f"Chisq/dof = {chi:.2f}")
```

## Configuration

ISGRI stores configuration in default config folder for each system (see: platformdirs package)

```bash
# View current config
isgri config

# Set paths
isgri config-set --archive /path/to/archive --catalog /path/to/catalog.fits
```

Config can also be used in Python:

```python
from isgri.config import get_config

cfg = get_config()
print(cfg.archive_path)
print(cfg.catalog_path)
```

Local config file `isgri_config.toml` in current directory overrides global config.


## Documentation

- **CLI Reference**: Run `isgri --help` or `isgri <command> --help`
- **Catalog Tutorial**: [scwquery_walkthrough.ipynb](https://github.com/dominp/isgri/blob/main/demo/scwquery_walkthrough.ipynb)
- **Light Curve Tutorial**: [lightcurve_walkthrough.ipynb](https://github.com/dominp/isgri/blob/main/demo/lightcurve_walkthrough.ipynb)
- **API Reference**: Use `help()` in Python or see docstrings

## Project Structure

```
isgri/
├── catalog/          # SCW catalog query tools
│   ├── scwquery.py   # Main query interface
│   └── wcs.py        # Coordinate transformations
├── utils/            # Light curve analysis utilities
│   ├── lightcurve.py # Light curve class
│   ├── quality.py    # Quality metrics
│   ├── pif.py        # PIF tools
│   ├── file_loaders.py
│   └── time_conversion.py
├── config.py         # Configuration management
└── cli.py            # Command line interface
```

## Requirements

- Python ≥ 3.10
- astropy
- numpy
