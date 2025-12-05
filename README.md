# ISGRI

Python toolkit for INTEGRAL/ISGRI data analysis.

## Features

### 📊 SCW Catalog Query
Query INTEGRAL Science Window catalogs with a fluent Python API:
- Filter by time, position, quality, revolution
- Calculate detector offsets
- Export results to FITS/CSV

### 💡 Light Curve Analysis
Extract and analyze ISGRI light curves:
- Event loading with PIF weighting
- Custom time binning
- Module-by-module analysis
- Quality metrics (chi-squared tests)
- Time conversions (IJD ↔ UTC)


## Installation

```bash
pip install isgri
```

## Quick Start

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

## Documentation

- **Catalog Tutorial**: [scwquery_walkthrough.ipynb](https://github.com/dominp/isgri/blob/main/demo/scwquery_walkthrough.ipynb)
- **Light Curve Tutorial**: [lightcurve_walkthrough.ipynb](https://github.com/dominp/isgri/blob/main/demo/lightcurve_walkthrough.ipynb)
- **API Reference**: Use `help()` in Python or see docstrings

## Project Structure

```
isgri/
├── catalog/          # SCW catalog query tools
│   ├── scwquery.py   # Main query interface
│   └── wcs.py        # Coordinate transformations
└── utils/            # Light curve analysis utilities
    ├── lightcurve.py # Light curve class
    ├── quality.py    # Quality metrics
    ├── pif.py        # PIF tools
    ├── file_loaders.py
    └── time_conversion.py
```

## Requirements

- Python ≥ 3.10
- astropy
- numpy
