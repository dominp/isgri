# isgri

Python package for ISGRI (INTEGRAL Soft Gamma-Ray Imager) lightcurve analysis.

## Installation

```bash
pip install isgri
```

Or from source:
```bash
git clone https://github.com/yourusername/isgri.git
cd isgri
pip install -e .
```

## Quick Start

```python
from isgri.utils import LightCurve, QualityMetrics

# Load data
lc = LightCurve.load_data(
    events_path='/path/to/isgri_events.fits',
    pif_path='/path/to/source_model.fits'
)

# Create lightcurve
time, counts = lc.rebin(binsize=1.0, emin=30, emax=300)

# Create lightcurve per each module
times, module_counts = lc.rebin_by_modules(binsize=1.0, emin=30, emax=300)

# Compute quality metrics
qm = QualityMetrics(lc, binsize=1.0, emin=30, emax=300)
chi = qm.raw_chi_squared()
```

## Features

- Load ISGRI events and PIF files
- Rebin lightcurves with custom binning
- Module-by-module analysis (8 detector modules)
- Quality metrics (chi-squared variability tests)
- Time conversions (IJD ↔ UTC)
- Custom event filtering

## Documentation

- **Tutorial**: See [demo/lightcurve_walkthrough.ipynb](demo/lightcurve_walkthrough.ipynb)
- **API docs**: All functions have docstrings - use `help(LightCurve)`