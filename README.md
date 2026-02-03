# ISGRI

Python toolkit for INTEGRAL/ISGRI data analysis.

## Features

### Command Line Interface
Query catalogs directly from the terminal:
- Interactive and direct query modes
- Filter by time, position, quality, revolution
- Export results to FITS/CSV or SWID lists
- Update catalogs with new science windows from archive

### SCW Catalog Query
Query INTEGRAL Science Window catalogs with a fluent Python API:
- Filter by time, position, quality, revolution
- Calculate detector offsets
- Export results to any auto detectable astropy table extension or in table aligned data for any other extension

### Catalog Builder
Build and update INTEGRAL/ISGRI science window catalogs:
- Automatic discovery of new science windows in archive
- Parallel processing of quality metrics
- Optional light curve caching
- Incremental catalog updates

### Light Curve Analysis
Extract and analyze ISGRI light curves:
- Load from file paths or SWID/source lookup
- PIF weighting with adjustable thresholds
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
isgri config-set --archive /path/to/archive --catalog ~/data/scw_catalog.fits --pif /path/to/pif

# View current config
isgri config

# Update catalog from archive
isgri update

# Interactive catalog query
isgri query
# query> time
# Start: 2010-01-01
# Stop: 2010-12-31
# → 1234 SCWs
# query> pos
# RA: 83.63
# Dec: 22.01
# ...

# Direct catalog query
isgri query --tstart 2010-01-01 --tstop 2010-12-31 --ra 83.63 --dec 22.01 --max-chi 2.0

# Get SWID list for batch processing
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

# Filter by revolution(s)
results = cat.revolution([1000, 1001, 1002]).get()

# Save selected columns
cat.write('results.fits', columns=['SWID', 'TSTART', 'TSTOP'])

# Get SWID list
swids = cat.get_swids()

print(f"Found {len(results)} observations")
```

### Build/Update SCW Catalog

#### Command Line

```bash
# Update catalog using configured paths
isgri update

# Update with custom paths
isgri update --archive /anita/archivio/ --catalog ~/data/catalog.fits 

# Enable light curve caching (15-1000 keV, 1s bins)
isgri update --cache ~/data/lightcurves/

# Limit CPU cores for parallel processing
isgri update --cores 4
```

#### Python API

```python
from isgri.catalog import CatalogBuilder

# Create builder instance
builder = CatalogBuilder(
    archive_path="/path/to/archive",
    catalog_path="scw_catalog.fits",
    lightcurve_cache="/path/to/cache",  # optional
    n_cores=8
)

# Update catalog with new science windows
builder.update_catalog()

# Find all science windows in archive
swids, paths = builder.find_scws()
print(f"Found {len(swids)} science windows")
```

The builder:
- Scans archive for new ScWs not in catalog
- Computes quality metrics (raw, sigma-clipped, GTI-filtered chi-squared)
- Processes in parallel by revolution
- Optionally caches 1s light curves (15-1000 keV)

### Analyze Light Curves

```python
from isgri.utils import LightCurve, QualityMetrics

# Method 1: Load from file paths
lc = LightCurve.load_data(
    events_path="isgri_events.fits",
    pif_path="source_model.fits",
    use_pif=True,
    pif_threshold=0.5
)

# Method 2: Load by SWID (requires config)
lc = LightCurve.load_data(swid="255900280010")

# Method 3: Load by SWID + source (auto-finds PIF)
lc = LightCurve.load_data(
    swid="255900280010",
    source="SGR1935",
    use_pif=True,
    pif_threshold=0.5
)

# Create binned lightcurve
time, counts = lc.rebin(binsize=1.0, emin=30, emax=100)

# Override PIF settings temporarily
time, counts = lc.rebin(binsize=1.0, emin=30, emax=100, use_pif=True, pif_threshold=0.8)

# Or change instance settings
lc.pif_threshold = 0.7
lc.use_pif = True
time, counts = lc.rebin(binsize=1.0, emin=30, emax=100)

# Module-by-module analysis
times, module_counts = lc.rebin_by_modules(binsize=1.0, emin=30, emax=300)

# Quality metrics
qm = QualityMetrics(lc, binsize=1.0, emin=30, emax=100)
chi_raw = qm.raw_chi_squared()
chi_clipped = qm.sigma_clip_chi_squared(sigma=1)
chi_gti = qm.gti_chi_squared()
print(f"Chisq/dof = {chi_raw:.2f}")

# Time conversions
from isgri.utils.time_conversion import ijd2utc, utc2ijd
print(f"Start time: {ijd2utc(lc.t0)}")
```

## Configuration

ISGRI stores configuration in default config folder for each system (see: platformdirs package)

```bash
# View current config
isgri config

# Set paths
isgri config-set --archive /path/to/archive
isgri config-set --catalog /path/to/catalog.fits
isgri config-set --pif /path/to/pif

# Set all at once
isgri config-set --archive /path/to/archive --catalog /path/to/catalog.fits --pif /path/to/pif
```

Config in Python:

```python
from isgri.config import Config

cfg = Config()
print(cfg.archive_path)
print(cfg.catalog_path)
print(cfg.pif_path)

# Create new config programmatically
cfg.create_new(
    archive_path="/path/to/archive",
    catalog_path="/path/to/catalog.fits",
    pif_path="/path/to/pif"
)
```

Local config file `isgri_config.toml` in current directory overrides global config.


## Documentation

- **CLI Reference**: Run `isgri --help` or `isgri <command> --help`
- **Catalog Tutorial**: [scwquery_walkthrough.ipynb](https://github.com/dominp/isgri/blob/main/demo/scwquery_walkthrough.ipynb)
- **Light Curve Tutorial**: [lightcurve_walkthrough.ipynb](https://github.com/dominp/isgri/blob/main/demo/lightcurve_walkthrough.ipynb)
- **API Reference**: Use `help()` in Python or see docstrings


## License

MIT