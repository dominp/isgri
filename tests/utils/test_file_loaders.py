import pytest
import numpy as np
from astropy.io import fits
import tempfile
import os


@pytest.fixture
def mock_events_file():
    """Create a minimal mock ISGRI events FITS file."""
    n_events = 1000
    
    events = np.zeros(n_events, dtype=[
        ('TIME', 'f8'),
        ('ISGRI_ENERGY', 'f4'),
        ('DETY', 'i2'),
        ('DETZ', 'i2'),
        ('SELECT_FLAG', 'i2'),
    ])
    
    events['TIME'] = np.linspace(0, 100/ 86400, n_events)
    events['ISGRI_ENERGY'] = np.random.uniform(30, 300, n_events)
    events['DETY'] = np.random.randint(0, 128, n_events)
    events['DETZ'] = np.random.randint(0, 134, n_events)
    events['SELECT_FLAG'] = 0
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.fits') as f:
        hdu = fits.BinTableHDU(data=events, name='ISGR-EVTS-ALL')
        hdu.header['REVOL'] = 1000
        hdu.header['SWID'] = '100000100010'
        hdu.header['TSTART'] = 0.0
        hdu.header['TSTOP'] = 100.0
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(f.name, overwrite=True)
        filepath = f.name
    
    yield filepath
    os.unlink(filepath)


from isgri.utils.file_loaders import load_isgri_events, verify_events_path


def test_verify_events_path_file(mock_events_file):
    """Test path verification with valid file."""
    path = verify_events_path(mock_events_file)
    assert path == mock_events_file


def test_verify_events_path_invalid():
    """Test path verification with invalid path."""
    with pytest.raises(FileNotFoundError):
        verify_events_path("/nonexistent/path.fits")


def test_load_isgri_events(mock_events_file):
    """Test loading events from FITS file."""
    events, gtis, metadata = load_isgri_events(mock_events_file)
    
    assert len(events) > 0
    assert 'TIME' in events.dtype.names
    assert 'ISGRI_ENERGY' in events.dtype.names
    assert 'DETY' in events.dtype.names
    assert 'DETZ' in events.dtype.names
    
    assert metadata['REVOL'] == 1000
    assert metadata['SWID'] == '100000100010'
    assert metadata['TSTART'] == 0.0
    
    assert gtis.shape[1] == 2  # Start/stop pairs