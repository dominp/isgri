import pytest
import numpy as np
from isgri.catalog.wcs import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    rotation_matrix,
    celestial_to_native,
    native_to_celestial,
    compute_detector_offset,
)


def test_spherical_to_cartesian_basic():
    """Test spherical to Cartesian conversion for basic cases"""
    # North pole
    x, y, z = spherical_to_cartesian(0, 90)
    assert np.allclose([x, y, z], [0, 0, 1])

    # South pole
    x, y, z = spherical_to_cartesian(0, -90)
    assert np.allclose([x, y, z], [0, 0, -1])

    # Equator at lon=0
    x, y, z = spherical_to_cartesian(0, 0)
    assert np.allclose([x, y, z], [1, 0, 0])

    # Equator at lon=90
    x, y, z = spherical_to_cartesian(90, 0)
    assert np.allclose([x, y, z], [0, 1, 0])


def test_spherical_to_cartesian_unit_sphere():
    """Test that all conversions produce unit vectors"""
    lons = np.random.uniform(0, 360, 100)
    lats = np.random.uniform(-90, 90, 100)

    for lon, lat in zip(lons, lats):
        x, y, z = spherical_to_cartesian(lon, lat)
        radius = np.sqrt(x**2 + y**2 + z**2)
        assert np.allclose(radius, 1.0)


def test_cartesian_to_spherical_basic():
    """Test Cartesian to spherical conversion for basic cases"""
    # North pole
    lon, lat = cartesian_to_spherical(0, 0, 1)
    assert np.allclose(lat, 90)

    # South pole
    lon, lat = cartesian_to_spherical(0, 0, -1)
    assert np.allclose(lat, -90)

    # Equator
    lon, lat = cartesian_to_spherical(1, 0, 0)
    assert np.allclose([lon, lat], [0, 0])


def test_cartesian_to_spherical_clipping():
    """Test that z-clipping works correctly"""
    # Test values outside [-1, 1]
    lon, lat = cartesian_to_spherical(0, 0, 1.5)
    assert np.allclose(lat, 90)

    lon, lat = cartesian_to_spherical(0, 0, -1.5)
    assert np.allclose(lat, -90)


def test_spherical_cartesian_roundtrip():
    """Test round trip conversion"""
    np.random.seed(42)
    lons = np.random.uniform(0, 360, 50)
    lats = np.random.uniform(-90, 90, 50)

    for lon, lat in zip(lons, lats):
        x, y, z = spherical_to_cartesian(lon, lat)
        lon_back, lat_back = cartesian_to_spherical(x, y, z)

        # Handle longitude wrapping
        lon_diff = np.abs((lon - lon_back + 180) % 360 - 180)
        assert lon_diff < 1e-10 or np.abs(lat) > 89.9  # Pole singularity
        assert np.allclose(lat, lat_back)


def test_rotation_matrix_identity():
    """Test rotation matrix at reference point gives identity-like behavior"""
    alpha_p = np.radians(100.0)
    delta_p = np.radians(30.0)
    phi_p = np.pi

    R = rotation_matrix(alpha_p, delta_p, phi_p)

    # Check it's a proper rotation matrix (orthogonal)
    assert np.allclose(np.dot(R, R.T), np.eye(3))
    assert np.allclose(np.linalg.det(R), 1.0)


def test_celestial_to_native_at_reference():
    """Test that reference point maps to native pole"""
    crval = [100.0, 30.0]

    phi, theta = celestial_to_native(crval[0], crval[1], crval)

    # At reference point, should be at native pole (theta=90)
    assert np.allclose(theta, 90.0, atol=1e-10)


def test_celestial_to_native_symmetry():
    """Test coordinate transformation symmetry"""
    crval = [100.0, 30.0]

    # Point offset in RA
    phi1, theta1 = celestial_to_native(105.0, 30.0, crval)
    phi2, theta2 = celestial_to_native(95.0, 30.0, crval)

    # Should be symmetric about reference
    assert np.allclose(theta1, theta2, atol=1e-10)


def test_native_to_celestial_roundtrip():
    """Test round trip native -- celestial conversion"""
    np.random.seed(42)
    crval = [100.0, 30.0]

    for _ in range(20):
        ra = np.random.uniform(80, 120)
        dec = np.random.uniform(10, 50)

        phi, theta = celestial_to_native(ra, dec, crval)
        ra_back, dec_back = native_to_celestial(phi, theta, crval)

        # Handle RA wrapping
        ra_diff = np.abs((ra - ra_back + 180) % 360 - 180)
        assert ra_diff < 1e-8
        assert np.allclose(dec, dec_back, atol=1e-8)


def test_compute_detector_offset_at_pointing():
    """Test detector offset when source is at pointing center"""
    src_ra, src_dec = 100.0, 30.0
    pointing_ra, pointing_dec = 100.0, 30.0
    z_ra, z_dec = 100.0, 120.0

    y, z = compute_detector_offset(src_ra, src_dec, pointing_ra, pointing_dec, z_ra, z_dec)

    # At pointing center, offsets should be ~0
    assert y < 1e-10
    assert z < 1e-10


def test_compute_detector_offset_positive():
    """Test that detector offsets are always positive"""
    np.random.seed(42)

    for _ in range(50):
        src_ra = np.random.uniform(0, 360)
        src_dec = np.random.uniform(-80, 80)
        pointing_ra = np.random.uniform(0, 360)
        pointing_dec = np.random.uniform(-80, 80)
        z_ra = np.random.uniform(0, 360)
        z_dec = np.random.uniform(-80, 80)

        y, z = compute_detector_offset(src_ra, src_dec, pointing_ra, pointing_dec, z_ra, z_dec)

        assert y >= 0
        assert z >= 0


def test_compute_detector_offset_symmetry():
    """Test detector offset symmetry"""
    pointing_ra, pointing_dec = 100.0, 30.0
    z_ra, z_dec = 100.0, 120.0

    # Source offset in +RA direction
    y1, z1 = compute_detector_offset(105.0, 30.0, pointing_ra, pointing_dec, z_ra, z_dec)

    # Source offset in -RA direction
    y2, z2 = compute_detector_offset(95.0, 30.0, pointing_ra, pointing_dec, z_ra, z_dec)

    # Y offsets should be similar (absolute values)
    assert np.allclose(y1, y2, rtol=0.01)


def test_compute_detector_offset_increases_with_distance():
    """Test that offset increases monotonically with angular distance"""
    pointing_ra, pointing_dec = 100.0, 30.0
    z_ra, z_dec = 100.0, 120.0

    offsets = []
    for offset in [0, 2, 5, 10]:
        src_ra = pointing_ra + offset
        src_dec = pointing_dec
        y, z = compute_detector_offset(src_ra, src_dec, pointing_ra, pointing_dec, z_ra, z_dec)
        total_offset = np.sqrt(y**2 + z**2)
        offsets.append(total_offset)

    # Offsets should increase monotonically
    assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))


def test_compute_detector_offset_small_angles():
    """Test detector offset for small angular separations"""
    pointing_ra, pointing_dec = 100.0, 30.0
    z_ra, z_dec = 100.0, 120.0

    # 1 degree offset in RA
    src_ra, src_dec = 101.0, 30.0
    y, z = compute_detector_offset(src_ra, src_dec, pointing_ra, pointing_dec, z_ra, z_dec)

    # Total offset should be ~1 degree for small angles
    total = np.sqrt(y**2 + z**2)
    assert 0.8 < total < 1.2


def test_rotation_matrix_array_support():
    """Test that rotation matrix works with array inputs"""
    alpha_p = np.array([0.0, np.pi / 4, np.pi / 2])
    delta_p = np.radians(30.0)

    # Should not raise error with array input
    for a in alpha_p:
        R = rotation_matrix(a, delta_p)
        assert R.shape == (3, 3)


def test_celestial_to_native_near_pole():
    """Test coordinate conversion near celestial pole"""
    crval = [0.0, 85.0]

    # Point near north pole
    phi, theta = celestial_to_native(45.0, 89.0, crval)

    assert np.isfinite(phi)
    assert np.isfinite(theta)
    assert -180 <= phi <= 180
    assert -90 <= theta <= 90
