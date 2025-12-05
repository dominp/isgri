"""
WCS coordinate transformations for celestial coordinates.

Implements spherical coordinate rotations following Calabretta & Greisen (2002),
"Representations of celestial coordinates in FITS", A&A 395, 1077-1122.
https://doi.org/10.1051/0004-6361:20021327
"""

from typing import Union
import numpy.typing as npt
import numpy as np


def spherical_to_cartesian(lon, lat):
    """
    Convert spherical coordinates to Cartesian unit vectors.

    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees

    Returns:
        tuple: (x, y, z) Cartesian coordinates on unit sphere
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian unit vectors to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        tuple: (lon, lat) in degrees
    """
    # Clamp z to valid range for arcsin
    z = np.clip(z, -1.0, 1.0)

    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    return lon, lat


def rotation_matrix(alpha_p, delta_p, phi_p=np.pi):
    """
    Compute rotation matrix for coordinate transformation.

    Following Calabretta & Greisen (2002), equations (5) and (7).
    Assumes theta_0 = 90° (most common case).

    Args:
        alpha_p: Reference point RA in radians
        delta_p: Reference point Dec in radians
        phi_p: Native longitude of celestial pole (default: π for standard orientation)

    Returns:
        ndarray: 3x3 rotation matrix
    """
    sa = np.sin(alpha_p)
    ca = np.cos(alpha_p)
    sd = np.sin(delta_p)
    cd = np.cos(delta_p)
    sp = np.sin(phi_p)
    cp = np.cos(phi_p)

    # Rotation matrix from Calabretta & Greisen (2002), eq. (5)
    R = np.array(
        [
            [-sa * sp - ca * cp * sd, ca * sp - sa * cp * sd, cp * cd],
            [sa * cp - ca * sp * sd, -ca * cp - sa * sp * sd, sp * cd],
            [ca * cd, sa * cd, sd],
        ]
    )

    return R


def celestial_to_native(lon, lat, crval, longpole=180.0):
    """
    Transform from celestial (RA/Dec) to native spherical coordinates.

    Args:
        lon: Celestial longitude (RA) in degrees
        lat: Celestial latitude (Dec) in degrees
        crval: Reference point [RA, Dec] in degrees
        longpole: Native longitude of celestial north pole (default: 180°)

    Returns:
        tuple: (phi, theta) native coordinates in degrees
    """
    alpha_p = np.radians(crval[0])
    delta_p = np.radians(crval[1])
    phi_p = np.radians(longpole)

    x, y, z = spherical_to_cartesian(lon, lat)

    R = rotation_matrix(alpha_p, delta_p, phi_p)
    x_rot = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
    y_rot = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
    z_rot = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z

    phi, theta = cartesian_to_spherical(x_rot, y_rot, z_rot)

    return phi, theta


def native_to_celestial(phi, theta, crval, longpole=180.0):
    """
    Transform from native spherical to celestial (RA/Dec) coordinates.

    Args:
        phi: Native longitude in degrees
        theta: Native latitude in degrees
        crval: Reference point [RA, Dec] in degrees
        longpole: Native longitude of celestial north pole (default: 180°)

    Returns:
        tuple: (lon, lat) celestial coordinates in degrees
    """
    alpha_p = np.radians(crval[0])
    delta_p = np.radians(crval[1])
    phi_p = np.radians(longpole)

    x, y, z = spherical_to_cartesian(phi, theta)

    R = rotation_matrix(alpha_p, delta_p, phi_p).T  # Transpose for inverse rotation
    x_rot = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
    y_rot = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
    z_rot = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z

    lon, lat = cartesian_to_spherical(x_rot, y_rot, z_rot)

    return lon, lat


def compute_detector_offset(
    src_ra: Union[float, npt.ArrayLike],
    src_dec: Union[float, npt.ArrayLike],
    pointing_ra: float,
    pointing_dec: float,
    z_ra: float,
    z_dec: float,
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute source offset in INTEGRAL detector coordinates.

    Args:
        src_ra: Source RA in degrees
        src_dec: Source Dec in degrees
        pointing_ra: Pointing axis RA in degrees
        pointing_dec: Pointing axis Dec in degrees
        z_ra: Z-axis RA in degrees
        z_dec: Z-axis Dec in degrees

    Returns:
        tuple: (y_offset, z_offset) in degrees (absolute values)
    """
    # Transform Z-axis to native coordinates to get roll angle
    scZ_phi, _ = celestial_to_native(z_ra, z_dec, [pointing_ra, pointing_dec])
    roll = scZ_phi - 180.0

    # Transform source to native coordinates
    phi, theta = celestial_to_native(src_ra, src_dec, [pointing_ra, pointing_dec])

    # Convert to detector coordinates
    # theta is elevation from pointing axis
    theta = 90.0 - theta

    # phi is azimuth, correct for roll
    phi = phi + 90.0 - roll

    # Project onto detector Y and Z axes
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    y = np.degrees(np.arctan(np.tan(theta_rad) * np.cos(phi_rad)))
    z = np.degrees(np.arctan(np.tan(theta_rad) * np.sin(phi_rad)))

    return np.abs(y), np.abs(z)
