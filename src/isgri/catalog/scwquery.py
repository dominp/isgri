from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
from dataclasses import dataclass
from isgri.utils import ijd2utc, utc2ijd
from .wcs import compute_detector_offset


@dataclass
class Filter:
    """Filter with mask and parameters"""

    name: str
    mask: np.ndarray
    params: dict


class ScwQuery:
    """
    Query interface for INTEGRAL SCW catalog.

    Parameters
    ----------
    catalog_path : str or Path
        Path to SCW catalog FITS file

    Examples
    --------
    >>> query = ScwQuery("data/scw_catalog.fits")
    >>> results = query.time(tstart=3000).quality(max_chi=2.0).get()
    >>>
    >>> # FOV-based filtering
    >>> results = query.position(ra=83.63, dec=22.01, fov_mode="full").get()

    See Also
    --------
    time : Filter by time range
    quality : Filter by data quality
    position : Filter by sky position
    revolution : Filter by revolution number
    """

    ISGRI_FULLY_CODED = 4.0  # half-width in degrees (8x8 total)
    ISGRI_DETECTOR_EDGE = 14.5  # half-width in degrees (29x29 total)

    def __init__(self, catalog_path: Union[str, Path]):
        self.catalog_path = Path(catalog_path)
        self._catalog: Optional[Table] = None
        self._mask: Optional[np.ndarray] = None
        self._filters: list[Filter] = []

    @property
    def catalog(self) -> Table:
        """Lazy load catalog from FITS file"""
        if self._catalog is None:
            if not self.catalog_path.exists():
                raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")
            self._catalog = Table.read(self.catalog_path)
            self._validate_catalog()
        return self._catalog

    def _validate_catalog(self):
        """Check required columns exist"""
        required = ["SWID", "TSTART", "TSTOP", "RA_SCX", "DEC_SCX", "RA_SCZ", "DEC_SCZ", "CHI"]
        missing = [col for col in required if col not in self._catalog.colnames]
        if missing:
            raise ValueError(f"Catalog missing required columns: {missing}")

    @property
    def mask(self) -> np.ndarray:
        """Initialize mask if needed"""
        if self._mask is None:
            self._mask = np.ones(len(self.catalog), dtype=bool)
        return self._mask

    def time(
        self, tstart: Optional[Union[float, str]] = None, tstop: Optional[Union[float, str]] = None
    ) -> "ScwQuery":
        """
        Filter by time range.

        Parameters
        ----------
        tstart : float or str, optional
            Start time in IJD (float) or ISO format (str)
        tstop : float or str, optional
            Stop time in IJD (float) or ISO format (str)

        Returns
        -------
        ScwQuery
            Self for method chaining

        Examples
        --------
        >>> query.time(tstart="2010-01-01", tstop="2010-12-31")
        >>> query.time(tstart=3000.0)  # IJD format
        >>> query.time(tstop="2015-01-01")  # Only upper bound
        """
        mask = np.ones(len(self.catalog), dtype=bool)

        if tstart is not None:
            tstart_ijd = self._parse_time(tstart)
            mask &= self.catalog["TSTOP"] >= tstart_ijd

        if tstop is not None:
            tstop_ijd = self._parse_time(tstop)
            mask &= self.catalog["TSTART"] <= tstop_ijd

        self._add_filter(Filter(name="time", mask=mask, params={"tstart": tstart, "tstop": tstop}))
        return self

    def quality(self, max_chi: Optional[float] = None, chi_type: str = "CHI") -> "ScwQuery":
        """
        Filter by quality metric (lower chi-squared means better quality).

        Parameters
        ----------
        max_chi : float, optional
            Maximum chi-squared value to accept
        chi_type : str, default "CHI"
            Column name: "CHI", "CUT_CHI", or "GTI_CHI"

        Returns
        -------
        ScwQuery
            Self for method chaining

        Examples
        --------
        >>> query.quality(max_chi=2.0)  # High quality data
        >>> query.quality(max_chi=5.0, chi_type="CUT_CHI")  # Alternative metric

        """
        if chi_type not in self.catalog.colnames:
            raise ValueError(f"Column {chi_type} not found in catalog")

        mask = np.ones(len(self.catalog), dtype=bool)

        if max_chi is not None:
            if max_chi <= 0:
                raise ValueError("max_chi must be positive")
            mask &= self.catalog[chi_type] <= max_chi

        self._add_filter(Filter(name="quality", mask=mask, params={"max_chi": max_chi, "chi_type": chi_type}))
        return self

    def position(
        self,
        ra: Optional[Union[float, str]] = None,
        dec: Optional[Union[float, str]] = None,
        radius: Optional[float] = None,
        target: Optional[SkyCoord] = None,
        fov_mode: Optional[Literal["full", "any"]] = None,
        max_offset: Optional[float] = None,
    ) -> "ScwQuery":
        """
        Filter by sky position using angular separation or FOV constraints.

        Parameters
        ----------
        ra : float or str, optional
            Right ascension in degrees or HMS format
        dec : float or str, optional
            Declination in degrees or DMS format
        radius : float, optional
            Angular separation radius in degrees (simple cone search)
        target : SkyCoord, optional
            Target position as SkyCoord (alternative to ra/dec)
        fov_mode : {'full', 'any'}, optional
            FOV filtering mode using detector coordinates:
            - 'full': fully coded FOV (both |Y| and |Z| <= 4 deg)
            - 'any': detector FOV (both |Y| and |Z| <= 14.5 deg)
        max_offset : float, optional
            Custom maximum offset in degrees (uses max of |Y|, |Z|)

        Returns
        -------
        ScwQuery
            Self for method chaining

        Notes
        -----
        When fov_mode or max_offset is specified, uses compute_detector_offset() to calculate
        detector Y/Z offsets from pointing center. Otherwise uses simple angular
        separation from X-axis pointing.

        Examples
        --------
        >>> query.position(ra=83.63, dec=22.01, radius=5.0)
        >>> query.position(ra=83.63, dec=22.01, fov_mode="full")
        >>> query.position(ra="05h34m31s", dec="+22d00m52s", fov_mode="any")
        """
        if target is None:
            if ra is None or dec is None:
                return self
            target = self._parse_position(ra, dec)

        if isinstance(ra, (int, float)) and not (0 <= ra < 360):
            raise ValueError(f"RA must be in [0, 360), got {ra}")
    
        if isinstance(dec, (int, float)) and not (-90 <= dec <= 90):
            raise ValueError(f"Dec must be in [-90, 90], got {dec}")
        
        if radius is not None and radius <= 0:
            raise ValueError("radius must be positive")
        
        if fov_mode is not None and fov_mode not in ['full', 'any']:
            raise ValueError(f"Invalid fov_mode: {fov_mode}. Use 'full' or 'any'")

        mask = np.ones(len(self.catalog), dtype=bool)

        if fov_mode is not None or max_offset is not None:
            y_off, z_off, max_off = self._compute_detector_offsets(target)

            if fov_mode == "full":
                mask &= (np.abs(y_off) <= self.ISGRI_FULLY_CODED) & (np.abs(z_off) <= self.ISGRI_FULLY_CODED)
                filter_params = {
                    "ra": target.ra.deg,
                    "dec": target.dec.deg,
                    "fov_mode": "full",
                    "max_offset": self.ISGRI_FULLY_CODED,
                    "y_offset": y_off,
                    "z_offset": z_off,
                    "max_offset_actual": max_off,
                }

            elif fov_mode == "any":
                mask &= (np.abs(y_off) <= self.ISGRI_DETECTOR_EDGE) & (np.abs(z_off) <= self.ISGRI_DETECTOR_EDGE)
                filter_params = {
                    "ra": target.ra.deg,
                    "dec": target.dec.deg,
                    "fov_mode": "any",
                    "max_offset": self.ISGRI_DETECTOR_EDGE,
                    "y_offset": y_off,
                    "z_offset": z_off,
                    "max_offset_actual": max_off,
                }

            elif max_offset is not None:
                mask &= max_off <= max_offset
                filter_params = {
                    "ra": target.ra.deg,
                    "dec": target.dec.deg,
                    "fov_mode": "custom",
                    "max_offset": max_offset,
                    "y_offset": y_off,
                    "z_offset": z_off,
                    "max_offset_actual": max_off,
                }

        else:
            pointings_x = SkyCoord(self.catalog["RA_SCX"], self.catalog["DEC_SCX"], unit="deg")
            separations = target.separation(pointings_x).deg

            if radius is not None:
                if radius <= 0:
                    raise ValueError("radius must be positive")
                mask &= separations <= radius

            filter_params = {
                "ra": target.ra.deg,
                "dec": target.dec.deg,
                "radius": radius,
                "separations": separations,
            }

        self._add_filter(Filter(name="position", mask=mask, params=filter_params))
        return self

    def revolution(self, revolutions: Union[int, str, list[Union[int, str]]]) -> "ScwQuery":
        """
        Filter by revolution number(s).

        Parameters
        ----------
        revolutions : int, str, or list
            Revolution number(s) as integer (255), 4-digit string ("0255"),
            or list of mixed types

        Returns
        -------
        ScwQuery
            Self for method chaining

        Examples
        --------
        >>> query.revolution(255)
        >>> query.revolution("0255")
        >>> query.revolution([255, "0256", 300])
        """
        if not isinstance(revolutions, list):
            revolutions = [revolutions]

        rev_ints = []
        for rev in revolutions:
            if isinstance(rev, int):
                rev_ints.append(rev)
            elif isinstance(rev, str):
                if len(rev) != 4:
                    raise ValueError(f"Revolution string must be 4 digits: '{rev}'")
                try:
                    rev_ints.append(int(rev))
                except ValueError:
                    raise ValueError(f"Invalid revolution string: '{rev}'")
            else:
                raise TypeError(f"Revolution must be int or str, got {type(rev)}")

        mask = np.isin(self.catalog["REVOL"], rev_ints)
        self._add_filter(Filter(name="revolution", mask=mask, params={"revolutions": rev_ints}))
        return self

    def get(self) -> Table:
        """
        Apply all filters and return filtered catalog.

        Returns
        -------
        Table
            Filtered catalog as astropy Table

        Notes
        -----
        This is typically the final call in a filter chain:

        Examples
        --------
        >>> results = query.time(tstart=3000).quality(max_chi=2.0).get()
        >>> print(len(results))
        """
        combined_mask = self.mask.copy()
        for filt in self._filters:
            combined_mask &= filt.mask
        return self.catalog[combined_mask]

    def count(self) -> int:
        """
        Count SCWs matching current filters.

        Returns
        -------
        int
            Number of matching SCWs

        Examples
        --------
        >>> query.time(tstart=3000).count()
        150
        >>> # Faster than len(query.get()) for large catalogs
        """
        return len(self.get())

    def reset(self) -> "ScwQuery":
        """
        Clear all filters and reset to full catalog.

        Returns
        -------
        ScwQuery
            Self for method chaining

        Examples
        --------
        >>> query.time(tstart=3000).get()  # First query
        >>> query.reset()  # Clear filters
        >>> query.quality(max_chi=2.0).get()  # New query
        """
        self._filters.clear()
        self._mask = None
        return self

    def _compute_detector_offsets(self, target: SkyCoord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute detector Y/Z offsets using compute_detector_offset.

        Parameters
        ----------
        target : SkyCoord
            Target sky position

        Returns
        -------
        y_offset : ndarray
            Y-axis offsets in degrees
        z_offset : ndarray
            Z-axis offsets in degrees
        max_offset : ndarray
            Maximum of |Y| and |Z| offsets
        """
        y_off, z_off = compute_detector_offset(
            target.ra.deg,
            target.dec.deg,
            self.catalog["RA_SCX"],
            self.catalog["DEC_SCX"],
            self.catalog["RA_SCZ"],
            self.catalog["DEC_SCZ"],
        )
        max_off = np.maximum(np.abs(y_off), np.abs(z_off))
        return y_off, z_off, max_off

    def _add_filter(self, filter: Filter):
        """Replace existing filter with same name or add new filter"""
        self._filters = [f for f in self._filters if f.name != filter.name]
        self._filters.append(filter)

    def _parse_time(self, time: Union[float, str]) -> float:
        """
        Parse time to IJD format.

        Parameters
        ----------
        time : float or str
            Time as IJD (< 51544), MJD (>= 51544), or ISO string

        Returns
        -------
        float
            Time in IJD format
        """
        if isinstance(time, (int, float)):
            return time if time < 51544 else time - 51544
        if isinstance(time, str):
            return utc2ijd(time)
        raise TypeError(f"Invalid time type: {type(time)}")

    def _parse_position(self, ra: Union[float, str], dec: Union[float, str]) -> SkyCoord:
        """
        Parse coordinates to SkyCoord.

        Parameters
        ----------
        ra : float or str
            Right ascension as degrees or HMS string
        dec : float or str
            Declination as degrees or DMS string

        Returns
        -------
        SkyCoord
            Parsed coordinate
        """
        if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
            return SkyCoord(ra, dec, unit="deg")

        if isinstance(ra, str) and isinstance(dec, str):
            try:
                return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            except:
                try:
                    return SkyCoord(ra, dec, unit="deg")
                except Exception as e:
                    raise ValueError(f"Could not parse position: {ra}, {dec}") from e

        raise TypeError(f"Invalid position types: {type(ra)}, {type(dec)}")

    @property
    def filters_summary(self) -> dict:
        """
        Get summary of applied filters.

        Returns
        -------
        dict
            Dictionary mapping filter names to their parameters
        """
        return {f.name: f.params for f in self._filters}

    def get_offsets(self, ra: Union[float, str], dec: Union[float, str]) -> Table:
        """
        Get filtered catalog with detector offsets computed.

        Parameters
        ----------
        ra : float or str
            Right ascension
        dec : float or str
            Declination

        Returns
        -------
        Table
            Filtered catalog with Y_OFFSET, Z_OFFSET, MAX_OFFSET columns added

        Examples
        --------
        >>> results = query.time(tstart=3000).get_offsets(ra=83.63, dec=22.01)
        >>> fully_coded = results[results['MAX_OFFSET'] <= 4.0]
        """
        target = self._parse_position(ra, dec)
        y_off, z_off, max_off = self._compute_detector_offsets(target)

        result = self.get()
        combined_mask = self._get_combined_mask()
        result["Y_OFFSET"] = y_off[combined_mask]
        result["Z_OFFSET"] = z_off[combined_mask]
        result["MAX_OFFSET"] = max_off[combined_mask]
        return result

    def _get_combined_mask(self) -> np.ndarray:
        """Get combined mask from all active filters"""
        combined_mask = self.mask.copy()
        for filt in self._filters:
            combined_mask &= filt.mask
        return combined_mask

    def __repr__(self) -> str:
        n_total = len(self.catalog)
        n_selected = self.count()
        return (
            f"ScwQuery(catalog={self.catalog_path.name}, "
            f"total={n_total}, selected={n_selected}, "
            f"filters={list(self.filters_summary.keys())})"
        )
