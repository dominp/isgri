import click
from pathlib import Path
from ..catalog import ScwQuery
from ..config import Config
from ..__version__ import __version__


def parse_time(time_str):
    """
    Parse time string as IJD float or ISO date string.

    Parameters
    ----------
    time_str : str or None
        Time as "YYYY-MM-DD" or IJD number

    Returns
    -------
    float or str or None
        Parsed time value
    """
    if time_str is None:
        return None

    try:
        return float(time_str)
    except ValueError:
        return time_str


def parse_coord(coord):
    """
    Parse RA and Dec strings as float degrees or sexagesimal strings.

    Parameters
    ----------
    coord : str or None
        Coordinate as float degrees or sexagesimal string
    Returns
    -------
    float or str or None
        Parsed coordinate value
    """
    if coord is None:
        return None

    try:
        return float(coord)
    except ValueError:
        return coord


def query_direct(
    catalog, tstart, tstop, ra, dec, separation, fov, max_chi, chi_type, revolution, output, list_swids, count
):
    try:
        # Load catalog
        q = ScwQuery(catalog)
        initial_count = len(q.catalog)

        # Parse times (handle both IJD and ISO)
        tstart = parse_time(tstart)
        tstop = parse_time(tstop)

        # Apply filters
        if tstart or tstop:
            q = q.time(tstart=tstart, tstop=tstop)

        if ra is not None and dec is not None:
            ra = parse_coord(ra)
            dec = parse_coord(dec)
            if separation is not None:
                q = q.position(ra=ra, dec=dec, separation=separation)
            else:
                q = q.position(ra=ra, dec=dec, fov_mode=fov)

        if max_chi is not None:
            q = q.quality(max_chi=max_chi, chi_type=chi_type)

        if revolution:
            q = q.revolution(revolution)

        results = q.get()

        if count:
            click.echo(len(results))

        elif list_swids:
            for swid in results["SWID"]:
                click.echo(swid)

        elif output:
            if output.endswith(".csv"):
                results.write(output, format="ascii.csv", overwrite=True)
            else:
                results.write(output, format="fits", overwrite=True)
            click.echo(f"Saved {len(results)} SCWs to {output}")

        else:
            click.echo(f"Found {len(results)}/{initial_count} SCWs")
            if len(results) > 0:
                display_cols = ["SWID", "TSTART", "TSTOP", "RA_SCX", "DEC_SCX"]
                chi_col = f"{chi_type}_CHI" if chi_type != "RAW" else "CHI"
                if chi_col in results.colnames:
                    display_cols.append(chi_col)
                click.echo(results[display_cols][:10])
                if len(results) > 10:
                    click.echo(f"... and {len(results) - 10} more")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
