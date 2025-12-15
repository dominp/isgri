import click
from pathlib import Path
from .catalog import ScwQuery
from .__version__ import __version__
from .config import Config


@click.group()
@click.version_option(version=__version__)
def main():
    """ISGRI - INTEGRAL/ISGRI data analysis toolkit."""
    pass


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


@main.command()
@click.option("--catalog", type=click.Path(), help="Path to catalog FITS file. If not provided, uses config value.")
@click.option("--tstart", help="Start time (YYYY-MM-DD or IJD)")
@click.option("--tstop", help="Stop time (YYYY-MM-DD or IJD)")
@click.option("--ra", type=float, help="Right ascension (degrees)")
@click.option("--dec", type=float, help="Declination (degrees)")
@click.option("--fov", type=click.Choice(["full", "any"]), default="any", help="Field of view mode")
@click.option("--max-chi", type=float, help="Maximum chi-squared value")
@click.option("--chi-type", type=click.Choice(["RAW", "CUT", "GTI"]), default="CUT", help="Type of chi-squared value")
@click.option("--revolution", "-r", help="Revolution number")
@click.option("--output", "-o", type=click.Path(), help="Output file (.fits or .csv)")
@click.option("--list-swids", is_flag=True, help="Only output SWID list")
@click.option("--count", is_flag=True, help="Only show count")
def query(catalog, tstart, tstop, ra, dec, fov, max_chi, chi_type, revolution, output, list_swids, count):
    """
    Query INTEGRAL science window catalog.

    If no catalog path is provided, uses the default from configuration.
    Multiple filters can be combined.

    Examples:
        Query by time range (IJD):

            isgri query --tstart 3000 --tstop 3100

        Query by time range (ISO date):

            isgri query --tstart 2010-01-01 --tstop 2010-12-31

        Query by sky position:

            isgri query --ra 83.63 --dec 22.01 --fov full

        Query with quality cut:

            isgri query --max-chi 2.0 --chi-type CUT

        Save results to file:

            isgri query --tstart 3000 --tstop 3100 --output results.fits

        Get only SWID list:

            isgri query --tstart 3000 --tstop 3100 --list-swids

        Count matching science windows:

            isgri query --ra 83.63 --dec 22.01 --count
    """
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


@main.command()
def config():
    """
    Show current configuration.

    Displays paths to config file, archive directory, and catalog file,
    along with their existence status.
    """
    cfg = Config()

    click.echo(f"Config file: {cfg.path}")
    click.echo(f"  Exists: {cfg.path.exists()}")
    click.echo()

    archive = cfg.archive_path
    click.echo(f"Archive path: {archive if archive else '(not set)'}")
    if archive:
        click.echo(f"  Exists: {archive.exists()}")

    try:
        catalog = cfg.catalog_path
        click.echo(f"Catalog path: {catalog if catalog else '(not set)'}")
        if catalog:
            click.echo(f"  Exists: {catalog.exists()}")
    except FileNotFoundError as e:
        click.echo(f"Catalog path: (configured but file not found)")
        click.echo(f"  Error: {e}")


@main.command()
@click.option("--archive", type=click.Path(), help="INTEGRAL archive directory path")
@click.option("--catalog", type=click.Path(), help="Catalog FITS file path")
def config_set(archive, catalog):
    """
    Set configuration values.

    Set default paths for archive directory and/or catalog file.
    Paths are expanded (~ becomes home directory) and resolved to absolute paths.
    Warns if path doesn't exist but allows setting anyway.

    Examples:

        Set archive path:

            isgri config-set --archive /anita/archivio/

        Set catalog path:

            isgri config-set --catalog ~/data/scw_catalog.fits

        Set both at once:

            isgri config-set --archive /anita/archivio/ --catalog ~/data/scw_catalog.fits
    """
    if not archive and not catalog:
        click.echo("Error: Specify at least one option (--archive or --catalog)", err=True)
        raise click.Abort()

    cfg = Config()

    if archive:
        archive_path = Path(archive).expanduser().resolve()
        if not archive_path.exists():
            click.echo(f"Warning: Archive path does not exist: {archive_path}", err=True)
            if not click.confirm("Set anyway?"):
                raise click.Abort()
        cfg.set(archive_path=archive_path)
        click.echo(f"✓ Archive path set to: {archive_path}")

    if catalog:
        catalog_path = Path(catalog).expanduser().resolve()
        if not catalog_path.exists():
            click.echo(f"Warning: Catalog file does not exist: {catalog_path}", err=True)
            if not click.confirm("Set anyway?"):
                raise click.Abort()
        cfg.set(catalog_path=catalog_path)
        click.echo(f"✓ Catalog path set to: {catalog_path}")

    click.echo()
    click.echo(f"Configuration saved to: {cfg.path}")


if __name__ == "__main__":
    main()
