import click
from pathlib import Path
from ..catalog import ScwQuery
from ..__version__ import __version__
from ..config import Config
from .query import query_direct, query_interactive


@click.group()
@click.version_option(version=__version__)
def main():
    """ISGRI - INTEGRAL/ISGRI data analysis toolkit."""
    pass


@main.command()
@click.option("--catalog", type=click.Path(), help="Path to catalog FITS file. If not provided, uses config value.")
@click.option("--tstart", help="Start time (YYYY-MM-DD or IJD)")
@click.option("--tstop", help="Stop time (YYYY-MM-DD or IJD)")
@click.option("--ra", help="Right ascension (degrees or HH:MM:SS)")
@click.option("--dec", help="Declination (degrees or DD:MM:SS)")
@click.option("--radius", type=float, help="Angular separation (degrees)")
@click.option("--fov", type=click.Choice(["full", "any"]), default="any", help="Field of view mode")
@click.option("--max-chi", type=float, help="Maximum chi-squared value")
@click.option("--chi-type", type=click.Choice(["RAW", "CUT", "GTI"]), default="CUT", help="Type of chi-squared value")
@click.option("--revolution", help="Revolution number")
@click.option(
    "--output", "-o", type=click.Path(), help="Output file (.fits or .csv or any if --list-swids or --count)"
)
@click.option("--list-swids", is_flag=True, help="Only output SWID list")
@click.option("--count", is_flag=True, help="Only show count")
def query(catalog, tstart, tstop, ra, dec, radius, fov, max_chi, chi_type, revolution, output, list_swids, count):
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
            isgri query --ra 83.63 --dec 22.01 --radius 5.0

        Query with quality cut:

            isgri query --max-chi 2.0 --chi-type CUT

        Save results to file:

            isgri query --tstart 3000 --tstop 3100 --output results.fits

        Get only SWID list:

            isgri query --tstart 3000 --tstop 3100 --list-swids

        Count matching science windows:

            isgri query --ra 83.63 --dec 22.01 --count
    """
    if catalog is None:
        cfg = Config()
        catalog = cfg.catalog_path

        if not catalog:
            click.echo("Error: No catalog configured", err=True)
            raise click.Abort()

    if any(param is not None for param in [tstart, tstop, ra, dec, radius, max_chi, revolution]):
        query_direct(
            catalog, tstart, tstop, ra, dec, radius, fov, max_chi, chi_type, revolution, output, list_swids, count
        )
    else:
        query_interactive(catalog)


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
