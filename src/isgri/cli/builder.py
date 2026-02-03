import click
from pathlib import Path
from ..catalog.builder import CatalogBuilder
from ..config import Config


@click.command()
@click.option(
    "--archive", type=click.Path(), help="Path to INTEGRAL archive directory. If not provided, uses config value."
)
@click.option("--catalog", type=click.Path(), help="Path to catalog FITS file. If not provided, uses config value.")
@click.option("--cache", type=click.Path(), help="Path to light curve cache directory. Optional.")
@click.option("--cores", type=int, help="Number of CPU cores for parallel processing. Defaults to all available.")
def update(archive, catalog, cache, cores):
    """
    Update science window catalog from archive.

    Scans the INTEGRAL archive for new science windows not present in the catalog,
    processes them in parallel by revolution, and adds results to the catalog.

    If archive or catalog paths are not provided, uses values from configuration.
    Optionally caches light curve arrays to the specified directory.

    Examples:

        Update using configured paths:

            isgri update

        Update with custom paths:

            isgri update --archive /anita/archivio/ --catalog ~/data/catalog.fits

        Update with light curve caching:

            isgri update --cache ~/data/lightcurves/

        Update using 4 CPU cores:

            isgri update --cores 4
    """
    if archive is None or catalog is None:
        cfg = Config()

        if archive is None:
            archive = cfg.archive_path
            if not archive:
                click.echo("Error: No archive path configured", err=True)
                raise click.Abort()

        if catalog is None:
            catalog = cfg.catalog_path
            if not catalog:
                click.echo("Error: No catalog path configured", err=True)
                raise click.Abort()

    archive_path = Path(archive).expanduser().resolve()
    catalog_path = Path(catalog).expanduser().resolve()
    cache_path = Path(cache).expanduser().resolve() if cache else None

    if not archive_path.exists():
        click.echo(f"Error: Archive directory does not exist: {archive_path}", err=True)
        raise click.Abort()

    if not catalog_path.parent.exists():
        click.echo(f"Warning: Catalog directory does not exist: {catalog_path.parent}", err=True)
        if not click.confirm("Create new catalog file?"):
            raise click.Abort()

    if cache_path and not cache_path.exists():
        click.echo(f"Warning: Cache directory does not exist: {cache_path}", err=True)
        if not click.confirm("Create directory?"):
            raise click.Abort()
        cache_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"✓ Created cache directory: {cache_path}")

    click.echo(f"Archive: {archive_path}")
    click.echo(f"Catalog: {catalog_path}")
    if cache_path:
        click.echo(f"Cache: {cache_path}")
    else:
        click.echo("Cache: (none)")
    if cores:
        click.echo(f"CPU cores: {cores}")
    click.echo()

    try:
        builder = CatalogBuilder(
            archive_path=str(archive_path),
            catalog_path=str(catalog_path),
            lightcurve_cache=str(cache_path) if cache_path else None,
            n_cores=cores,
        )

        builder.update_catalog()

        click.echo()
        click.echo("Catalog update complete")

    except Exception as e:
        click.echo(f"Error during catalog update: {e}", err=True)
        raise click.Abort()
