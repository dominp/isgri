import click
from pathlib import Path
from .catalog import ScwQuery
from .__version__ import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    pass


def parse_time(time_str):
    if time_str is None:
        return None

    try:
        return float(time_str)
    except ValueError:
        return time_str


@main.command()
@click.argument("catalog", type=click.Path(exists=True))
@click.option("--tstart", help="Start time (YYYY-MM-DD or IJD)")
@click.option("--tstop", help="Stop time (YYYY-MM-DD or IJD)")
@click.option("--ra", type=float, help="Right ascension (degrees)")
@click.option("--dec", type=float, help="Declination (degrees)")
@click.option("--fov", type=click.Choice(["full", "any"]), default="any", help="Field of view mode")
@click.option("--max-chi", type=float, help="Maximum chi-squared value")
@click.option('--chi-type', type=click.Choice(['RAW','CUT','GTI']), default='CUT',help="Type of chi-squared value")
@click.option("--revolution", type=click.Choice(['']) "-r", help="Revolution number")
@click.option("--output", "-o", type=click.Path(), help="Output file (.fits or .csv)")
@click.option("--list-swids", is_flag=True, help="Only output SWID list")
@click.option("--count", is_flag=True, help="Only show count")
def query(catalog, tstart, tstop, ra, dec, fov, max_chi, chi_type, revolution, output, list_swids, count):
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
                if "CHI" in results.colnames:
                    display_cols.append("CHI")
                click.echo(results[display_cols][:10])
                if len(results) > 10:
                    click.echo(f"... and {len(results) - 10} more")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
