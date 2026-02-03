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


def match_command(user_input):
    """Match user input to command, allowing partial matches."""
    commands = {
        "time": ["time", "t"],
        "pos": ["pos", "position", "p"],
        "quality": ["quality", "qual"],
        "revolution": ["revolution", "rev"],
        "show": ["show", "s", "display"],
        "reset": ["reset", "clear", "r"],
        "save": ["save", "write"],
        "help": ["help", "h", "?"],
        "exit": ["exit", "quit", "q"],
    }

    user_input = user_input.lower().strip()

    for cmd, aliases in commands.items():
        if user_input in aliases:
            return cmd
        for alias in aliases:
            if alias.startswith(user_input) and len(user_input) >= 2:
                return cmd

    return None


def query_direct(
    catalog_path,
    tstart,
    tstop,
    ra,
    dec,
    radius,
    fov,
    max_chi,
    chi_type,
    revolution,
    output,
    list_swids,
    count,
    columns,
):
    try:
        q = ScwQuery(catalog_path)
        initial_count = len(q.catalog)

        tstart = parse_time(tstart)
        tstop = parse_time(tstop)

        if tstart or tstop:
            q = q.time(tstart=tstart, tstop=tstop)

        if ra is not None and dec is not None:
            ra = parse_coord(ra)
            dec = parse_coord(dec)
            if radius is not None:
                q = q.position(ra=ra, dec=dec, radius=radius)
            else:
                q = q.position(ra=ra, dec=dec, fov_mode=fov)

        if max_chi is not None:
            q = q.quality(max_chi=max_chi, chi_type=chi_type)

        if revolution:
            rev_list = [int(r.strip()) for r in revolution.split(",")]
            q = q.revolution(rev_list)

        if count:
            click.echo(q.count())

        elif output:
            col_list = None
            if columns and not list_swids:
                col_list = [c.strip() for c in columns.split(",")]
                available = q.catalog.colnames
                invalid = [c for c in col_list if c not in available]
                if invalid:
                    click.echo(f"Error: Invalid columns: {invalid}", err=True)
                    click.echo(f"Available: {', '.join(available)}", err=True)
                    raise click.Abort()

            q.write(output, overwrite=True, swid_only=list_swids, columns=col_list)
            click.echo(f"Saved {q.count()} SCWs to {output}")

        else:
            results = q.get()
            click.echo(f"Found {len(results)}/{initial_count} SCWs")
            if len(results) > 0:
                display_cols = ["SWID", "TSTART", "TSTOP", "RA_SCX", "DEC_SCX", "CHI", "CUT_CHI", "GTI_CHI"]
                click.echo(results[display_cols][:10])
                if len(results) > 10:
                    click.echo(f"... and {len(results) - 10} more")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def query_interactive(catalog_path):
    click.echo("=== Interactive Query Mode ===\n")

    q = ScwQuery(catalog_path)
    click.echo(f"Loaded {len(q.catalog)} SCWs")
    click.echo("Type 'help' for available commands\n")

    while True:
        try:
            user_input = click.prompt("query>", default="").strip()
            cmd = match_command(user_input)

            if cmd == "exit":
                break
            elif cmd == "help":
                click.echo("\nAvailable commands:")
                click.echo("  time       - Filter by time range")
                click.echo("  pos        - Filter by position (FOV or radius)")
                click.echo("  quality    - Filter by chi-squared quality")
                click.echo("  revolution - Filter by revolution number(s)")
                click.echo("  show       - Display current results")
                click.echo("  reset      - Clear all filters")
                click.echo("  save       - Save results to file")
                click.echo("  exit       - Exit interactive mode")
                click.echo("\nExamples:")
                click.echo("  time: Start=2020-01-01, Stop=2020-12-31 or IJD")
                click.echo("  pos: RA=83.63, Dec=22.01 (degrees or sexagesimal)")
                click.echo("  revolution: 1234 (single) or 1234, 1235, 1236 (multiple)")
                click.echo("\nTip: You can use abbreviations (t, p, q, rev, etc.)")
                click.echo()
            elif cmd == "time":
                tstart = click.prompt("Start", default="", show_default=False)
                tstop = click.prompt("Stop", default="", show_default=False)
                tstart = parse_time(tstart) if tstart else None
                tstop = parse_time(tstop) if tstop else None
                q = q.time(tstart=tstart or None, tstop=tstop or None)
                click.echo(f"→ {q.count()} SCWs")
            elif cmd == "pos":
                ra = click.prompt("RA")
                dec = click.prompt("Dec")
                mode = click.prompt("Mode", type=click.Choice(["fov", "radius"]), default="fov")
                if mode == "radius":
                    radius = click.prompt("Radius (deg)", type=float, default=10.0)
                    q = q.position(ra=parse_coord(ra), dec=parse_coord(dec), radius=radius)
                else:
                    fov_mode = click.prompt("FOV mode", type=click.Choice(["full", "any"]), default="any")
                    q = q.position(ra=parse_coord(ra), dec=parse_coord(dec), fov_mode=fov_mode)
                click.echo(f"→ {q.count()} SCWs")
            elif cmd == "quality":
                max_chi = click.prompt("Max chi-squared", type=float)
                chi_type = click.prompt("Chi type", type=click.Choice(["CHI", "CUT", "GTI"]), default="CUT")
                q = q.quality(max_chi=max_chi, chi_type=chi_type)
                click.echo(f"→ {q.count()} SCWs")
            elif cmd == "revolution":
                rev_str = click.prompt("Revolution(s)", default="", show_default=False)
                if rev_str:
                    rev_list = [int(r.strip()) for r in rev_str.split(",")]
                    q = q.revolution(rev_list)
                    click.echo(f"→ {q.count()} SCWs")
            elif cmd == "show":
                results = q.get()
                click.echo(f"\n{len(results)} SCWs:")
                display_cols = ["SWID", "TSTART", "TSTOP", "RA_SCX", "DEC_SCX", "CHI", "CUT_CHI", "GTI_CHI"]
                click.echo(results[display_cols][:10])
                if len(results) > 10:
                    click.echo(f"... and {len(results) - 10} more")
            elif cmd == "reset":
                q = q.reset()
                click.echo(f"→ {len(q.catalog)} SCWs")
            elif cmd == "save":
                swid_only = click.confirm("Save only SWID list?", default=False)

                col_list = None
                if not swid_only:
                    if click.confirm("Select specific columns?", default=False):
                        click.echo("\nAvailable columns:")
                        available = q.catalog.colnames
                        for i, col in enumerate(available, 1):
                            click.echo(f"  {i:2d}. {col}")

                        click.echo("\nEnter column names (comma-separated) or leave empty for all:")
                        col_input = click.prompt("Columns", default="", show_default=False)

                        if col_input:
                            col_list = [c.strip() for c in col_input.split(",")]
                            invalid = [c for c in col_list if c not in available]
                            if invalid:
                                click.echo(f"Warning: Invalid columns ignored: {invalid}")
                                col_list = [c for c in col_list if c in available]

                            if not col_list:
                                click.echo("No valid columns selected, using all")
                                col_list = None

                path = click.prompt("File")
                q.write(path, overwrite=True, swid_only=swid_only, columns=col_list)
                click.echo(f"Saved {q.count()} SCWs to {path}")

            elif cmd is None:
                click.echo(f"Unknown command: {user_input}. Type 'help' for available commands.")
            else:
                click.echo(f"Unknown command: {user_input}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            click.echo("\nUse 'exit' to quit")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
