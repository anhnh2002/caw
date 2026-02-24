"""CLI subcommands for `caw auth`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer


app = typer.Typer(help="Manage credentials for Docker containers.")


@app.command()
def collect(
    agents: Annotated[
        Optional[list[str]],
        typer.Option("--agents", "-a", help="Agents to include (claude, codex, gemini, cursor, or all)"),
    ] = None,
    source_home: Annotated[
        str,
        typer.Option("--source-home", help="Source home directory to read credentials from"),
    ] = str(Path.home()),
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing auth dir")] = False,
    link: Annotated[bool, typer.Option("--link", "-l", help="Also link credential files after collecting")] = False,
):
    """Collect credentials from host into ~/.caw/auth/."""
    from .collector import collect as do_collect

    do_collect(agents=agents or ["all"], source_home=source_home, force=force, link=link)


@app.command()
def link(
    agents: Annotated[
        Optional[list[str]],
        typer.Option("--agents", "-a", help="Agents to link"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show what would be done")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing backups")] = False,
):
    """Replace host credential files with symlinks to ~/.caw/auth/."""
    from .linker import link as do_link

    do_link(agents=agents, dry_run=dry_run, force=force)


@app.command()
def unlink(
    agents: Annotated[
        Optional[list[str]],
        typer.Option("--agents", "-a", help="Agents to unlink"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show what would be done")] = False,
):
    """Restore original credential files from backups."""
    from .linker import unlink as do_unlink

    do_unlink(agents=agents, dry_run=dry_run)


@app.command("status")
def status_cmd(
    agents: Annotated[
        Optional[list[str]],
        typer.Option("--agents", "-a", help="Agents to show"),
    ] = None,
):
    """Show symlink state, token expiry, and last modified time."""
    from .status import status as do_status

    do_status(agents=agents)


@app.command("docker-flags")
def docker_flags():
    """Output the -v flag string for docker."""
    from .status import get_docker_flags as do_get_docker_flags

    try:
        typer.echo(do_get_docker_flags())
    except FileNotFoundError:
        typer.echo("Error: manifest.json not found. Run `caw auth collect` first.", err=True)
        raise typer.Exit(1)
