"""CLI subcommands for `caw auth`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer


app = typer.Typer(help="Manage credentials for Docker containers.")


@app.command()
def setup(
    agents: Annotated[
        Optional[list[str]],
        typer.Option("--agents", "-a", help="Agents to include (claude, codex, gemini, cursor, or all)"),
    ] = None,
    source_home: Annotated[
        str,
        typer.Option("--source-home", help="Source home directory to read credentials from"),
    ] = str(Path.home()),
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing auth dir and backups")] = False,
):
    """Collect credentials from host into ~/.caw/auth/ and symlink them."""
    from .collector import setup as do_setup

    do_setup(agents=agents or ["all"], source_home=source_home, force=force)


@app.command()
def teardown(
    agents: Annotated[
        Optional[list[str]],
        typer.Option("--agents", "-a", help="Agents to restore"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show what would be done")] = False,
):
    """Restore original credential files from backups."""
    from .linker import teardown as do_teardown

    do_teardown(agents=agents, dry_run=dry_run)


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
        typer.echo("Error: manifest.json not found. Run `caw auth setup` first.", err=True)
        raise typer.Exit(1)
