"""caw CLI — main entry point."""

from __future__ import annotations

import signal
import sys

import typer

from caw.auth.cli import app as auth_app

app = typer.Typer(
    name="caw",
    help="Coding Agent Wrapper — tools for managing coding agents.",
    no_args_is_help=True,
)

app.add_typer(auth_app, name="auth")


@app.command()
def viewer(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to."),
    port: int = typer.Option(0, "--port", "-p", help="Port to bind to (0 = auto)."),
):
    """Launch the trajectory viewer web UI."""
    from caw.viewer import start_viewer_server

    server = start_viewer_server(
        host=host,
        port=port or None,
    )
    typer.echo(f"Trajectory viewer running at {server.url}")
    typer.echo("Press Ctrl+C to stop.")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    app()


if __name__ == "__main__":
    main()
