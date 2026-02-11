"""caw CLI — main entry point."""

from __future__ import annotations

import typer

from caw.auth.cli import app as auth_app

app = typer.Typer(
    name="caw",
    help="Coding Agent Wrapper — tools for managing coding agents.",
    no_args_is_help=True,
)

app.add_typer(auth_app, name="auth")


def main():
    app()


if __name__ == "__main__":
    main()
