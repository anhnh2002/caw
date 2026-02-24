"""Display status of auth files — symlink state, token expiry, last modified."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .manifest import Manifest

console = Console()

AUTH_DIR = Path.home() / ".caw" / "auth"


@dataclass
class AuthFileStatus:
    """Status of a single managed auth file."""

    agent: str
    file: str  # host_original relative path
    type: str  # "credential" or "config"
    strategy: str  # "symlink" or "copy"
    symlink_state: str  # "linked", "wrong_target", "not_linked", "missing", "n/a"
    exists: bool  # whether the canonical file exists in auth dir
    token_expiry: str | None  # human-readable token info, or None


def _check_token_expiry(auth_dir: Path, agent_name: str) -> str | None:
    """Check token expiry for known agents. Returns human-readable status or None."""
    try:
        if agent_name == "claude":
            cred_path = auth_dir / "claude" / "credentials.json"
            if cred_path.exists():
                with open(cred_path) as f:
                    creds = json.load(f)
                expires_at = creds.get("claudeAiOauth", {}).get("expiresAt")
                if expires_at:
                    dt = datetime.fromtimestamp(expires_at / 1000, tz=timezone.utc)
                    now = datetime.now(timezone.utc)
                    if dt < now:
                        delta = now - dt
                        return f"EXPIRED ({_format_delta(delta)} ago)"
                    else:
                        delta = dt - now
                        return f"valid ({_format_delta(delta)} remaining)"
    except Exception:
        pass
    return None


def _format_delta(delta) -> str:
    """Format a timedelta to a human-readable string."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        return f"{total_seconds // 60}m"
    elif total_seconds < 86400:
        return f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m"
    else:
        return f"{total_seconds // 86400}d {(total_seconds % 86400) // 3600}h"


def _format_mtime(path: Path) -> str:
    """Format last modified time of a file."""
    try:
        # Resolve symlinks to get the actual file's mtime
        real_path = path.resolve()
        mtime = real_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        return f"{_format_delta(delta)} ago"
    except Exception:
        return "unknown"


def get_status(
    agents: list[str] | None = None,
    auth_dir: str | Path | None = None,
) -> list[AuthFileStatus]:
    """Return structured status of all managed auth files.

    Args:
        agents: Agent names to include, or None for all.
        auth_dir: Custom auth directory. Defaults to ~/.caw/auth/.

    Returns:
        List of AuthFileStatus for each managed file.

    Raises:
        FileNotFoundError: If the manifest.json doesn't exist in auth_dir.
    """
    resolved_dir = Path(auth_dir) if auth_dir else AUTH_DIR
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found at {manifest_path}")

    manifest = Manifest.load(manifest_path)
    host_home = Path(manifest.host_home)
    agent_names = set(agents) if agents and "all" not in agents else set(manifest.agents.keys())

    results: list[AuthFileStatus] = []
    for agent_name, agent_manifest in manifest.agents.items():
        if agent_name not in agent_names:
            continue

        token_info = _check_token_expiry(resolved_dir, agent_name)

        for mf in agent_manifest.files:
            canonical = resolved_dir / mf.src
            original = host_home / mf.host_original

            # Determine symlink state
            if mf.strategy == "symlink":
                if original.is_symlink():
                    if original.resolve() == canonical.resolve():
                        symlink_state = "linked"
                    else:
                        symlink_state = "wrong_target"
                elif original.exists():
                    symlink_state = "not_linked"
                else:
                    symlink_state = "missing"
            else:
                symlink_state = "n/a"

            results.append(
                AuthFileStatus(
                    agent=agent_name,
                    file=mf.host_original,
                    type=mf.type,
                    strategy=mf.strategy,
                    symlink_state=symlink_state,
                    exists=canonical.exists(),
                    token_expiry=token_info if mf.type == "credential" else None,
                )
            )

    return results


def get_docker_flags(auth_dir: str | Path | None = None) -> str:
    """Return the Docker ``-v`` flag for mounting the auth directory.

    Args:
        auth_dir: Custom auth directory. Defaults to ~/.caw/auth/.

    Returns:
        A string like ``-v /path/to/auth:/tmp/caw_auth:rw``.

    Raises:
        FileNotFoundError: If the manifest.json doesn't exist in auth_dir.
    """
    resolved_dir = Path(auth_dir) if auth_dir else AUTH_DIR
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found at {manifest_path}")

    manifest = Manifest.load(manifest_path)
    return f"-v {resolved_dir}:{manifest.mount_point}:rw"


def status(agents: list[str] | None = None, auth_dir: str | Path | None = None) -> None:
    """Show status of all managed auth files.

    Args:
        agents: Agent names to show, or None for all.
        auth_dir: Custom auth directory. Defaults to ~/.caw/auth/.
    """
    resolved_dir = Path(auth_dir) if auth_dir else AUTH_DIR
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.exists():
        console.print("[yellow]No auth directory found.[/yellow] Run `caw auth setup` first.")
        return

    manifest = Manifest.load(manifest_path)
    host_home = Path(manifest.host_home)

    agent_names = set(agents) if agents and "all" not in agents else set(manifest.agents.keys())

    table = Table(title="caw auth status", show_lines=True)
    table.add_column("Agent", style="bold")
    table.add_column("File", style="dim")
    table.add_column("Type")
    table.add_column("Strategy")
    table.add_column("Symlink State")
    table.add_column("Last Modified")
    table.add_column("Token")

    for agent_name, agent_manifest in manifest.agents.items():
        if agent_name not in agent_names:
            continue

        token_info = _check_token_expiry(resolved_dir, agent_name)

        for i, mf in enumerate(agent_manifest.files):
            canonical = resolved_dir / mf.src
            original = host_home / mf.host_original

            # Check symlink state
            if mf.strategy == "symlink":
                if original.is_symlink():
                    target = os.readlink(str(original))
                    if original.resolve() == canonical.resolve():
                        symlink_state = "[green]linked[/green]"
                    else:
                        symlink_state = f"[yellow]wrong target[/yellow] ({target})"
                elif original.exists():
                    symlink_state = "[yellow]not linked[/yellow] (regular file)"
                else:
                    symlink_state = "[red]missing[/red]"
            else:
                symlink_state = "[dim]n/a (copy)[/dim]"

            # Last modified
            mtime = _format_mtime(canonical) if canonical.exists() else "[red]missing[/red]"

            # Token info only on first row per agent
            token_col = token_info if (i == 0 and token_info) else ""

            table.add_row(
                agent_name if i == 0 else "",
                mf.host_original,
                mf.type,
                mf.strategy,
                symlink_state,
                mtime,
                token_col,
            )

    console.print(table)

    # Docker flags hint
    console.print(f"\n[dim]Docker mount flag: -v {resolved_dir}:{manifest.mount_point}:rw[/dim]")
