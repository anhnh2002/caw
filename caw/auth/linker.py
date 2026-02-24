"""Link/unlink credential files between original locations and ~/.caw/auth/."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from .manifest import Manifest

console = Console()

AUTH_DIR = Path.home() / ".caw" / "auth"
BACKUPS_DIR = AUTH_DIR / ".backups"


def _load_manifest(auth_dir: Path | None = None) -> tuple[Manifest, Path]:
    """Load manifest from auth_dir. Returns (manifest, resolved_auth_dir)."""
    resolved = auth_dir if auth_dir else AUTH_DIR
    manifest_path = resolved / "manifest.json"
    if not manifest_path.exists():
        console.print("[red]Error: manifest.json not found. Run `caw auth setup` first.[/red]")
        raise SystemExit(1)
    return Manifest.load(manifest_path), resolved


def link(
    agents: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    auth_dir: Path | None = None,
) -> None:
    """Replace host credential files with symlinks to the auth directory.

    Only links files with type=credential and strategy=symlink.
    Backs up originals to <auth_dir>/.backups/ first.

    Args:
        agents: Agent names to link, or None for all.
        dry_run: Show what would be done without making changes.
        force: Overwrite existing backups.
        auth_dir: Custom auth directory. Defaults to ~/.caw/auth/.
    """
    manifest, resolved_dir = _load_manifest(auth_dir)
    backups_dir = resolved_dir / ".backups"
    host_home = Path(manifest.host_home)

    console.print("[bold]Linking credential files...[/bold]\n")

    # Filter agents
    agent_names = set(agents) if agents and "all" not in agents else set(manifest.agents.keys())

    linked = 0
    skipped = 0

    for agent_name, agent_manifest in manifest.agents.items():
        if agent_name not in agent_names:
            continue

        for mf in agent_manifest.files:
            if mf.type != "credential" or mf.strategy != "symlink":
                continue

            canonical = resolved_dir / mf.src
            original = host_home / mf.host_original

            if not canonical.exists():
                console.print(f"  [yellow]Skip {mf.host_original}:[/yellow] canonical file not found at {canonical}")
                skipped += 1
                continue

            # Check if already symlinked correctly
            if original.is_symlink() and original.resolve() == canonical.resolve():
                console.print(f"  [dim]Already linked: {mf.host_original} -> {canonical}[/dim]")
                skipped += 1
                continue

            if dry_run:
                console.print(f"  [cyan]Would link:[/cyan] {mf.host_original} -> {canonical}")
                linked += 1
                continue

            # Backup original if it exists and is not already a symlink
            if original.exists() and not original.is_symlink():
                backup_path = backups_dir / mf.host_original
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                if backup_path.exists() and not force:
                    console.print(
                        f"  [yellow]Backup already exists for {mf.host_original}.[/yellow] Use --force to overwrite."
                    )
                    skipped += 1
                    continue
                shutil.copy2(str(original), str(backup_path))
                console.print(f"  [dim]Backed up: {mf.host_original} -> .backups/{mf.host_original}[/dim]")

            # Remove original and create symlink
            if original.exists() or original.is_symlink():
                original.unlink()
            original.parent.mkdir(parents=True, exist_ok=True)
            original.symlink_to(canonical)
            console.print(f"  [green]Linked:[/green] {mf.host_original} -> {canonical}")
            linked += 1

    action = "Would link" if dry_run else "Linked"
    console.print(f"\n{action} {linked} file(s), skipped {skipped}.")


def teardown(
    agents: list[str] | None = None,
    dry_run: bool = False,
    auth_dir: Path | None = None,
) -> None:
    """Restore original credential files from backups.

    Removes symlinks and copies backups back to original locations.

    Args:
        agents: Agent names to restore, or None for all.
        dry_run: Show what would be done without making changes.
        auth_dir: Custom auth directory. Defaults to ~/.caw/auth/.
    """
    manifest, resolved_dir = _load_manifest(auth_dir)
    backups_dir = resolved_dir / ".backups"
    host_home = Path(manifest.host_home)

    console.print("[bold]Unlinking credential files...[/bold]\n")

    agent_names = set(agents) if agents and "all" not in agents else set(manifest.agents.keys())

    restored = 0
    skipped = 0

    for agent_name, agent_manifest in manifest.agents.items():
        if agent_name not in agent_names:
            continue

        for mf in agent_manifest.files:
            if mf.type != "credential" or mf.strategy != "symlink":
                continue

            original = host_home / mf.host_original
            backup_path = backups_dir / mf.host_original

            # Check if it's currently a symlink pointing to our canonical
            canonical = resolved_dir / mf.src
            if original.is_symlink() and original.resolve() == canonical.resolve():
                if backup_path.exists():
                    if dry_run:
                        console.print(f"  [cyan]Would restore:[/cyan] {mf.host_original} from backup")
                        restored += 1
                        continue

                    original.unlink()
                    shutil.copy2(str(backup_path), str(original))
                    console.print(f"  [green]Restored:[/green] {mf.host_original} from backup")
                    restored += 1
                else:
                    if dry_run:
                        console.print(f"  [cyan]Would copy canonical:[/cyan] {mf.host_original} (no backup found)")
                        restored += 1
                        continue

                    # No backup — copy the canonical file as a regular file
                    original.unlink()
                    shutil.copy2(str(canonical), str(original))
                    console.print(f"  [green]Restored:[/green] {mf.host_original} from canonical (no backup)")
                    restored += 1
            else:
                console.print(f"  [dim]Skip {mf.host_original}: not a symlink to our canonical[/dim]")
                skipped += 1

    action = "Would restore" if dry_run else "Restored"
    console.print(f"\n{action} {restored} file(s), skipped {skipped}.")
