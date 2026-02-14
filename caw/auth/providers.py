"""Agent auth providers — knows where each agent stores credentials and config."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .manifest import ManifestFile

console = Console()

# Default container home directory
CONTAINER_HOME = "/home/playground"


@dataclass
class CollectedFile:
    """A file collected from the host, ready to be written to ~/.caw/auth/."""

    manifest_file: ManifestFile
    content: bytes


class AgentAuthProvider(ABC):
    """Base class for agent auth providers."""

    name: str

    @abstractmethod
    def validate(self, src_home: Path) -> list[str]:
        """Return list of missing required file paths (as strings)."""

    @abstractmethod
    def describe(self, src_home: Path) -> str:
        """Return a short account info summary."""

    @abstractmethod
    def collect(self, src_home: Path) -> list[CollectedFile]:
        """Collect cleaned auth/config files. Returns list of CollectedFile."""


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

CLAUDE_JSON_KEEP_KEYS = {
    "oauthAccount",
    "userID",
    "hasCompletedOnboarding",
    "lastOnboardingVersion",
    "numStartups",
    "installMethod",
    "firstStartTime",
    "claudeCodeFirstTokenDate",
    "s1mAccessCache",
    "passesEligibilityCache",
    "groveConfigCache",
    "sonnet45MigrationComplete",
    "opus45MigrationComplete",
    "opusProMigrationComplete",
    "thinkingMigrationComplete",
    "autoUpdates",
}


def _build_clean_claude_json(source: dict) -> dict:
    """Build a minimal .claude.json, keeping only essential keys."""
    clean = {k: source[k] for k in CLAUDE_JSON_KEEP_KEYS if k in source}
    clean["projects"] = {
        CONTAINER_HOME: {
            "allowedTools": [],
            "mcpContextUris": [],
            "mcpServers": {},
            "enabledMcpjsonServers": [],
            "disabledMcpjsonServers": [],
            "hasTrustDialogAccepted": False,
            "projectOnboardingSeenCount": 1,
            "hasClaudeMdExternalIncludesApproved": False,
            "hasClaudeMdExternalIncludesWarningShown": False,
            "exampleFiles": [],
            "lastTotalWebSearchRequests": 0,
        }
    }
    return clean


class ClaudeAuthProvider(AgentAuthProvider):
    name = "claude"

    def validate(self, src_home: Path) -> list[str]:
        missing = []
        if not (src_home / ".claude.json").exists():
            missing.append(str(src_home / ".claude.json"))
        if not (src_home / ".claude" / ".credentials.json").exists():
            missing.append(str(src_home / ".claude" / ".credentials.json"))
        return missing

    def describe(self, src_home: Path) -> str:
        try:
            with open(src_home / ".claude.json") as f:
                cfg = json.load(f)
            account = cfg.get("oauthAccount", {})
            email = account.get("emailAddress", "unknown")
            org = account.get("organizationName", "unknown")

            with open(src_home / ".claude" / ".credentials.json") as f:
                creds = json.load(f)
            expires_at = creds.get("claudeAiOauth", {}).get("expiresAt")
            parts = [f"Account: {email}", f"Org: {org}"]
            if expires_at:
                from datetime import datetime, timezone

                dt = datetime.fromtimestamp(expires_at / 1000, tz=timezone.utc)
                parts.append(f"Token expires: {dt.isoformat()}")
            return ", ".join(parts)
        except Exception:
            return "Could not read account info"

    def collect(self, src_home: Path) -> list[CollectedFile]:
        # credentials.json — credential, symlinked for token refresh write-back
        with open(src_home / ".claude" / ".credentials.json") as f:
            credentials = json.load(f)

        cred_file = CollectedFile(
            manifest_file=ManifestFile(
                src="claude/credentials.json",
                container_target=".claude/.credentials.json",
                host_original=".claude/.credentials.json",
                type="credential",
                strategy="symlink",
                mode="0600",
            ),
            content=json.dumps(credentials).encode(),
        )

        # config.json — cleaned .claude.json for containers, copied (not symlinked)
        with open(src_home / ".claude.json") as f:
            local_config = json.load(f)

        clean_config = _build_clean_claude_json(local_config)
        original_keys = len(local_config)
        clean_keys = len(clean_config)
        original_projects = len(local_config.get("projects", {}))
        console.print(f"  [dim]Stripped .claude.json: {original_keys} keys -> {clean_keys} keys[/dim]")
        console.print(f"  [dim]Stripped projects: {original_projects} entries -> 1 entry[/dim]")

        config_file = CollectedFile(
            manifest_file=ManifestFile(
                src="claude/config.json",
                container_target=".claude.json",
                host_original=".claude.json",
                type="config",
                strategy="copy",
                mode="0644",
            ),
            content=json.dumps(clean_config, indent=2).encode(),
        )

        return [cred_file, config_file]


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


def _build_clean_codex_config(source_toml: str) -> str:
    """Build a minimal config.toml for codex, stripping local project trust."""
    lines: list[str] = []
    skip_section = False

    for line in source_toml.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if stripped.startswith("[project_trust."):
                skip_section = True
                continue
            skip_section = False
        if skip_section:
            continue
        lines.append(line)

    lines.append("")
    lines.append(f'[project_trust."{CONTAINER_HOME}"]')
    lines.append('trust_mode = "full"')
    lines.append("")

    return "\n".join(lines)


class CodexAuthProvider(AgentAuthProvider):
    name = "codex"

    def validate(self, src_home: Path) -> list[str]:
        missing = []
        if not (src_home / ".codex" / "auth.json").exists():
            missing.append(str(src_home / ".codex" / "auth.json"))
        return missing

    def describe(self, src_home: Path) -> str:
        try:
            with open(src_home / ".codex" / "auth.json") as f:
                auth_data = json.load(f)
            has_token = bool(auth_data.get("tokens") or auth_data.get("token") or auth_data.get("access_token"))
            has_api_key = bool(auth_data.get("OPENAI_API_KEY"))
            parts = []
            if has_api_key:
                parts.append("API key present")
            if has_token:
                parts.append("OAuth tokens present")
            return ", ".join(parts) if parts else "Auth file found (no recognized keys)"
        except Exception:
            return "Could not read auth info"

    def collect(self, src_home: Path) -> list[CollectedFile]:
        files: list[CollectedFile] = []

        # auth.json — credential, symlinked
        with open(src_home / ".codex" / "auth.json", "rb") as f:
            auth_content = f.read()
        files.append(
            CollectedFile(
                manifest_file=ManifestFile(
                    src="codex/auth.json",
                    container_target=".codex/auth.json",
                    host_original=".codex/auth.json",
                    type="credential",
                    strategy="symlink",
                    mode="0600",
                ),
                content=auth_content,
            )
        )

        # config.toml — config, copied (cleaned)
        config_path = src_home / ".codex" / "config.toml"
        if config_path.exists():
            config_text = config_path.read_text()
            clean_config = _build_clean_codex_config(config_text)
            files.append(
                CollectedFile(
                    manifest_file=ManifestFile(
                        src="codex/config.toml",
                        container_target=".codex/config.toml",
                        host_original=".codex/config.toml",
                        type="config",
                        strategy="copy",
                        mode="0644",
                    ),
                    content=clean_config.encode(),
                )
            )
            console.print("  [dim]Stripped config.toml: local project trust -> container trust only[/dim]")

        return files


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiAuthProvider(AgentAuthProvider):
    name = "gemini"

    _CREDENTIAL_FILES = ["oauth_creds.json"]
    _CONFIG_FILES = ["google_accounts.json", "settings.json", "installation_id"]

    def validate(self, src_home: Path) -> list[str]:
        creds = src_home / ".gemini" / "oauth_creds.json"
        if not creds.exists():
            return [str(creds)]
        return []

    def describe(self, src_home: Path) -> str:
        try:
            accounts_path = src_home / ".gemini" / "google_accounts.json"
            if accounts_path.exists():
                with open(accounts_path) as f:
                    accounts = json.load(f)
                if isinstance(accounts, list) and accounts:
                    email = accounts[0].get("email", "unknown")
                    return f"Account: {email}"
            return "Credentials present"
        except Exception:
            return "Could not read account info"

    def collect(self, src_home: Path) -> list[CollectedFile]:
        files: list[CollectedFile] = []
        gemini_dir = src_home / ".gemini"

        # Credential files — symlinked
        for filename in self._CREDENTIAL_FILES:
            path = gemini_dir / filename
            if path.exists():
                files.append(
                    CollectedFile(
                        manifest_file=ManifestFile(
                            src=f"gemini/{filename}",
                            container_target=f".gemini/{filename}",
                            host_original=f".gemini/{filename}",
                            type="credential",
                            strategy="symlink",
                            mode="0600",
                        ),
                        content=path.read_bytes(),
                    )
                )

        # Config files — copied
        for filename in self._CONFIG_FILES:
            path = gemini_dir / filename
            if path.exists():
                files.append(
                    CollectedFile(
                        manifest_file=ManifestFile(
                            src=f"gemini/{filename}",
                            container_target=f".gemini/{filename}",
                            host_original=f".gemini/{filename}",
                            type="config",
                            strategy="copy",
                            mode="0600",
                        ),
                        content=path.read_bytes(),
                    )
                )

        found = [f.manifest_file.src.split("/")[-1] for f in files]
        console.print(f"  [dim]Collected {len(found)} files: {', '.join(found)}[/dim]")
        return files


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

CURSOR_CLI_CONFIG_KEEP_KEYS = {
    "authInfo",
    "permissions",
    "model",
    "approvalMode",
    "sandbox",
}


class CursorAuthProvider(AgentAuthProvider):
    name = "cursor"

    def validate(self, src_home: Path) -> list[str]:
        missing = []
        auth_json = src_home / ".config" / "cursor" / "auth.json"
        cli_config = src_home / ".cursor" / "cli-config.json"
        if not auth_json.exists() and not cli_config.exists():
            missing.append(str(auth_json))
            missing.append(str(cli_config))
        return missing

    def describe(self, src_home: Path) -> str:
        try:
            auth_path = src_home / ".config" / "cursor" / "auth.json"
            if auth_path.exists():
                with open(auth_path) as f:
                    auth_data = json.load(f)
                email = auth_data.get("email", auth_data.get("user", "unknown"))
                return f"Account: {email}"
            return "Credentials present"
        except Exception:
            return "Could not read account info"

    def collect(self, src_home: Path) -> list[CollectedFile]:
        files: list[CollectedFile] = []

        # .config/cursor/auth.json — credential, symlinked
        auth_path = src_home / ".config" / "cursor" / "auth.json"
        if auth_path.exists():
            files.append(
                CollectedFile(
                    manifest_file=ManifestFile(
                        src="cursor/auth.json",
                        container_target=".config/cursor/auth.json",
                        host_original=".config/cursor/auth.json",
                        type="credential",
                        strategy="symlink",
                        mode="0600",
                    ),
                    content=auth_path.read_bytes(),
                )
            )

        # .cursor/cli-config.json — config, copied (cleaned)
        cli_config_path = src_home / ".cursor" / "cli-config.json"
        if cli_config_path.exists():
            with open(cli_config_path) as f:
                full_config = json.load(f)
            clean = {k: full_config[k] for k in CURSOR_CLI_CONFIG_KEEP_KEYS if k in full_config}
            stripped = len(full_config) - len(clean)
            console.print(f"  [dim]Stripped cli-config.json: removed {stripped} keys, kept {len(clean)}[/dim]")
            files.append(
                CollectedFile(
                    manifest_file=ManifestFile(
                        src="cursor/cli-config.json",
                        container_target=".cursor/cli-config.json",
                        host_original=".cursor/cli-config.json",
                        type="config",
                        strategy="copy",
                        mode="0600",
                    ),
                    content=json.dumps(clean, indent=2).encode(),
                )
            )

        found = [f.manifest_file.src for f in files]
        console.print(f"  [dim]Files: {', '.join(found)}[/dim]")
        return files


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, AgentAuthProvider] = {
    p.name: p
    for p in [
        ClaudeAuthProvider(),
        CodexAuthProvider(),
        GeminiAuthProvider(),
        CursorAuthProvider(),
    ]
}
