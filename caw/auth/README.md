# caw auth — Credential Management for Docker Containers

Coding agents store OAuth credentials in home directory files (e.g., `~/.claude/.credentials.json`). When running agents inside Docker containers, token refresh creates new tokens (OAuth rotation), invalidating the host's tokens.

`caw auth` solves this by making `~/.caw/auth/` the canonical location for all credential files, mounting it read-write into containers, and using symlinks so writes propagate in real-time.

## How it works

```
HOST:                                         CONTAINER:

~/.claude/.credentials.json                   /home/playground/.claude/.credentials.json
    ↓ symlink                                     ↑ copy + inotify sync
~/.caw/auth/claude/credentials.json  ←—RW mount—→  /tmp/caw_auth/claude/credentials.json
```

On the host, credential files are replaced with symlinks into `~/.caw/auth/`. Inside the container, credentials are copied from the bind mount and kept in sync bidirectionally via an inotify-based guard.

## Usage

```bash
caw auth setup                        # collect + symlink all detected agents
caw auth setup --agents claude codex  # specific agents only
caw auth setup --force                # overwrite existing auth dir and backups

caw auth status                       # symlink state, token expiry, last modified

docker run $(caw auth docker-flags) -v ./project:/work my-image

caw auth teardown                     # restore original files from backups
caw auth teardown --dry-run           # preview what would be restored
```

## Commands

### `caw auth setup`

Reads credentials from `~/.claude/`, `~/.codex/`, `~/.gemini/`, `~/.config/cursor/`. Writes canonical files to `~/.caw/auth/`, generates `manifest.json` and `setup-container.sh`, then replaces host credential files with symlinks.

- Credential files (tokens, OAuth) are marked `strategy: symlink` — shared read-write
- Config files (.claude.json, config.toml) are marked `strategy: copy` — cleaned/stripped for containers

### `caw auth teardown`

Restores original credential files from `~/.caw/auth/.backups/`.

### `caw auth status`

Shows a table with symlink state, token expiry, and last modified time for all managed files.

### `caw auth docker-flags`

Outputs the `-v` flag for Docker:

```bash
$ caw auth docker-flags
-v /home/user/.caw/auth:/tmp/caw_auth:rw
```

## Container setup

The generated `setup-container.sh` runs inside the container (called from your entrypoint). It reads `manifest.json`, copies credentials, and starts a bidirectional inotify guard for credential sync.

```bash
# In your entrypoint.sh:
if [ -f /tmp/caw_auth/setup-container.sh ]; then
    /tmp/caw_auth/setup-container.sh /tmp/caw_auth /home/playground playground
fi
```

Requires `jq` in the container image. `inotify-tools` is installed automatically if not present.

## Directory structure

```
~/.caw/auth/
├── manifest.json              # file map + metadata
├── setup-container.sh         # POSIX script for container setup
├── .backups/                  # originals before symlinking
├── claude/
│   ├── credentials.json       # credential (symlinked)
│   └── config.json            # cleaned .claude.json (copied)
├── codex/
│   ├── auth.json              # credential (symlinked)
│   └── config.toml            # cleaned config (copied)
├── gemini/
│   ├── oauth_creds.json       # credential (symlinked)
│   └── ...                    # config files (copied)
└── cursor/
    ├── auth.json              # credential (symlinked)
    └── cli-config.json        # cleaned config (copied)
```

## Supported agents

| Agent | Credential files | Config files |
|-------|-----------------|--------------|
| Claude Code | `.claude/.credentials.json` | `.claude.json` (stripped to essential keys) |
| Codex | `.codex/auth.json` | `.codex/config.toml` (local trust removed) |
| Gemini CLI | `.gemini/oauth_creds.json` | `google_accounts.json`, `settings.json`, `installation_id` |
| Cursor | `.config/cursor/auth.json` | `.cursor/cli-config.json` (stripped) |

## Programmatic API

```python
from caw.auth import setup, teardown, get_status, get_docker_flags

setup(agents=["claude"])
statuses = get_status()
flags = get_docker_flags()
teardown()
```

See [`examples/auth.py`](../../examples/auth.py) for a full example.

## Known limitation

OAuth token rotation means a refresh returns a new refresh token, invalidating the old one. If two processes refresh simultaneously, one gets an invalid token. Don't run the same agent identity in two places at once.
