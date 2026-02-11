# caw

**Coding Agent Wrapper** — a Python library and CLI for orchestrating coding agents (Claude Code, Codex, Gemini CLI, Cursor) with a unified interface, MCP tool servers, and credential management for Docker containers.

## Install

```bash
pip install -e .
```

Requires Python 3.10+.

## Library: Unified Agent Interface

caw wraps multiple coding agent CLIs behind a single `Agent` / `Session` API. Swap providers without changing your code.

### Quick start

```python
from caw import Agent

agent = Agent()  # defaults to claude_code
traj = agent.completion("Explain what this repository does")
print(traj.result)
print(f"{traj.usage.total_tokens} tokens, ${traj.usage.cost_usd:.4f}")
```

### Multi-turn sessions

```python
from caw import Agent

agent = Agent(provider="claude_code", model="opus", reasoning="high")
agent.set_system_prompt("You are a security reviewer.")

with agent.start_session() as session:
    turn1 = session.send("Review src/auth.py for vulnerabilities")
    print(turn1.result)

    turn2 = session.send("Now check src/api.py")
    print(turn2.result)

# session.end() called automatically, returns full Trajectory
```

### Providers

| Provider | CLI | Provider name |
|----------|-----|---------------|
| Claude Code | `claude` | `claude_code` |
| Codex | `codex` | `codex` |

Set via constructor, environment variable, or at runtime:

```python
agent = Agent(provider="codex")
# or
os.environ["CAW_PROVIDER"] = "codex"
# or
agent.set_provider("codex")
```

### MCP tool servers

Attach MCP servers so the agent can call external tools:

```python
from caw import Agent, MCPServer

agent = Agent()
agent.add_mcp_server(MCPServer(
    name="my_db",
    command="python",
    args=["-m", "my_mcp_server"],
))
```

### ToolKit: declarative tool servers

Define tools as Python classes. caw spins up an HTTP MCP server automatically:

```python
from caw import Agent, ToolKit, tool

class UserDB(ToolKit, server_name="user_db"):
    def __init__(self):
        self.users = ["Alice", "Bob"]

    @tool(description="List all users")
    async def list_users(self) -> str:
        return ", ".join(self.users)

    @tool(description="Add a user")
    async def add_user(self, name: str) -> str:
        self.users.append(name)
        return f"Added {name}"

db = UserDB()
agent = Agent(system_prompt="You have access to a user database.")
agent.add_tool_server(db.as_server())

traj = agent.completion("Add Eve to the user database, then list all users")
```

### Subagents

Register child agents that the parent can invoke as tools:

```python
from caw import Agent, AgentSpec

reviewer = AgentSpec(
    name="security_reviewer",
    description="Reviews code for security issues",
    system_prompt="You are a security expert. Review the given code.",
)

agent = Agent()
agent.add_subagent(reviewer)
traj = agent.completion("Review the auth module for vulnerabilities")

# Subagent trajectories are captured:
for sub in traj.subagent_trajectories:
    print(f"  subagent: {sub.agent}, {sub.num_turns} turns")
```

### Data models

Every interaction produces a `Trajectory` with structured data:

```
Trajectory
├── agent, model, session_id, created_at
├── turns: list[Turn]
│   ├── input: str
│   ├── output: list[TextBlock | ThinkingBlock | ToolUse]
│   │   └── ToolUse.subagent_trajectory: Trajectory | None
│   ├── usage: UsageStats
│   └── duration_ms: int
├── usage: UsageStats (own)
└── total_usage: UsageStats (own + all nested subagents)
```

Sessions are persisted to JSONL in `caw_data/` by default.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `CAW_PROVIDER` | Default provider (`claude_code`, `codex`) |
| `CAW_MODEL` | Default model name |
| `CAW_EFFORT` | Default reasoning effort (`high`, `medium`, `low`) |

---

## CLI: `caw auth` — Credential Management for Docker Containers

Coding agents store OAuth credentials in home directory files (e.g., `~/.claude/.credentials.json`). When running agents inside Docker containers, token refresh creates new tokens (OAuth rotation), invalidating the host's tokens.

`caw auth` solves this by making `~/.caw/auth/` the canonical location for all credential files, mounting it read-write into containers, and using symlinks so writes propagate in real-time.

### How it works

```
HOST:                                         CONTAINER:

~/.claude/.credentials.json                   /home/playground/.claude/.credentials.json
    ↓ symlink                                     ↓ symlink
~/.caw/auth/claude/credentials.json  ←—RW mount—→  /tmp/caw_auth/claude/credentials.json
```

When any process (host or container) refreshes a token, the write goes through the symlink to the canonical file in `~/.caw/auth/`, visible everywhere immediately.

### Usage

```bash
# Collect credentials from all detected agents, then symlink
caw auth collect --link

# Or step by step:
caw auth collect                  # gather into ~/.caw/auth/
caw auth link                     # replace originals with symlinks

# Check status
caw auth status

# Run a container with credentials
docker run $(caw auth docker-flags) -v ./project:/work my-image

# Restore original files
caw auth unlink
```

### Commands

#### `caw auth collect`

Reads credentials from `~/.claude/`, `~/.codex/`, `~/.gemini/`, `~/.config/cursor/`. Writes canonical files to `~/.caw/auth/`. Generates `manifest.json` and `setup-container.sh`.

- Credential files (tokens, OAuth) are marked `strategy: symlink` — shared read-write
- Config files (.claude.json, config.toml) are marked `strategy: copy` — cleaned/stripped for containers

```bash
caw auth collect                        # all agents
caw auth collect --agents claude codex  # specific agents
caw auth collect --force --link         # overwrite + symlink in one step
```

#### `caw auth link`

Backs up originals to `~/.caw/auth/.backups/`, then replaces credential files with symlinks to `~/.caw/auth/`.

```bash
caw auth link
caw auth link --dry-run    # preview changes
caw auth link --force      # overwrite existing backups
```

#### `caw auth unlink`

Restores original credential files from backups.

```bash
caw auth unlink
caw auth unlink --dry-run
```

#### `caw auth status`

Shows a table with symlink state, token expiry, and last modified time for all managed files.

#### `caw auth docker-flags`

Outputs the `-v` flag for Docker:

```bash
$ caw auth docker-flags
-v /home/user/.caw/auth:/tmp/caw_auth:rw
```

### Container setup

The generated `setup-container.sh` runs inside the container (called from your entrypoint). It reads `manifest.json` and creates symlinks/copies:

```bash
# In your entrypoint.sh:
if [ -f /tmp/caw_auth/setup-container.sh ]; then
    /tmp/caw_auth/setup-container.sh /tmp/caw_auth /home/playground playground
fi
```

Requires `jq` in the container image.

### Directory structure

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

### Supported agents

| Agent | Credential files | Config files |
|-------|-----------------|--------------|
| Claude Code | `.claude/.credentials.json` | `.claude.json` (stripped to essential keys) |
| Codex | `.codex/auth.json` | `.codex/config.toml` (local trust removed) |
| Gemini CLI | `.gemini/oauth_creds.json` | `google_accounts.json`, `settings.json`, `installation_id` |
| Cursor | `.config/cursor/auth.json` | `.cursor/cli-config.json` (stripped) |

### Known limitation

OAuth token rotation means a refresh returns a new refresh token, invalidating the old one. If two processes refresh simultaneously, one gets an invalid token. Don't run the same agent identity in two places at once.

## License

MIT
