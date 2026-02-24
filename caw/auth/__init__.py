"""caw.auth — Credential management for Docker containers."""

from .collector import AUTH_DIR, setup
from .linker import teardown
from .manifest import AgentManifest, Manifest, ManifestFile
from .providers import PROVIDERS, AgentAuthProvider, CollectedFile
from .status import AuthFileStatus, get_docker_flags, get_status, status

__all__ = [
    "AUTH_DIR",
    "AgentAuthProvider",
    "AgentManifest",
    "AuthFileStatus",
    "CollectedFile",
    "Manifest",
    "ManifestFile",
    "PROVIDERS",
    "get_docker_flags",
    "get_status",
    "setup",
    "status",
    "teardown",
]
