"""caw.auth — Credential management for Docker containers."""

from .collector import AUTH_DIR, collect
from .linker import link, unlink
from .manifest import AgentManifest, Manifest, ManifestFile
from .providers import PROVIDERS, AgentAuthProvider, CollectedFile
from .status import status

__all__ = [
    "AUTH_DIR",
    "AgentAuthProvider",
    "AgentManifest",
    "CollectedFile",
    "Manifest",
    "ManifestFile",
    "PROVIDERS",
    "collect",
    "link",
    "status",
    "unlink",
]
