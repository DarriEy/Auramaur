"""Least-privilege environments for untrusted analysis subprocesses."""

from __future__ import annotations

import os


# Claude needs its own authentication and a small amount of normal process
# plumbing. Venue keys, wallet keys, webhooks, and database paths are omitted.
_ANALYSIS_ENV_ALLOWLIST = {
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "NO_PROXY",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
}


def analysis_subprocess_env() -> dict[str, str]:
    """Return a new environment containing no trading or signing secrets."""
    return {
        name: value
        for name, value in os.environ.items()
        if name in _ANALYSIS_ENV_ALLOWLIST
    }
