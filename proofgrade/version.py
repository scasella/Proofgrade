from __future__ import annotations

import os
import subprocess
from functools import lru_cache

__version__ = "0.1.0"


@lru_cache(maxsize=1)
def get_git_sha() -> str | None:
    env_sha = os.getenv("GITHUB_SHA") or os.getenv("PROOFGRADE_GIT_SHA")
    if env_sha:
        return env_sha[:12]
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return output or None

