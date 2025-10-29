"""
env_utils: Load all environment variables.
"""
import os
from typing import Any, Callable, Optional

from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger("env_utils")


def getenv(name: str,
           default: Optional[Any] = None,
           required: bool = False,
           cast: Optional[Callable[[str], Any]] = None) -> Any:
    """
    Get environment variable with optional default, 'required' flag, and casting.
    - If required=True and not set: exit with a clear message.
    - If cast is provided, applies cast(value).
    """
    val = os.getenv(name, default)
    if val is None or val == "":
        if required:
            raise SystemExit(f"Missing {name}. Export it in local.env.")
        return default
    return cast(val) if cast else val
