import logging
import os

CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
SEED = int(os.getenv("SEED", "42"))


def extract_from_valid_names(
    name: str, valid_names: list[str] = ["train", "val", "test"]
) -> str:
    for name in valid_names:
        if name in name:
            return name

    raise ValueError(f"None of valid names {valid_names} found in {name!r}")


def is_env_enabled(env_var: str, default: str = "") -> bool:
    return os.getenv(env_var, default).lower() in ("1", "true", "yes")


VERBOSE = logging.INFO if is_env_enabled("VERBOSE") else logging.WARNING
LOGGING_LV = None
if _logging_lv_env_var := os.getenv("LOGGING_LV"):
    LOGGING_LV = int(_logging_lv_env_var)
