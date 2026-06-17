import logging

from rich.logging import RichHandler

from .constant import LOGGING_LV, VERBOSE

logging.basicConfig(
    format="%(message)s",
    level=LOGGING_LV or VERBOSE,
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

def get_logger(name: str, logging_level: int | str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging_level or LOGGING_LV or VERBOSE)
    return logger
