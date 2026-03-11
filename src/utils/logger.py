import logging
from pathlib import Path


def setup_logging(dir: Path) -> logging.Logger:
    """Setup logging to file and console."""

    log_file = dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)
