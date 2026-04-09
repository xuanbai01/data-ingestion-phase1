import logging
import os
from datetime import datetime

LOG_DIR = "logs"


def setup_logger():
    """Set up logger that writes to both console and file."""
    os.makedirs(LOG_DIR, exist_ok=True)

    log_filename = os.path.join(
        LOG_DIR,
        f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Writing to {log_filename}")
    return logger