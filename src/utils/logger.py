# utils/logger.py
from loguru import logger
import os
import sys

from config.settings import get_settings
settings = get_settings()

def setup_loguru_logger(log_file="logs/app.log", enable_file_logging=settings.enable_file_logging, level="INFO"):
    logger.remove()  # Remove default handlers

    # Console logging
    logger.add(sys.stdout, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # File logging (optional)
    if enable_file_logging:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, level=level, rotation="10 MB", retention="5 days", encoding="utf-8")

    logger.info("Loguru logger initialized.")

setup_loguru_logger()
