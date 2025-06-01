import logging
import os
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(
    name: str = "rag_system", 
    log_file: str = "logs/rag_system.log",
    level: int = logging.INFO,
    max_bytes: int = 10*1024*1024,
    backup_count: int = 5,
    enable_file_logging: bool = True  # NEW FLAG
) -> logging.Logger:
    # Create logs directory if file logging is enabled
    log_dir = os.path.dirname(log_file)
    if enable_file_logging and log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler (conditionally enabled)
    if enable_file_logging:
        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger '{name}' initialized - Level: {logging.getLevelName(level)}")
    return logger
