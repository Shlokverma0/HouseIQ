"""
logger.py
---------
Centralized logger for the entire application.
"""
import logging
import sys
import os

# Logger configure karo
logger = logging.getLogger("houseiq")
logger.setLevel(logging.INFO)

# Duplicate handlers se bachne ke liye
if not logger.handlers:
    # Console handler (terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler (logs/app.log me save)
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Formatting
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
