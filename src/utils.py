# src/utils.py
import logging
import sys

def setup_logging():
    """
    Sets up the logging configuration for the application.
    Logs are output to the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("ForexBot")

# Create a global logger instance
logger = setup_logging()
