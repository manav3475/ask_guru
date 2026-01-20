import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger("ASK_GURU")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = logging.FileHandler(log_dir / "main.log")


# This is the format in which logs will be displayed in log file
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# assign the formatter to file_handler object
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add the handler to logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
