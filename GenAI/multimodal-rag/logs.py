import logging
import datetime

now = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")

log_format = "%(asctime)s [%(levelname)s]  %(message)s"
log_d_format = "%m/%d/%Y %I:%M:%S %p"

logging.basicConfig(
    format=log_format,
    datefmt=log_d_format,
    level=logging.INFO,
)

log_formatter = logging.Formatter(log_format, datefmt=log_d_format)
logger = logging.getLogger()

file_handler = logging.FileHandler(f"execution-{now}.log")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
