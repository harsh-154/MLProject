# we will create a logger that will log to a file with the current date and time in the filename
# and will log messages in a specific format including the timestamp, line number, logger name, log level, and message.
import logging
import os
from datetime import datetime

LOG_FILE= f"{datetime.now().strftime('%m-%d_%Y_%H-%M-%S')}.log"
logs_path= os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# lets test it

# if __name__ == "__main__":
#     logging.info("Logging setup complete. This is an info message.")
