from loguru import logger
import sys

logger.debug("That's it, beautiful and simple logging!")

logger.remove()
logger.level("warning")
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("file_1.log", rotation="500 MB", level="DEBUG")

logger.debug("That's it, beautiful and simple logging!")   # will go to file_1.log (DEBUG) but not stderr (INFO+)
logger.info("Informational message")                       # will appear on both stderr and file
logger.warning("A warning")

