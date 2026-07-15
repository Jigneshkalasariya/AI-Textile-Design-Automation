import sys
import logging
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    Default handler from python logging to loguru.
    See: https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 0
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging(debug: bool = True) -> None:
    # Remove default handler
    logger.remove()

    # Add console logger
    log_level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stdout,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add file logger
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="10 days",
        level="INFO",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # Intercept all logging from other libraries
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set levels for libraries we don't want too verbose
    for name in ["urllib3", "boto3", "botocore", "celery", "pika"]:
        logging.getLogger(name).setLevel(logging.WARNING)

# Initialize logger right away
setup_logging(debug=True)
