"""
Logging configuration for Decision Intelligence Studio
"""
import sys
import logging
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Add custom handlers
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file handler for errors
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logger.add(
    log_dir / "errors.log",
    rotation="10 MB",
    retention="7 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} - {message}"
)

# Add file handler for all logs
logger.add(
    log_dir / "pipeline.log",
    rotation="50 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
)


def get_logger(name: str):
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Configure standard logging to use loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Intercept standard logging
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)