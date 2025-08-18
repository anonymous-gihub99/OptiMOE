# ===================================
# utils/logger.py
"""
Logging configuration for Antashiri System
Provides centralized logging with multiple outputs
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    debug: bool = False,
    format_string: Optional[str] = None
):
    """
    Setup logging configuration for Antashiri
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        debug: Enable debug mode with verbose output
        format_string: Custom format string for logs
    """
    
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        if debug:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        else:
            format_string = (
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<level>{message}</level>"
            )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=debug,
        diagnose=debug
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string.replace('<green>', '').replace('</green>', '')
                                .replace('<cyan>', '').replace('</cyan>', '')
                                .replace('<level>', '').replace('</level>', ''),
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=debug,
            diagnose=debug
        )
        
        logger.info(f"Logging to file: {log_file}")
    
    # Set logging level
    logger.info(f"Logging initialized - Level: {level}, Debug: {debug}")
    
    return logger


def get_logger(name: str):
    """
    Get a logger instance with specific name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Performance logging decorator
def log_performance(func):
    """Decorator to log function performance"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed:.3f}s: {e}")
            raise
    
    return wrapper


# Error logging decorator
def log_errors(func):
    """Decorator to log function errors"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.exception("Full traceback:")
            raise
    
    return wrapper


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = Config()
    
    print("System Configuration:")
    print(f"  Name: {config.get('system.name')}")
    print(f"  Version: {config.get('system.version')}")
    print(f"  Debug: {config.get('system.debug')}")
    print(f"  Emotion Model: {config.get('models.emotion_detector.model_name')}")
    print(f"  Whisper Model: {config.get('models.whisper.model_size')}")
    print(f"  Server Port: {config.get('server.port')}")
    
    # Test logging
    setup_logging(level="DEBUG", debug=True)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    @log_performance
    def test_function():
        import time
        time.sleep(0.5)
        return "Success"
    
    result = test_function()
    logger.success(f"Test completed: {result}")