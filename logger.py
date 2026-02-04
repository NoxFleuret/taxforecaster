"""
Structured logging module for TaxForecaster application.

Provides consistent logging across all components with file rotation,
different log levels, and formatted output.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional
import os


class TaxForecasterLogger:
    """Custom logger for TaxForecaster application"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_level: str = "INFO") -> logging.Logger:
        """
        Get or create a logger instance
        
        Args:
            name: Logger name (typically module name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # File handler (DEBUG and above)
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"taxforecaster_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
        
        cls._loggers[name] = logger
        return logger


def get_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name
        level: Log level
        
    Returns:
        Logger instance
        
    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting forecast process")
    """
    return TaxForecasterLogger.get_logger(name, level)


def log_operation(
    logger: logging.Logger,
    operation: str,
    status: str = "started",
    **kwargs
):
    """
    Log an operation with structured data
    
    Args:
        logger: Logger instance
        operation: Operation name
        status: Operation status (started, completed, failed)
        **kwargs: Additional context data
    """
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    message = f"[{operation.upper()}] {status}"
    
    if context:
        message += f" | {context}"
    
    if status == "failed":
        logger.error(message)
    elif status == "completed":
        logger.info(message)
    else:
        logger.debug(message)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_seconds: float,
    **kwargs
):
    """
    Log performance metrics
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_seconds: Time taken
        **kwargs: Additional metrics
    """
    metrics = f"duration={duration_seconds:.2f}s"
    
    for key, value in kwargs.items():
        metrics += f" | {key}={value}"
    
    logger.info(f"[PERFORMANCE] {operation} | {metrics}")


# Example usage
if __name__ == "__main__":
    # Test logger
    logger = get_logger("test_module", "DEBUG")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging
    log_operation(
        logger,
        "data_loading",
        status="started",
        file="tax_history.csv",
        rows=100
    )
    
    log_operation(
        logger,
        "data_loading",
        status="completed",
        file="tax_history.csv",
        rows=100
    )
    
    log_performance(
        logger,
        "model_training",
        duration_seconds=45.3,
        models=15,
        accuracy=92.5
    )
