"""
Global logging configuration for the calibrion_ft package.

This module provides a centralized logging setup that can be used
throughout the package to ensure consistent logging behavior.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    log_level: int = logging.INFO,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and configuration.
    
    Args:
        name: Logger name. If None, uses the calling module's name.
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log format
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the caller's module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'calibrion_ft')
    
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers to the same logger
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. Creates one with default settings if it doesn't exist.
    
    Args:
        name: Logger name. If None, uses the calling module's name.
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get the caller's module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'calibrion_ft')
    
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def configure_package_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the entire calibrion_ft package.
    
    Args:
        level: Default logging level for the package
    """
    # Set up the root package logger
    package_logger = logging.getLogger('calibrion_ft')
    package_logger.setLevel(level)
    
    # Ensure we don't duplicate handlers
    if not package_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        package_logger.addHandler(handler)
        package_logger.propagate = False


# Configure package logging when module is imported
configure_package_logging()