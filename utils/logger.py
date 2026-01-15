"""
Custom logging utility for Monitor/Drift Agent.

This module provides a custom logger wrapper with additional
functionality for structured logging and context management.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime


class CustomLogger:
    """Custom logger with additional functionality."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize custom logger.
        
        Args:
            name: Logger name
            logger: Optional existing logger instance
        """
        self.name = name
        self.logger = logger or logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """
        Set context variables for logging.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context variables."""
        self.context = {}
    
    def _format_message(self, message: str) -> str:
        """
        Format message with context.
        
        Args:
            message: Log message
        
        Returns:
            Formatted message string
        
        TODO: Implement message formatting with context
        """
        if self.context:
            context_str = " | ".join([f"{k}={v}" for k, v in self.context.items()])
            return f"{message} | Context: {context_str}"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message), **kwargs)
    
    def log_metric(self, metric_name: str, value: float, **metadata):
        """
        Log a metric with structured data.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            **metadata: Additional metadata
        """
        message = f"Metric: {metric_name}={value}"
        if metadata:
            metadata_str = " | ".join([f"{k}={v}" for k, v in metadata.items()])
            message = f"{message} | {metadata_str}"
        self.info(message)
    
    def log_alert(self, alert_type: str, severity: str, message: str, **metadata):
        """
        Log an alert with structured data.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            **metadata: Additional metadata
        """
        log_message = f"Alert [{severity.upper()}] {alert_type}: {message}"
        if metadata:
            metadata_str = " | ".join([f"{k}={v}" for k, v in metadata.items()])
            log_message = f"{log_message} | {metadata_str}"
        
        # Use appropriate log level based on severity
        severity_levels = {
            "low": self.info,
            "medium": self.warning,
            "high": self.error,
            "critical": self.critical
        }
        log_func = severity_levels.get(severity.lower(), self.warning)
        log_func(log_message)


def get_logger(name: str) -> CustomLogger:
    """
    Get a custom logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        CustomLogger instance
    
    TODO: Implement logger factory
    """
    return CustomLogger(name)
