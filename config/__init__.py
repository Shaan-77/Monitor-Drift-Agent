"""
Configuration module for Monitor/Drift Agent.

This module provides configuration management including settings
for thresholds, alert channels, and logging configuration.
"""

from .settings import *
from .logging_config import *

__all__ = ['settings', 'logging_config']
