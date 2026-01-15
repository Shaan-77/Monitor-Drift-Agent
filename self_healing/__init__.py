"""
Self-healing module for Monitor/Drift Agent.

This module provides functionality for automatic remediation actions
such as scaling down resources or shutting down services when
cost spikes or anomalies are detected.
"""

from .auto_scaling import *

__all__ = ['auto_scaling']
