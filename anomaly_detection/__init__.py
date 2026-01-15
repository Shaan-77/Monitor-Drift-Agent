"""
Anomaly detection module for Monitor/Drift Agent.

This module provides functionality to detect anomalies in system metrics
and cloud usage patterns using threshold-based and machine learning methods.
"""

from .threshold_detection import *
from .alert_trigger import *

# Make machine_learning import optional (requires numpy/scikit-learn)
try:
    from .machine_learning import *
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    # Don't fail if ML dependencies are missing

__all__ = ['threshold_detection', 'alert_trigger']
