"""
Threshold-based anomaly detection logic.

This module provides functionality to detect anomalies in metrics
by comparing values against predefined thresholds.
"""

from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import time

# Import settings for threshold configuration
try:
    from config.settings import get_settings
except ImportError:
    def get_settings():
        return None

# Module-level tracking for threshold exceedance duration
# Key: (metric_name, resource_type), Value: start_timestamp (float - Unix timestamp)
_threshold_exceedance_times: Dict[tuple, float] = {}

# Module-level tracking for sustained cost spikes
# Key: resource_name, Value: dict with 'count' (int) and 'last_check_date' (date)
_sustained_cost_spike_tracking: Dict[str, Dict] = {}


class ThresholdRule:
    """Represents a threshold rule for anomaly detection."""
    
    def __init__(
        self,
        metric_name: str,
        threshold_value: float,
        comparison_operator: str,
        severity: str = "medium"
    ):
        """
        Initialize a threshold rule.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold_value: Threshold value to compare against
            comparison_operator: Comparison operator ('gt', 'lt', 'gte', 'lte', 'eq')
            severity: Severity level if threshold is breached
        """
        self.metric_name = metric_name
        self.threshold_value = threshold_value
        self.comparison_operator = comparison_operator
        self.severity = severity
    
    def evaluate(self, metric_value: float) -> bool:
        """
        Evaluate if metric value breaches the threshold.
        
        Args:
            metric_value: Current metric value
        
        Returns:
            True if threshold is breached, False otherwise
        
        TODO: Implement threshold evaluation logic
        """
        # TODO: Implement comparison logic based on operator
        operators = {
            'gt': lambda x, y: x > y,
            'lt': lambda x, y: x < y,
            'gte': lambda x, y: x >= y,
            'lte': lambda x, y: x <= y,
            'eq': lambda x, y: x == y
        }
        
        if self.comparison_operator not in operators:
            return False
        
        return operators[self.comparison_operator](metric_value, self.threshold_value)


class ThresholdDetector:
    """Detector for threshold-based anomaly detection."""
    
    def __init__(self, rules: List[ThresholdRule]):
        """
        Initialize threshold detector with rules.
        
        Args:
            rules: List of threshold rules to evaluate
        """
        self.rules = rules
    
    def detect(self, metrics: Dict) -> List[Dict]:
        """
        Detect anomalies in metrics based on threshold rules.
        
        Args:
            metrics: Dictionary containing metric values
        
        Returns:
            List of detected anomalies
        
        TODO: Implement anomaly detection logic
        """
        anomalies = []
        
        for rule in self.rules:
            if rule.metric_name in metrics:
                metric_value = metrics[rule.metric_name]
                if rule.evaluate(metric_value):
                    anomalies.append({
                        "metric_name": rule.metric_name,
                        "metric_value": metric_value,
                        "threshold_value": rule.threshold_value,
                        "comparison_operator": rule.comparison_operator,
                        "severity": rule.severity,
                        "timestamp": datetime.now().isoformat()
                    })
        
        return anomalies
    
    def add_rule(self, rule: ThresholdRule):
        """Add a new threshold rule."""
        self.rules.append(rule)
    
    def remove_rule(self, metric_name: str):
        """Remove threshold rules for a specific metric."""
        self.rules = [r for r in self.rules if r.metric_name != metric_name]


def create_threshold_detector(rules: List[Dict]) -> ThresholdDetector:
    """
    Create a threshold detector from rule definitions.
    
    Args:
        rules: List of rule dictionaries
    
    Returns:
        ThresholdDetector instance
    
    TODO: Implement detector creation from rule definitions
    """
    threshold_rules = []
    for rule_dict in rules:
        rule = ThresholdRule(
            metric_name=rule_dict.get("metric_name"),
            threshold_value=rule_dict.get("threshold_value"),
            comparison_operator=rule_dict.get("comparison_operator", "gt"),
            severity=rule_dict.get("severity", "medium")
        )
        threshold_rules.append(rule)
    
    return ThresholdDetector(threshold_rules)


def check_thresholds(
    cpu_usage: float,
    cloud_cost: float,
    cpu_threshold: Optional[float] = None,
    cost_threshold: Optional[float] = None,
    duration: Optional[int] = None,
    resource_type: str = "Server"
) -> List[Dict]:
    """
    Check if CPU usage or cloud cost exceeds thresholds and track duration.
    
    This function compares current CPU usage and cloud cost against predefined
    thresholds. It tracks if each metric exceeds its threshold for the required
    duration (e.g., 5 minutes) before triggering an alert. The tracking uses
    time.time() for efficient timestamp management. If a metric falls below its
    threshold before the duration requirement is met, the tracking is reset.
    
    Args:
        cpu_usage: Current CPU usage percentage (0-100)
        cloud_cost: Current cloud cost in dollars per day
        cpu_threshold: CPU usage threshold percentage (defaults to settings)
        cost_threshold: Cloud cost threshold in dollars per day (defaults to settings)
        duration: Duration in minutes for threshold to be sustained (defaults to settings)
        resource_type: Resource type identifier (Server, AWS, GCP, Azure)
    
    Returns:
        List of triggered alerts (empty if no thresholds exceeded)
    
    Raises:
        RuntimeError: If required dependencies are not available
        ValueError: If input values are invalid
    """
    triggered_alerts = []
    
    # Get thresholds from settings if not provided
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available. Cannot determine thresholds.")
    
    if cpu_threshold is None:
        cpu_threshold = settings.cpu_usage_threshold
    if cost_threshold is None:
        cost_threshold = settings.cloud_cost_threshold
    if duration is None:
        duration = settings.cpu_threshold_duration
    
    # Validate inputs
    if not isinstance(cpu_usage, (int, float)) or cpu_usage < 0 or cpu_usage > 100:
        raise ValueError(f"Invalid CPU usage value: {cpu_usage}. Must be between 0 and 100.")
    if not isinstance(cloud_cost, (int, float)) or cloud_cost < 0:
        raise ValueError(f"Invalid cloud cost value: {cloud_cost}. Must be non-negative.")
    if duration < 0:
        raise ValueError(f"Invalid duration value: {duration}. Must be non-negative.")
    
    # Use time.time() for efficient timestamp tracking
    current_timestamp = time.time()
    # Use datetime.utcnow() for alert timestamps (database compatibility)
    current_datetime = datetime.utcnow()
    
    # Track CPU threshold exceedance
    cpu_key = ("cpu_usage", resource_type)
    
    if cpu_usage > cpu_threshold:
        if cpu_key not in _threshold_exceedance_times:
            # Start tracking exceedance
            _threshold_exceedance_times[cpu_key] = current_timestamp
        else:
            # Check if duration requirement is met
            exceedance_start = _threshold_exceedance_times[cpu_key]
            exceedance_duration = (current_timestamp - exceedance_start) / 60.0  # Convert to minutes
            
            if exceedance_duration >= duration:
                # Duration requirement met, trigger alert
                try:
                    from anomaly_detection.alert_trigger import trigger_alert
                    success = trigger_alert("CPU Usage", cpu_usage, current_datetime, resource_type)
                    if success:
                        triggered_alerts.append({
                            "metric_name": "CPU Usage",
                            "value": cpu_usage,
                            "timestamp": current_datetime,
                            "resource_type": resource_type,
                            "threshold": cpu_threshold,
                            "duration_exceeded": exceedance_duration
                        })
                        # Reset tracking after alert is triggered
                        del _threshold_exceedance_times[cpu_key]
                except ImportError:
                    # alert_trigger module not available yet (will be implemented in Phase 2)
                    # For now, just log or continue
                    pass
                except Exception as e:
                    # Log error but don't crash
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.error(f"Error triggering CPU alert: {e}", exc_info=True)
                    except ImportError:
                        print(f"Error triggering CPU alert: {e}")
    else:
        # CPU usage is below threshold, reset tracking
        if cpu_key in _threshold_exceedance_times:
            del _threshold_exceedance_times[cpu_key]
    
    # Track cloud cost threshold exceedance (with duration requirement)
    cost_key = ("cloud_cost", resource_type)
    
    if cloud_cost > cost_threshold:
        if cost_key not in _threshold_exceedance_times:
            # Start tracking exceedance
            _threshold_exceedance_times[cost_key] = current_timestamp
        else:
            # Check if duration requirement is met
            exceedance_start = _threshold_exceedance_times[cost_key]
            exceedance_duration = (current_timestamp - exceedance_start) / 60.0  # Convert to minutes
            
            if exceedance_duration >= duration:
                # Duration requirement met, trigger alert
                try:
                    from anomaly_detection.alert_trigger import trigger_alert
                    success = trigger_alert("Cloud Cost", cloud_cost, current_datetime, resource_type)
                    if success:
                        triggered_alerts.append({
                            "metric_name": "Cloud Cost",
                            "value": cloud_cost,
                            "timestamp": current_datetime,
                            "resource_type": resource_type,
                            "threshold": cost_threshold,
                            "duration_exceeded": exceedance_duration
                        })
                        # Reset tracking after alert is triggered
                        del _threshold_exceedance_times[cost_key]
                except ImportError:
                    # alert_trigger module not available yet (will be implemented in Phase 2)
                    pass
                except Exception as e:
                    # Log error but don't crash
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.error(f"Error triggering cloud cost alert: {e}", exc_info=True)
                    except ImportError:
                        print(f"Error triggering cloud cost alert: {e}")
    else:
        # Cloud cost is below threshold, reset tracking
        if cost_key in _threshold_exceedance_times:
            del _threshold_exceedance_times[cost_key]
    
    return triggered_alerts


def compare_cost_to_historical(
    current_cost: float,
    resource_name: str,
    spike_threshold_percent: Optional[float] = None,
    lookback_days: Optional[int] = None
) -> Dict:
    """
    Compare real-time cloud cost to historical average and detect spikes.
    
    This function retrieves historical cost data from the database, calculates
    the average cost, and compares it to the current cost. If the current cost
    exceeds the historical average by more than the configured threshold (default 20%),
    it triggers an alert.
    
    Args:
        current_cost: Current real-time cloud cost
        resource_name: Name of the cloud resource (e.g., 'AWS EC2', 'S3 Storage')
        spike_threshold_percent: Percentage above historical average to trigger alert
                                 (defaults to settings.cost_spike_threshold_percent)
        lookback_days: Number of days to look back for historical data
                      (defaults to settings.historical_cost_lookback_days)
    
    Returns:
        Dictionary containing:
        {
            'spike_detected': bool,
            'current_cost': float,
            'historical_average': float,
            'spike_percentage': float,
            'alert_triggered': bool
        }
    
    Raises:
        RuntimeError: If settings not available
        ValueError: If input values are invalid
    """
    # Get thresholds from settings if not provided
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available. Cannot determine thresholds.")
    
    if spike_threshold_percent is None:
        spike_threshold_percent = settings.cost_spike_threshold_percent
    if lookback_days is None:
        lookback_days = settings.historical_cost_lookback_days
    
    # Validate inputs
    if not isinstance(current_cost, (int, float)) or current_cost < 0:
        raise ValueError(f"Invalid current_cost value: {current_cost}. Must be non-negative.")
    
    if not resource_name or not isinstance(resource_name, str):
        raise ValueError(f"Invalid resource_name: {resource_name}. Must be a non-empty string.")
    
    if not isinstance(spike_threshold_percent, (int, float)) or spike_threshold_percent < 0:
        raise ValueError(f"Invalid spike_threshold_percent: {spike_threshold_percent}. Must be non-negative.")
    
    if not isinstance(lookback_days, int) or lookback_days < 1:
        raise ValueError(f"Invalid lookback_days: {lookback_days}. Must be a positive integer.")
    
    # Initialize result dictionary
    result = {
        'spike_detected': False,
        'current_cost': current_cost,
        'historical_average': 0.0,
        'spike_percentage': 0.0,
        'alert_triggered': False
    }
    
    try:
        # Retrieve historical cost data
        from data_collection.database import get_historical_cost_data
        
        historical_costs = get_historical_cost_data(resource_name, lookback_days)
        
        # If no historical data, return early (no comparison possible)
        if not historical_costs or len(historical_costs) == 0:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"No historical cost data found for {resource_name}. Cannot compare.")
            except ImportError:
                print(f"No historical cost data found for {resource_name}. Cannot compare.")
            return result
        
        # Calculate historical average
        historical_average = sum(historical_costs) / len(historical_costs)
        result['historical_average'] = historical_average
        
        # Handle division by zero edge case
        if historical_average == 0:
            # If historical average is 0, any positive current cost is a spike
            if current_cost > 0:
                result['spike_detected'] = True
                result['spike_percentage'] = float('inf') if historical_average == 0 else 0.0
            else:
                result['spike_percentage'] = 0.0
        else:
            # Calculate spike percentage
            spike_percentage = ((current_cost - historical_average) / historical_average) * 100.0
            result['spike_percentage'] = spike_percentage
            
            # Check if current cost exceeds threshold
            threshold_multiplier = 1 + (spike_threshold_percent / 100.0)
            if current_cost > historical_average * threshold_multiplier:
                result['spike_detected'] = True
                
                # Trigger alert
                try:
                    from anomaly_detection.alert_trigger import trigger_cost_alert
                    alert_triggered = trigger_cost_alert(
                        current_cost=current_cost,
                        resource_name=resource_name,
                        timestamp=datetime.utcnow(),
                        alert_reason="Historical Average Exceeded"
                    )
                    result['alert_triggered'] = alert_triggered
                    
                    # Trigger auto-scaling if enabled
                    try:
                        # Use module-level get_settings to avoid UnboundLocalError
                        settings = get_settings()
                        if settings and settings.self_healing_enabled:
                            from self_healing.auto_scaling import scale_down_resources
                            scaling_result = scale_down_resources(
                                resource_name=resource_name,
                                current_cost=current_cost
                            )
                            result['scaling_action'] = scaling_result.get('action_taken')
                            result['scaling_status'] = scaling_result.get('status')
                    except (ImportError, AttributeError) as e:
                        # Self-healing not available - continue without scaling
                        try:
                            from utils.logger import get_logger
                            logger = get_logger(__name__)
                            logger.debug(f"Self-healing not available: {e}")
                        except ImportError:
                            pass
                except ImportError:
                    # Fallback to trigger_alert if trigger_cost_alert not available
                    try:
                        from anomaly_detection.alert_trigger import trigger_alert
                        alert_triggered = trigger_alert(
                            "Cloud Cost (Historical Spike)",
                            current_cost,
                            datetime.utcnow(),
                            resource_name
                        )
                        result['alert_triggered'] = alert_triggered
                    except ImportError:
                        try:
                            from utils.logger import get_logger
                            logger = get_logger(__name__)
                            logger.warning("Alert trigger functions not available")
                        except ImportError:
                            print("Alert trigger functions not available")
                except Exception as e:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.error(f"Error triggering cost alert: {e}", exc_info=True)
                    except ImportError:
                        print(f"Error triggering cost alert: {e}")
    
    except ImportError as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database module not available for historical cost comparison: {e}")
        except ImportError:
            print(f"Database module not available for historical cost comparison: {e}")
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error comparing cost to historical: {e}", exc_info=True)
        except ImportError:
            print(f"Error comparing cost to historical: {e}")
    
    return result


def monitor_sustained_spikes(
    resource_name: str,
    current_cost: float,
    threshold: Optional[float] = None,
    consecutive_days: Optional[int] = None
) -> Dict:
    """
    Monitor and track sustained cost spikes over consecutive days.
    
    This function tracks how many consecutive days a resource's cost exceeds
    a threshold. An alert is triggered only after the cost exceeds the threshold
    for the required number of consecutive days (e.g., 3 days).
    
    Args:
        resource_name: Name of the cloud resource (e.g., 'AWS EC2', 'S3 Storage')
        current_cost: Current real-time cloud cost
        threshold: Cost threshold in dollars per day (defaults to settings.cloud_cost_threshold)
        consecutive_days: Number of consecutive days required (defaults to settings.sustained_cost_spike_days)
    
    Returns:
        Dictionary containing:
        {
            'spike_detected': bool,
            'consecutive_days': int,
            'threshold': float,
            'current_cost': float,
            'alert_triggered': bool
        }
    
    Raises:
        RuntimeError: If settings not available
        ValueError: If input values are invalid
    """
    # Get thresholds from settings if not provided
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available. Cannot determine thresholds.")
    
    if threshold is None:
        threshold = settings.cloud_cost_threshold
    if consecutive_days is None:
        consecutive_days = settings.sustained_cost_spike_days
    
    # Validate inputs
    if not isinstance(current_cost, (int, float)) or current_cost < 0:
        raise ValueError(f"Invalid current_cost value: {current_cost}. Must be non-negative.")
    
    if not resource_name or not isinstance(resource_name, str):
        raise ValueError(f"Invalid resource_name: {resource_name}. Must be a non-empty string.")
    
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError(f"Invalid threshold: {threshold}. Must be non-negative.")
    
    if not isinstance(consecutive_days, int) or consecutive_days < 1:
        raise ValueError(f"Invalid consecutive_days: {consecutive_days}. Must be a positive integer.")
    
    # Initialize result dictionary
    result = {
        'spike_detected': False,
        'consecutive_days': 0,
        'threshold': threshold,
        'current_cost': current_cost,
        'alert_triggered': False
    }
    
    # Get current date (date only, not datetime)
    current_date = date.today()
    
    # Initialize or get tracking for this resource
    if resource_name not in _sustained_cost_spike_tracking:
        _sustained_cost_spike_tracking[resource_name] = {
            'count': 0,
            'last_check_date': None
        }
    
    tracking = _sustained_cost_spike_tracking[resource_name]
    last_check_date = tracking.get('last_check_date')
    
    # Update tracking based on date and cost
    if last_check_date is None:
        # First check ever for this resource
        if current_cost > threshold:
            tracking['count'] = 1
            tracking['last_check_date'] = current_date
            result['spike_detected'] = True
            result['consecutive_days'] = 1
        else:
            tracking['count'] = 0
            tracking['last_check_date'] = current_date
            result['consecutive_days'] = 0
    
    elif last_check_date == current_date:
        # Already checked today, don't increment (prevent multiple increments per day)
        result['consecutive_days'] = tracking['count']
        result['spike_detected'] = tracking['count'] > 0 and current_cost > threshold
    
    elif last_check_date == current_date - timedelta(days=1):
        # Consecutive day (yesterday was last check)
        if current_cost > threshold:
            tracking['count'] += 1
            tracking['last_check_date'] = current_date
            result['spike_detected'] = True
            result['consecutive_days'] = tracking['count']
        else:
            # Cost fell below threshold, reset counter
            tracking['count'] = 0
            tracking['last_check_date'] = current_date
            result['consecutive_days'] = 0
    
    else:
        # Gap in days (not consecutive), reset counter
        if current_cost > threshold:
            tracking['count'] = 1
            tracking['last_check_date'] = current_date
            result['spike_detected'] = True
            result['consecutive_days'] = 1
        else:
            tracking['count'] = 0
            tracking['last_check_date'] = current_date
            result['consecutive_days'] = 0
    
    # Check if consecutive_days requirement is met
    if tracking['count'] >= consecutive_days:
        # Trigger alert for sustained spike
        try:
            from anomaly_detection.alert_trigger import trigger_cost_alert
            alert_triggered = trigger_cost_alert(
                current_cost=current_cost,
                resource_name=resource_name,
                timestamp=datetime.utcnow(),
                alert_reason="Sustained Cost Spike"
            )
            result['alert_triggered'] = alert_triggered
            
            # Trigger auto-scaling if enabled
            try:
                # Use module-level get_settings to avoid UnboundLocalError
                settings = get_settings()
                if settings and settings.self_healing_enabled:
                    from self_healing.auto_scaling import scale_down_resources
                    scale_down_resources(
                        resource_name=resource_name,
                        current_cost=current_cost
                    )
            except (ImportError, AttributeError) as e:
                # Self-healing not available - continue without scaling
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.debug(f"Self-healing not available: {e}")
                except ImportError:
                    pass
            
            # Reset counter after alert is triggered
            tracking['count'] = 0
            result['consecutive_days'] = 0
        except ImportError:
            # Fallback to trigger_alert if trigger_cost_alert not available
            try:
                from anomaly_detection.alert_trigger import trigger_alert
                alert_triggered = trigger_alert(
                    "Cloud Cost (Sustained Spike)",
                    current_cost,
                    datetime.utcnow(),
                    resource_name
                )
                result['alert_triggered'] = alert_triggered
                tracking['count'] = 0
                result['consecutive_days'] = 0
            except ImportError:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning("Alert trigger functions not available")
                except ImportError:
                    print("Alert trigger functions not available")
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error triggering sustained cost spike alert: {e}", exc_info=True)
            except ImportError:
                print(f"Error triggering sustained cost spike alert: {e}")
    
    return result


def reset_sustained_spike_tracking(resource_name: Optional[str] = None) -> None:
    """
    Reset sustained cost spike tracking.
    
    This function clears the tracking dictionary for a specific resource
    or all resources. Useful for testing or manual reset.
    
    Args:
        resource_name: Optional resource name to reset. If None, resets all.
    """
    global _sustained_cost_spike_tracking
    
    if resource_name:
        if resource_name in _sustained_cost_spike_tracking:
            del _sustained_cost_spike_tracking[resource_name]
    else:
        _sustained_cost_spike_tracking.clear()
