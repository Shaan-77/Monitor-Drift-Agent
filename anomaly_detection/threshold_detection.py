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

# Import policy management for dynamic policy loading
try:
    from policy_management.policy_definition import list_resource_policies_from_db, ResourcePolicy
    POLICY_MANAGEMENT_AVAILABLE = True
except ImportError:
    POLICY_MANAGEMENT_AVAILABLE = False
    list_resource_policies_from_db = None
    ResourcePolicy = None

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


def load_policies_from_db(
    resource_type: str = None,
    threshold_type: str = None
) -> Dict[str, ResourcePolicy]:
    """
    Load policies from database dynamically.
    
    This function loads enabled policies from the database and returns them
    as a dictionary keyed by resource name for efficient lookup.
    
    Args:
        resource_type: Optional filter by resource type (Server, AWS, GCP, Azure)
        threshold_type: Optional filter by threshold type (usage, cost)
    
    Returns:
        Dictionary mapping resource names to ResourcePolicy instances
    """
    policies_dict = {}
    
    if not POLICY_MANAGEMENT_AVAILABLE or list_resource_policies_from_db is None:
        return policies_dict
    
    try:
        # Load enabled policies from database
        policies = list_resource_policies_from_db(
            enabled_only=True,
            threshold_type=threshold_type
        )
        
        # Filter by resource_type if specified
        if resource_type:
            # Map resource_type to common resource name patterns
            resource_patterns = {
                "Server": ["Server", "CPU", "Memory", "Disk", "Network"],
                "AWS": ["AWS", "EC2", "S3", "RDS"],
                "GCP": ["GCP", "Compute", "Storage", "Cloud"],
                "Azure": ["Azure", "VM", "Blob"]
            }
            
            patterns = resource_patterns.get(resource_type, [resource_type])
            policies = [
                p for p in policies
                if any(pattern.lower() in p.resource_name.lower() for pattern in patterns)
            ]
        
        # Convert to dictionary for efficient lookup
        for policy in policies:
            policies_dict[policy.resource_name] = policy
        
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Error loading policies from database: {e}")
        except ImportError:
            pass
    
    return policies_dict


def get_policy_for_metric(
    metric_name: str,
    resource_type: str,
    policies_dict: Dict[str, ResourcePolicy] = None
) -> Optional[ResourcePolicy]:
    """
    Get the appropriate policy for a metric based on metric name and resource type.
    
    Args:
        metric_name: Name of the metric (e.g., "CPU Usage", "Cloud Cost")
        resource_type: Resource type (Server, AWS, GCP, Azure)
        policies_dict: Optional pre-loaded policies dictionary
    
    Returns:
        Matching ResourcePolicy or None if not found
    """
    if policies_dict is None:
        # Determine threshold type from metric name
        threshold_type = "cost" if "cost" in metric_name.lower() else "usage"
        policies_dict = load_policies_from_db(resource_type=resource_type, threshold_type=threshold_type)
    
    # Try exact match first
    if metric_name in policies_dict:
        return policies_dict[metric_name]
    
    # Normalize strings for matching
    metric_lower = metric_name.lower().strip()
    
    # Try partial matches - more flexible matching
    for resource_name, policy in policies_dict.items():
        resource_lower = resource_name.lower().strip()
        
        # Check threshold type compatibility first
        is_cost_metric = "cost" in metric_lower
        is_usage_metric = "usage" in metric_lower or "cpu" in metric_lower or "memory" in metric_lower or "network" in metric_lower
        
        # Skip if threshold types don't match
        if is_cost_metric and policy.threshold_type != "cost":
            continue
        if is_usage_metric and policy.threshold_type != "usage":
            continue
        
        # Try various matching strategies
        # 1. Exact match (case insensitive)
        if metric_lower == resource_lower:
            return policy
        
        # 2. Metric name contains policy name or vice versa
        if resource_lower in metric_lower or metric_lower in resource_lower:
            return policy
        
        # 3. Both contain common keywords
        cpu_keywords = ["cpu", "processor"]
        memory_keywords = ["memory", "mem", "ram"]
        cost_keywords = ["cost", "price", "billing"]
        
        metric_has_cpu = any(kw in metric_lower for kw in cpu_keywords)
        metric_has_memory = any(kw in metric_lower for kw in memory_keywords)
        metric_has_cost = any(kw in metric_lower for kw in cost_keywords)
        
        policy_has_cpu = any(kw in resource_lower for kw in cpu_keywords)
        policy_has_memory = any(kw in resource_lower for kw in memory_keywords)
        policy_has_cost = any(kw in resource_lower for kw in cost_keywords)
        
        # Match if both have same keyword
        if metric_has_cpu and policy_has_cpu and policy.threshold_type == "usage":
            return policy
        if metric_has_memory and policy_has_memory and policy.threshold_type == "usage":
            return policy
        if metric_has_cost and policy_has_cost and policy.threshold_type == "cost":
            return policy
    
    # Try common patterns as last resort
    if "cpu" in metric_lower:
        for name, policy in policies_dict.items():
            if ("cpu" in name.lower() or "processor" in name.lower()) and policy.threshold_type == "usage":
                return policy
    if "memory" in metric_lower or "mem" in metric_lower:
        for name, policy in policies_dict.items():
            if ("memory" in name.lower() or "mem" in name.lower() or "ram" in name.lower()) and policy.threshold_type == "usage":
                return policy
    if "cost" in metric_lower:
        for name, policy in policies_dict.items():
            if "cost" in name.lower() and policy.threshold_type == "cost":
                return policy
    
    return None


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
    
    This function dynamically loads policies from the database and uses them
    for threshold detection. If no policies are found, it falls back to
    settings. It tracks if each metric exceeds its threshold for the required
    duration (e.g., 5 minutes) before triggering an alert.
    
    Args:
        cpu_usage: Current CPU usage percentage (0-100)
        cloud_cost: Current cloud cost in dollars per day
        cpu_threshold: CPU usage threshold percentage (overrides policy if provided)
        cost_threshold: Cloud cost threshold in dollars per day (overrides policy if provided)
        duration: Duration in minutes for threshold to be sustained (overrides policy if provided)
        resource_type: Resource type identifier (Server, AWS, GCP, Azure)
    
    Returns:
        List of triggered alerts (empty if no thresholds exceeded)
    
    Raises:
        RuntimeError: If required dependencies are not available
        ValueError: If input values are invalid
    """
    triggered_alerts = []
    
    # Load policies dynamically from database
    usage_policies = load_policies_from_db(resource_type=resource_type, threshold_type="usage")
    cost_policies = load_policies_from_db(resource_type=resource_type, threshold_type="cost")
    
    # Get CPU policy
    cpu_policy = get_policy_for_metric("CPU Usage", resource_type, usage_policies)
    if cpu_policy and cpu_threshold is None:
        cpu_threshold = cpu_policy.threshold_value
        if duration is None:
            duration = cpu_policy.duration
    
    # Get cost policy
    cost_policy = get_policy_for_metric("Cloud Cost", resource_type, cost_policies)
    if cost_policy and cost_threshold is None:
        cost_threshold = cost_policy.threshold_value
        if duration is None:
            duration = cost_policy.duration
    
    # Fallback to settings if no policies found
    if cpu_threshold is None or cost_threshold is None or duration is None:
        settings = get_settings()
        if settings is None:
            raise RuntimeError("Settings not available and no policies found. Cannot determine thresholds.")
        
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


def check_thresholds_with_policies(
    metrics: Dict[str, float],
    resource_type: str = "Server"
) -> List[Dict]:
    """
    Check thresholds dynamically using policies from database.
    
    This function loads policies from the database and evaluates all provided
    metrics against their corresponding policies.
    
    Args:
        metrics: Dictionary mapping metric names to values
        resource_type: Resource type identifier (Server, AWS, GCP, Azure)
    
    Returns:
        List of triggered alerts (empty if no thresholds exceeded)
    """
    triggered_alerts = []
    
    # Load all relevant policies
    usage_policies = load_policies_from_db(resource_type=resource_type, threshold_type="usage")
    cost_policies = load_policies_from_db(resource_type=resource_type, threshold_type="cost")
    all_policies = {**usage_policies, **cost_policies}
    
    # Debug: Log policy loading
    try:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.debug(f"Loaded {len(all_policies)} policies for resource_type={resource_type}")
        for name, policy in all_policies.items():
            logger.debug(f"Policy: {name} - threshold={policy.threshold_value}, enabled={policy.enabled}, duration={policy.duration}")
    except ImportError:
        pass
    
    # If no policies found, return empty (no alerts)
    if not all_policies:
        print(f"[DEBUG] No policies found for resource_type={resource_type}")
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"No policies found for resource_type={resource_type}, threshold_type=usage or cost")
        except ImportError:
            pass
        return triggered_alerts
    
    print(f"[DEBUG] Loaded {len(all_policies)} policies: {list(all_policies.keys())}")
    
    current_timestamp = time.time()
    current_datetime = datetime.utcnow()
    
    # Evaluate each metric against its policy
    for metric_name, metric_value in metrics.items():
        policy = get_policy_for_metric(metric_name, resource_type, all_policies)
        
        # Debug: Log policy matching
        if policy is None:
            print(f"[DEBUG] No policy found for metric: {metric_name}, resource_type: {resource_type}")
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.debug(f"No policy found for metric: {metric_name}, resource_type: {resource_type}")
            except ImportError:
                pass
            continue
        elif not policy.enabled:
            print(f"[DEBUG] Policy found for {metric_name} but it's disabled")
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.debug(f"Policy found for {metric_name} but it's disabled")
            except ImportError:
                pass
            continue
        else:
            print(f"[DEBUG] Evaluating {metric_name} (value={metric_value}) against policy {policy.resource_name} (threshold={policy.threshold_value}, duration={policy.duration})")
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.debug(f"Evaluating {metric_name} (value={metric_value}) against policy {policy.resource_name} (threshold={policy.threshold_value})")
            except ImportError:
                pass
        
        # Check if metric exceeds threshold
        if metric_value > policy.threshold_value:
            metric_key = (metric_name, resource_type)
            
            # If duration is 0 or very small (<= 0.1 minutes = 6 seconds), trigger immediately
            if policy.duration <= 0.1:
                # Immediate trigger for testing or real-time alerts
                try:
                    from anomaly_detection.alert_trigger import trigger_alert
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"Triggering immediate alert: {metric_name}={metric_value} exceeds threshold={policy.threshold_value}")
                    except ImportError:
                        print(f"Triggering immediate alert: {metric_name}={metric_value} exceeds threshold={policy.threshold_value}")
                    
                    success = trigger_alert(metric_name, metric_value, current_datetime, resource_type)
                    if success:
                        triggered_alerts.append({
                            "metric_name": metric_name,
                            "value": metric_value,
                            "timestamp": current_datetime,
                            "resource_type": resource_type,
                            "threshold": policy.threshold_value,
                            "duration_exceeded": 0.0,
                            "policy_id": policy.policy_id
                        })
                        try:
                            from utils.logger import get_logger
                            logger = get_logger(__name__)
                            logger.info(f"Alert triggered successfully for {metric_name}")
                        except ImportError:
                            print(f"Alert triggered successfully for {metric_name}")
                    else:
                        try:
                            from utils.logger import get_logger
                            logger = get_logger(__name__)
                            logger.error(f"Failed to trigger alert for {metric_name} - trigger_alert returned False")
                        except ImportError:
                            print(f"ERROR: Failed to trigger alert for {metric_name}")
                except Exception as e:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.error(f"Error triggering immediate alert for {metric_name}: {e}", exc_info=True)
                    except ImportError:
                        print(f"ERROR triggering alert for {metric_name}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                # Duration-based tracking
                if metric_key not in _threshold_exceedance_times:
                    # Start tracking exceedance
                    _threshold_exceedance_times[metric_key] = current_timestamp
                    print(f"[DEBUG] Started tracking {metric_name} exceedance (value={metric_value:.1f} > threshold={policy.threshold_value}). Waiting {policy.duration} minutes before alert.")
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"Started tracking {metric_name} exceedance. Will alert after {policy.duration} minutes if threshold remains exceeded.")
                    except ImportError:
                        pass
                else:
                    # Check if duration requirement is met
                    exceedance_start = _threshold_exceedance_times[metric_key]
                    exceedance_duration = (current_timestamp - exceedance_start) / 60.0  # Convert to minutes
                    
                    print(f"[DEBUG] {metric_name} still exceeding threshold. Duration so far: {exceedance_duration:.2f}/{policy.duration} minutes")
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.debug(f"{metric_name} exceedance duration: {exceedance_duration:.2f}/{policy.duration} minutes")
                    except ImportError:
                        pass
                    
                    if exceedance_duration >= policy.duration:
                        # Duration requirement met, trigger alert
                        print(f"[DEBUG] Duration requirement met ({exceedance_duration:.2f} >= {policy.duration} minutes). Triggering alert for {metric_name}.")
                        try:
                            from anomaly_detection.alert_trigger import trigger_alert
                            try:
                                from utils.logger import get_logger
                                logger = get_logger(__name__)
                                logger.info(f"Triggering alert: {metric_name}={metric_value:.1f} exceeded threshold={policy.threshold_value} for {exceedance_duration:.2f} minutes")
                            except ImportError:
                                print(f"Triggering alert: {metric_name}={metric_value:.1f} exceeded threshold={policy.threshold_value} for {exceedance_duration:.2f} minutes")
                            
                            success = trigger_alert(metric_name, metric_value, current_datetime, resource_type)
                            if success:
                                triggered_alerts.append({
                                    "metric_name": metric_name,
                                    "value": metric_value,
                                    "timestamp": current_datetime,
                                    "resource_type": resource_type,
                                    "threshold": policy.threshold_value,
                                    "duration_exceeded": exceedance_duration,
                                    "policy_id": policy.policy_id
                                })
                                print(f"[DEBUG] Alert triggered successfully for {metric_name} after {exceedance_duration:.2f} minutes")
                                # Reset tracking after alert is triggered
                                del _threshold_exceedance_times[metric_key]
                                try:
                                    from utils.logger import get_logger
                                    logger = get_logger(__name__)
                                    logger.info(f"Alert triggered successfully for {metric_name}. Tracking reset.")
                                except ImportError:
                                    pass
                            else:
                                print(f"[ERROR] Failed to trigger alert for {metric_name} - trigger_alert returned False")
                                try:
                                    from utils.logger import get_logger
                                    logger = get_logger(__name__)
                                    logger.error(f"Failed to trigger alert for {metric_name} - trigger_alert returned False")
                                except ImportError:
                                    pass
                        except Exception as e:
                            try:
                                from utils.logger import get_logger
                                logger = get_logger(__name__)
                                logger.error(f"Error triggering duration-based alert for {metric_name}: {e}", exc_info=True)
                            except ImportError:
                                print(f"ERROR triggering alert for {metric_name}: {e}")
                                import traceback
                                traceback.print_exc()
        else:
            # Metric is below threshold, reset tracking
            metric_key = (metric_name, resource_type)
            if metric_key in _threshold_exceedance_times:
                print(f"[DEBUG] {metric_name} is now below threshold ({metric_value:.1f} <= {policy.threshold_value}). Resetting duration tracking.")
                del _threshold_exceedance_times[metric_key]
    
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
    # Try to get thresholds from policies first
    cost_policies = load_policies_from_db(threshold_type="cost")
    cost_policy = get_policy_for_metric("Cloud Cost", resource_name, cost_policies)
    
    # Get thresholds from settings if not provided and no policy found
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available. Cannot determine thresholds.")
    
    if spike_threshold_percent is None:
        # Use policy threshold if available, otherwise use settings
        if cost_policy:
            # Convert threshold value to percentage (assuming 20% default)
            spike_threshold_percent = 20.0  # Default spike threshold
        else:
            spike_threshold_percent = settings.cost_spike_threshold_percent
    if lookback_days is None:
        if cost_policy:
            # Use policy duration as lookback days (convert minutes to days, min 1 day)
            lookback_days = max(1, cost_policy.duration // (24 * 60))
        else:
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
    # Try to get thresholds from policies first
    cost_policies = load_policies_from_db(threshold_type="cost")
    cost_policy = get_policy_for_metric("Cloud Cost", resource_name, cost_policies)
    
    # Get thresholds from settings if not provided and no policy found
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available. Cannot determine thresholds.")
    
    if threshold is None:
        if cost_policy:
            threshold = cost_policy.threshold_value
        else:
            threshold = settings.cloud_cost_threshold
    if consecutive_days is None:
        if cost_policy:
            # Use policy duration converted to days (min 1 day)
            consecutive_days = max(1, cost_policy.duration // (24 * 60))
        else:
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
