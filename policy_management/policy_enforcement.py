"""
Logic for enforcing policies and triggering alerts.

This module provides functionality to evaluate policies against
current system state and trigger appropriate actions.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time
from .policy_definition import Policy, PolicyRule, ResourcePolicy


class PolicyEvaluator:
    """Evaluates policy rules against context data."""
    
    def evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        """
        Evaluate a condition against context data.
        
        Args:
            condition: Condition definition dictionary
            context: Context data to evaluate against
        
        Returns:
            True if condition is met, False otherwise
        
        TODO: Implement condition evaluation logic
        """
        # TODO: Implement various condition types:
        # - Metric comparisons (threshold checks)
        # - Time-based conditions
        # - Aggregation conditions (average, max, min over time)
        # - Logical operators (AND, OR, NOT)
        
        condition_type = condition.get("type")
        
        if condition_type == "metric_threshold":
            # TODO: Evaluate metric threshold condition
            metric_name = condition.get("metric_name")
            operator = condition.get("operator", "gt")
            threshold = condition.get("threshold")
            
            if metric_name in context:
                metric_value = context[metric_name]
                # TODO: Implement comparison logic
                return False
        
        elif condition_type == "time_range":
            # TODO: Evaluate time-based condition
            pass
        
        elif condition_type == "aggregation":
            # TODO: Evaluate aggregation condition
            pass
        
        return False
    
    def evaluate_rule(self, rule: PolicyRule, context: Dict) -> bool:
        """
        Evaluate a policy rule against context data.
        
        Args:
            rule: PolicyRule instance to evaluate
            context: Context data to evaluate against
        
        Returns:
            True if rule condition is met, False otherwise
        
        TODO: Implement rule evaluation
        """
        return self.evaluate_condition(rule.condition, context)


class PolicyEnforcer:
    """Enforces policies and triggers actions."""
    
    def __init__(self, policy_manager=None, alert_system=None):
        """
        Initialize policy enforcer.
        
        Args:
            policy_manager: Optional policy definition manager
            alert_system: Optional alerting system for triggering alerts
        """
        self.policy_manager = policy_manager
        self.alert_system = alert_system
        self.evaluator = PolicyEvaluator()
    
    def enforce_policy(self, policy: Policy, context: Dict) -> Dict:
        """
        Enforce a policy against context data.
        
        Args:
            policy: Policy instance to enforce
            context: Context data to evaluate policy against
        
        Returns:
            Dictionary containing enforcement result
        
        TODO: Implement policy enforcement logic
        """
        if not policy.enabled:
            return {
                "policy_id": policy.policy_id,
                "enforced": False,
                "reason": "Policy is disabled",
                "violations": [],
                "actions_taken": []
            }
        
        violations = []
        actions_taken = []
        
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(policy.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if self.evaluator.evaluate_rule(rule, context):
                violations.append({
                    "rule": rule.to_dict(),
                    "context": context,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Execute action
                action_result = self._execute_action(rule.action, context, policy)
                actions_taken.append(action_result)
        
        return {
            "policy_id": policy.policy_id,
            "enforced": True,
            "violations": violations,
            "actions_taken": actions_taken,
            "timestamp": datetime.now().isoformat()
        }
    
    def enforce_all_policies(self, context: Dict) -> List[Dict]:
        """
        Enforce all enabled policies against context data.
        
        Args:
            context: Context data to evaluate policies against
        
        Returns:
            List of enforcement result dictionaries
        
        TODO: Implement batch policy enforcement
        """
        if not self.policy_manager:
            return []
        
        policies = self.policy_manager.list_policies(enabled_only=True)
        results = []
        
        for policy in policies:
            result = self.enforce_policy(policy, context)
            results.append(result)
        
        return results
    
    def _execute_action(self, action: Dict, context: Dict, policy: Policy) -> Dict:
        """
        Execute an action defined in a policy rule.
        
        Args:
            action: Action definition dictionary
            context: Context data
            policy: Policy instance
        
        Returns:
            Dictionary containing action execution result
        
        TODO: Implement action execution logic
        """
        action_type = action.get("type")
        
        if action_type == "alert":
            # TODO: Trigger alert using alert_system
            alert_data = {
                "alert_type": "policy_violation",
                "severity": action.get("severity", "medium"),
                "message": action.get("message", f"Policy violation: {policy.name}"),
                "metadata": {
                    "policy_id": policy.policy_id,
                    "policy_name": policy.name,
                    "context": context
                }
            }
            
            if self.alert_system:
                # TODO: self.alert_system.create_alert(alert_data)
                pass
            
            return {
                "action_type": "alert",
                "status": "triggered",
                "alert_data": alert_data
            }
        
        elif action_type == "scale_down":
            # TODO: Trigger scale down action
            return {
                "action_type": "scale_down",
                "status": "pending",
                "message": "Scale down action not implemented"
            }
        
        elif action_type == "shutdown":
            # TODO: Trigger shutdown action
            return {
                "action_type": "shutdown",
                "status": "pending",
                "message": "Shutdown action not implemented"
            }
        
        return {
            "action_type": action_type,
            "status": "unknown",
            "message": f"Unknown action type: {action_type}"
        }


# Module-level tracking for resource policy exceedance duration
# Key: policy_id, Value: start_timestamp (float - Unix timestamp)
_resource_policy_exceedance_times: Dict[str, float] = {}


def enforce_resource_policy(
    policy: ResourcePolicy,
    real_time_value: float
) -> Dict[str, Any]:
    """
    Enforce a resource policy against real-time data.
    
    This function compares the real-time data value against the policy threshold
    and determines if a violation has occurred.
    
    Args:
        policy: ResourcePolicy instance to enforce
        real_time_value: Current real-time metric value
    
    Returns:
        Dictionary containing enforcement result:
        {
            'policy_id': str,
            'resource_name': str,
            'violated': bool,
            'current_value': float,
            'threshold_value': float,
            'violation_amount': float,  # How much over threshold (0 if not violated)
            'timestamp': datetime
        }
    
    Raises:
        ValueError: If policy or real_time_value is invalid
    """
    # Validate inputs
    if not isinstance(policy, ResourcePolicy):
        raise ValueError(f"policy must be a ResourcePolicy instance, got {type(policy)}")
    
    if not isinstance(real_time_value, (int, float)):
        raise ValueError(f"real_time_value must be numeric, got {type(real_time_value)}")
    
    if real_time_value is None:
        raise ValueError("real_time_value cannot be None")
    
    # Initialize result
    result = {
        'policy_id': policy.policy_id,
        'resource_name': policy.resource_name,
        'violated': False,
        'current_value': float(real_time_value),
        'threshold_value': float(policy.threshold_value),
        'violation_amount': 0.0,
        'timestamp': datetime.utcnow()
    }
    
    # Check if policy is enabled
    if not policy.enabled:
        return result
    
    # Compare value against threshold
    if real_time_value > policy.threshold_value:
        result['violated'] = True
        result['violation_amount'] = float(real_time_value - policy.threshold_value)
    
    return result


def enforce_resource_policy_with_duration(
    policy: ResourcePolicy,
    real_time_value: float
) -> Dict[str, Any]:
    """
    Enforce a resource policy with duration tracking for sustained exceedance.
    
    This function tracks how long a metric exceeds the threshold and only
    triggers a violation after the specified duration requirement is met.
    
    Args:
        policy: ResourcePolicy instance to enforce
        real_time_value: Current real-time metric value
    
    Returns:
        Dictionary containing enforcement result (same format as enforce_resource_policy)
    
    Raises:
        ValueError: If policy or real_time_value is invalid
    """
    global _resource_policy_exceedance_times
    
    # Validate inputs
    if not isinstance(policy, ResourcePolicy):
        raise ValueError(f"policy must be a ResourcePolicy instance, got {type(policy)}")
    
    if not isinstance(real_time_value, (int, float)):
        raise ValueError(f"real_time_value must be numeric, got {type(real_time_value)}")
    
    if real_time_value is None:
        raise ValueError("real_time_value cannot be None")
    
    # Initialize result
    result = {
        'policy_id': policy.policy_id,
        'resource_name': policy.resource_name,
        'violated': False,
        'current_value': float(real_time_value),
        'threshold_value': float(policy.threshold_value),
        'violation_amount': 0.0,
        'timestamp': datetime.utcnow()
    }
    
    # Check if policy is enabled
    if not policy.enabled:
        return result
    
    policy_key = policy.policy_id or policy.resource_name
    current_timestamp = time.time()
    
    # Check if value exceeds threshold
    if real_time_value > policy.threshold_value:
        # Check if we're already tracking this exceedance
        if policy_key not in _resource_policy_exceedance_times:
            # Start tracking exceedance
            _resource_policy_exceedance_times[policy_key] = current_timestamp
            result['violated'] = False  # Not yet, need to wait for duration
        else:
            # Check if duration requirement is met
            exceedance_start = _resource_policy_exceedance_times[policy_key]
            exceedance_duration_seconds = (current_timestamp - exceedance_start) / 60.0  # Convert to minutes
            
            if exceedance_duration_seconds >= policy.duration:
                # Duration requirement met - violation occurred
                result['violated'] = True
                result['violation_amount'] = float(real_time_value - policy.threshold_value)
            else:
                # Still within duration requirement
                result['violated'] = False
    else:
        # Value is below threshold - reset tracking
        if policy_key in _resource_policy_exceedance_times:
            del _resource_policy_exceedance_times[policy_key]
        result['violated'] = False
    
    return result


def fetch_resource_policies_from_db(
    enabled_only: bool = True,
    threshold_type: Optional[str] = None
) -> List[ResourcePolicy]:
    """
    Fetch resource policies from the database.
    
    This is a wrapper around list_resource_policies_from_db() from policy_definition.py
    for use in policy enforcement.
    
    Args:
        enabled_only: If True, only return enabled policies (default: True)
        threshold_type: Optional filter by threshold type ("usage" or "cost")
    
    Returns:
        List of ResourcePolicy instances
    """
    try:
        from policy_management.policy_definition import list_resource_policies_from_db
        return list_resource_policies_from_db(
            enabled_only=enabled_only,
            threshold_type=threshold_type
        )
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("policy_definition module not available for fetching policies")
        except ImportError:
            print("Error: policy_definition module not available for fetching policies")
        return []
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error fetching resource policies: {e}", exc_info=True)
        except ImportError:
            print(f"Error fetching resource policies: {e}")
        return []


def get_realtime_data_for_policy(
    policy: ResourcePolicy
) -> Optional[Tuple[float, datetime, str]]:
    """
    Fetch real-time data for a specific resource policy.
    
    This function maps resource_name to the appropriate data source and
    extracts the relevant metric value.
    
    Args:
        policy: ResourcePolicy instance
    
    Returns:
        Tuple of (value: float, timestamp: datetime, resource_type: str) if successful,
        None if data unavailable or resource_name not recognized
    """
    resource_name = policy.resource_name.lower()
    
    try:
        # Handle CPU Usage
        if "cpu" in resource_name and "usage" in resource_name:
            try:
                from data_collection.system_metrics import get_server_metrics
                server_metrics = get_server_metrics()
                if 'cpu_usage' in server_metrics:
                    return (
                        float(server_metrics['cpu_usage']),
                        server_metrics.get('timestamp', datetime.utcnow()),
                        "Server"
                    )
            except (ImportError, RuntimeError, KeyError) as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Failed to fetch CPU usage data: {e}")
                except ImportError:
                    print(f"Warning: Failed to fetch CPU usage data: {e}")
            return None
        
        # Handle Memory Usage
        elif "memory" in resource_name and "usage" in resource_name:
            try:
                from data_collection.system_metrics import get_server_metrics
                server_metrics = get_server_metrics()
                if 'memory_usage' in server_metrics:
                    return (
                        float(server_metrics['memory_usage']),
                        server_metrics.get('timestamp', datetime.utcnow()),
                        "Server"
                    )
            except (ImportError, RuntimeError, KeyError) as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Failed to fetch memory usage data: {e}")
                except ImportError:
                    print(f"Warning: Failed to fetch memory usage data: {e}")
            return None
        
        # Handle Cloud Cost
        elif "cloud" in resource_name and "cost" in resource_name:
            try:
                from data_collection.cloud_metrics import get_cloud_costs
                cloud_data = get_cloud_costs(store_in_db=False)
                if 'total_cost' in cloud_data:
                    return (
                        float(cloud_data['total_cost']),
                        cloud_data.get('timestamp', datetime.utcnow()),
                        "Cloud"
                    )
            except (ImportError, RuntimeError, KeyError) as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Failed to fetch cloud cost data: {e}")
                except ImportError:
                    print(f"Warning: Failed to fetch cloud cost data: {e}")
            return None
        
        # Unknown resource name
        else:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Unknown resource name for policy: {policy.resource_name}")
            except ImportError:
                print(f"Warning: Unknown resource name for policy: {policy.resource_name}")
            return None
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error fetching real-time data for policy {policy.resource_name}: {e}", exc_info=True)
        except ImportError:
            print(f"Error fetching real-time data for policy {policy.resource_name}: {e}")
        return None


def enforce_all_resource_policies() -> List[Dict[str, Any]]:
    """
    Enforce all enabled resource policies against real-time data.
    
    This function fetches all enabled policies from the database, gets real-time
    data for each, and enforces them. Errors for individual policies are handled
    gracefully so one failure doesn't stop the entire process.
    
    Returns:
        List of enforcement result dictionaries
    """
    results = []
    
    try:
        # Fetch all enabled policies
        policies = fetch_resource_policies_from_db(enabled_only=True)
        
        if not policies:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info("No enabled resource policies found to enforce")
            except ImportError:
                print("Info: No enabled resource policies found to enforce")
            return results
        
        # Enforce each policy
        for policy in policies:
            try:
                # Get real-time data for this policy
                data_result = get_realtime_data_for_policy(policy)
                
                if data_result is None:
                    # Data unavailable - skip this policy
                    results.append({
                        'policy_id': policy.policy_id,
                        'resource_name': policy.resource_name,
                        'violated': False,
                        'error': 'Real-time data unavailable',
                        'timestamp': datetime.utcnow()
                    })
                    continue
                
                value, timestamp, resource_type = data_result
                
                # Enforce policy (with duration tracking if duration > 0)
                if policy.duration > 0:
                    enforcement_result = enforce_resource_policy_with_duration(policy, value)
                else:
                    enforcement_result = enforce_resource_policy(policy, value)
                
                # Add resource_type to result
                enforcement_result['resource_type'] = resource_type
                results.append(enforcement_result)
            
            except Exception as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(
                        f"Error enforcing policy {policy.resource_name} (ID: {policy.policy_id}): {e}",
                        exc_info=True
                    )
                except ImportError:
                    print(f"Error enforcing policy {policy.resource_name}: {e}")
                
                # Add error result
                results.append({
                    'policy_id': policy.policy_id,
                    'resource_name': policy.resource_name,
                    'violated': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow()
                })
        
        return results
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error in enforce_all_resource_policies: {e}", exc_info=True)
        except ImportError:
            print(f"Error in enforce_all_resource_policies: {e}")
        return results


def enforce_and_alert() -> Dict[str, Any]:
    """
    Enforce all resource policies and trigger alerts for violations.
    
    This function orchestrates the complete flow:
    1. Fetch all enabled policies from database
    2. Get real-time data for each policy
    3. Enforce policies against data
    4. Trigger alerts for violations
    
    Returns:
        Dictionary containing summary:
        {
            'policies_checked': int,
            'violations_found': int,
            'alerts_triggered': int,
            'results': List[Dict]  # Individual enforcement results
        }
    """
    summary = {
        'policies_checked': 0,
        'violations_found': 0,
        'alerts_triggered': 0,
        'results': []
    }
    
    try:
        # Enforce all policies
        enforcement_results = enforce_all_resource_policies()
        summary['policies_checked'] = len(enforcement_results)
        summary['results'] = enforcement_results
        
        # Process violations and trigger alerts
        for result in enforcement_results:
            if result.get('violated', False):
                summary['violations_found'] += 1
                
                # Trigger alert
                try:
                    from anomaly_detection.alert_trigger import trigger_alert
                    
                    metric_name = result.get('resource_name', 'Unknown Resource')
                    value = result.get('current_value', 0.0)
                    timestamp = result.get('timestamp', datetime.utcnow())
                    resource_type = result.get('resource_type', 'Unknown')
                    
                    # Ensure timestamp is datetime object
                    if isinstance(timestamp, str):
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if timestamp.tzinfo:
                            timestamp = timestamp.replace(tzinfo=None)
                    
                    alert_triggered = trigger_alert(
                        metric_name=metric_name,
                        value=float(value),
                        timestamp=timestamp,
                        resource_type=resource_type
                    )
                    
                    if alert_triggered:
                        summary['alerts_triggered'] += 1
                
                except ImportError:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.error("Alert trigger module not available")
                    except ImportError:
                        print("Error: Alert trigger module not available")
                except Exception as e:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.error(f"Error triggering alert for policy violation: {e}", exc_info=True)
                    except ImportError:
                        print(f"Error triggering alert: {e}")
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(
                f"Policy enforcement complete: {summary['policies_checked']} policies checked, "
                f"{summary['violations_found']} violations found, "
                f"{summary['alerts_triggered']} alerts triggered"
            )
        except ImportError:
            print(
                f"Policy enforcement complete: {summary['policies_checked']} policies checked, "
                f"{summary['violations_found']} violations found"
            )
        
        return summary
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error in enforce_and_alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error in enforce_and_alert: {e}")
        return summary


def create_policy_enforcer(
    policy_manager=None,
    alert_system=None
) -> PolicyEnforcer:
    """
    Create a policy enforcer instance.
    
    Args:
        policy_manager: Optional policy definition manager
        alert_system: Optional alerting system
    
    Returns:
        PolicyEnforcer instance
    
    TODO: Implement enforcer factory
    """
    return PolicyEnforcer(
        policy_manager=policy_manager,
        alert_system=alert_system
    )
