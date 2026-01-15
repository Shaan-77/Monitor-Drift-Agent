"""
Logic for triggering alerts based on detected anomalies.

This module provides functionality to trigger alerts when anomalies
are detected by the anomaly detection system.
"""

from typing import Dict, List, Optional
from datetime import datetime


class AlertTrigger:
    """Handles alert triggering based on anomaly detection results."""
    
    def __init__(self, alert_system=None):
        """
        Initialize alert trigger.
        
        Args:
            alert_system: Reference to alerting system module
        """
        self.alert_system = alert_system
    
    def trigger_alert(self, anomaly: Dict) -> Dict:
        """
        Trigger an alert for a detected anomaly.
        
        Args:
            anomaly: Dictionary containing anomaly information
        
        Returns:
            Dictionary containing created alert information
        
        TODO: Implement alert triggering logic
        """
        # TODO: Determine alert severity based on anomaly
        severity = self._determine_severity(anomaly)
        
        # TODO: Format alert message
        message = self._format_alert_message(anomaly)
        
        # TODO: Create alert using alerting system
        alert_data = {
            "alert_type": "anomaly",
            "severity": severity,
            "message": message,
            "metadata": {
                "anomaly": anomaly,
                "detected_at": anomaly.get("timestamp", datetime.now().isoformat())
            }
        }
        
        if self.alert_system:
            # TODO: Call alert_system.create_alert(alert_data)
            pass
        
        return alert_data
    
    def trigger_batch_alerts(self, anomalies: List[Dict]) -> List[Dict]:
        """
        Trigger alerts for multiple detected anomalies.
        
        Args:
            anomalies: List of anomaly dictionaries
        
        Returns:
            List of created alert dictionaries
        
        TODO: Implement batch alert triggering
        """
        alerts = []
        for anomaly in anomalies:
            alert = self.trigger_alert(anomaly)
            alerts.append(alert)
        return alerts
    
    def _determine_severity(self, anomaly: Dict) -> str:
        """
        Determine alert severity based on anomaly characteristics.
        
        Args:
            anomaly: Dictionary containing anomaly information
        
        Returns:
            Severity level string
        
        TODO: Implement severity determination logic
        """
        # TODO: Implement logic to determine severity based on:
        # - Anomaly score/magnitude
        # - Metric type
        # - Historical patterns
        # - Policy rules
        
        if "severity" in anomaly:
            return anomaly["severity"]
        
        # Default severity based on anomaly score if available
        if "anomaly_score" in anomaly:
            score = abs(anomaly["anomaly_score"])
            if score > 0.8:
                return "critical"
            elif score > 0.6:
                return "high"
            elif score > 0.4:
                return "medium"
            else:
                return "low"
        
        return "medium"
    
    def _format_alert_message(self, anomaly: Dict) -> str:
        """
        Format alert message from anomaly information.
        
        Args:
            anomaly: Dictionary containing anomaly information
        
        Returns:
            Formatted alert message string
        
        TODO: Implement message formatting
        """
        # TODO: Create descriptive alert message
        metric_name = anomaly.get("metric_name", "Unknown metric")
        metric_value = anomaly.get("metric_value", "N/A")
        timestamp = anomaly.get("timestamp", datetime.now().isoformat())
        
        return f"Anomaly detected in {metric_name}: {metric_value} at {timestamp}"


def create_alert_trigger(alert_system=None) -> AlertTrigger:
    """
    Create an alert trigger instance.
    
    Args:
        alert_system: Optional reference to alerting system
    
    Returns:
        AlertTrigger instance
    
    TODO: Implement alert trigger factory
    """
    return AlertTrigger(alert_system=alert_system)


def trigger_cost_alert(
    current_cost: float,
    resource_name: str,
    timestamp: datetime,
    alert_reason: str = "Cost Threshold Exceeded"
) -> bool:
    """
    Trigger an alert when cloud cost exceeds historical averages or predefined limits.
    
    This function stores the alert in the PostgreSQL database with all necessary
    details including metric name, cost value, timestamp, and resource type.
    
    Args:
        current_cost: Current cloud cost that exceeded the threshold
        resource_name: Name of the cloud resource (e.g., 'AWS EC2', 'S3 Storage')
        timestamp: Timestamp when the alert was triggered (datetime object)
        alert_reason: Reason for the alert (e.g., 'Cost Threshold Exceeded', 
                     'Historical Average Exceeded') - used in metric_name
    
    Returns:
        True if alert stored successfully, False otherwise
    
    Example:
        >>> from datetime import datetime
        >>> trigger_cost_alert(600.0, "AWS EC2", datetime.utcnow(), "Historical Average Exceeded")
        True
    """
    try:
        # Validate inputs
        if not isinstance(current_cost, (int, float)) or current_cost < 0:
            print(f"Error: current_cost must be a non-negative number, got {type(current_cost)}")
            return False
        
        if not resource_name or not isinstance(resource_name, str):
            print(f"Error: resource_name must be a non-empty string, got {type(resource_name)}")
            return False
        
        if not isinstance(timestamp, datetime):
            print(f"Error: timestamp must be a datetime object, got {type(timestamp)}")
            return False
        
        # Extract provider from resource_name for resource_type
        # Examples: "AWS EC2" -> "AWS", "GCP Compute Engine" -> "GCP", "Azure VM" -> "Azure"
        resource_type = resource_name
        if "AWS" in resource_name or "aws" in resource_name.lower():
            resource_type = "AWS"
        elif "GCP" in resource_name or "gcp" in resource_name.lower() or "Google" in resource_name:
            resource_type = "GCP"
        elif "Azure" in resource_name or "azure" in resource_name.lower():
            resource_type = "Azure"
        else:
            # Default to using resource_name as resource_type
            resource_type = resource_name
        
        # Construct metric name with alert reason
        metric_name = f"Cloud Cost ({alert_reason})"
        
        # Call existing trigger_alert function
        return trigger_alert(metric_name, current_cost, timestamp, resource_type)
    
    except Exception as e:
        # Log error but don't crash
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error triggering cost alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error triggering cost alert: {e}")
        return False


def trigger_alert(
    metric_name: str,
    value: float,
    timestamp: datetime,
    resource_type: str
) -> bool:
    """
    Trigger an alert and store it in the database, then send notifications.
    
    This function stores the alert in the PostgreSQL database when
    an anomaly is detected (e.g., threshold exceeded). The alert
    contains all necessary details: metric name, value, timestamp,
    and resource type. After storing, it sends notifications via
    appropriate channels based on alert preferences.
    
    Args:
        metric_name: Name of the metric that triggered the alert (e.g., 'CPU Usage', 'Cloud Cost')
        value: Metric value that exceeded the threshold
        timestamp: Timestamp when the alert was triggered (datetime object)
        resource_type: Resource type identifier (Server, AWS, GCP, Azure)
    
    Returns:
        True if alert stored successfully, False otherwise
        (Returns True even if notifications fail - alert is still stored)
    
    Example:
        >>> from datetime import datetime
        >>> trigger_alert("CPU Usage", 85.5, datetime.utcnow(), "Server")
        True
    """
    try:
        # Import here to avoid circular dependencies
        from data_collection.database import store_alert_in_db
        
        # Validate inputs
        if not metric_name or not isinstance(metric_name, str):
            print(f"Error: metric_name must be a non-empty string, got {type(metric_name)}")
            return False
        
        if not isinstance(value, (int, float)):
            print(f"Error: value must be numeric, got {type(value)}")
            return False
        
        if not isinstance(timestamp, datetime):
            print(f"Error: timestamp must be a datetime object, got {type(timestamp)}")
            return False
        
        if not resource_type or not isinstance(resource_type, str):
            print(f"Error: resource_type must be a non-empty string, got {type(resource_type)}")
            return False
        
        # Ensure timestamp is UTC and timezone-naive for database operations
        if isinstance(timestamp, datetime):
            alert_timestamp = timestamp
            if alert_timestamp.tzinfo is not None:
                alert_timestamp = alert_timestamp.astimezone().replace(tzinfo=None)
        else:
            alert_timestamp = datetime.utcnow()
        
        # Determine severity before storing (needed for database)
        metric_lower = metric_name.lower()
        severity = "medium"  # Default
        if 'cost' in metric_lower:
            # Cost-based severity
            try:
                from config.settings import get_settings
                from unittest.mock import Mock as MockClass
                settings = get_settings()
                if settings:
                    threshold = getattr(settings, 'cloud_cost_threshold', 500.0)
                    # Ensure threshold is numeric (handle Mock objects in tests)
                    if isinstance(threshold, (int, float)) and not isinstance(threshold, MockClass):
                        if value > threshold * 2:
                            severity = "critical"
                        elif value > threshold:
                            severity = "high"
                        else:
                            severity = "medium"
                    else:
                        # Fallback if threshold is not numeric or is a Mock
                        severity = "high" if value > 500.0 else "medium"
                else:
                    severity = "high" if value > 500.0 else "medium"
            except (ImportError, AttributeError, TypeError):
                severity = "medium"
        elif 'usage' in metric_lower or 'cpu' in metric_lower or 'memory' in metric_lower:
            # Usage-based severity
            if value > 90:
                severity = "critical"
            elif value > 80:
                severity = "high"
            elif value > 60:
                severity = "medium"
            else:
                severity = "low"
        
        # Store alert in database (will update action_taken after notifications)
        # For now, use default action_taken, will be updated if notifications are sent
        action_taken = "Alert Triggered"
        
        alert_stored = store_alert_in_db(metric_name, value, timestamp, resource_type, severity, action_taken)
        
        if alert_stored:
            # Log alert trigger
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(
                    f"Alert triggered: {metric_name} = {value} "
                    f"for {resource_type} at {timestamp.isoformat()}"
                )
            except ImportError:
                # Logger not available, use print
                print(f"Alert triggered: {metric_name} = {value} for {resource_type} at {timestamp.isoformat()}")
            
            # Send notifications via appropriate channels
            try:
                from alerting.alert_system import (
                    determine_alert_channels,
                    create_alert_system_from_settings
                )
                from config.settings import get_settings
                
                # Determine which channels to use
                channels = determine_alert_channels(metric_name, value)
                
                if channels:
                    # Create alert system with configured channels
                    alert_system = create_alert_system_from_settings()
                    
                    # Severity already determined above, reuse it
                    
                    # Format alert message
                    message = f"Alert: {metric_name} = {value:.2f}\n"
                    message += f"Resource: {resource_type}\n"
                    message += f"Timestamp: {timestamp.isoformat()}\n"
                    
                    # Prepare metadata
                    metadata = {
                        "metric_name": metric_name,
                        "value": value,
                        "resource_type": resource_type,
                        "timestamp": timestamp.isoformat(),
                        "severity": severity
                    }
                    
                    # Add recipient info for email/SMS if available
                    try:
                        settings = get_settings()
                        if 'email' in channels and settings.email_recipients:
                            metadata["recipient"] = settings.email_recipients[0]  # Use first recipient
                        if 'sms' in channels:
                            # SMS recipient should be provided via settings or metadata
                            # For now, we'll try to get from settings if available
                            # In practice, this should be configured per alert
                            pass
                    except (ImportError, AttributeError):
                        pass
                    
                    # Send alert through all configured channels
                    # The AlertSystem will only use channels that are actually configured
                    result = alert_system.send_alert(
                        alert_type="metric_alert",
                        severity=severity,
                        message=message,
                        metadata=metadata
                    )
                    
                    # Determine action_taken based on notification results
                    actions = []
                    notification_failed = False
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        for channel_name, channel_result in result.get("channels", {}).items():
                            status = channel_result.get("status", "unknown")
                            if status == "success":
                                # Extract channel type from channel name
                                if "Email" in channel_name:
                                    actions.append("Email Sent")
                                elif "Slack" in channel_name:
                                    actions.append("Slack Sent")
                                elif "SMS" in channel_name:
                                    actions.append("SMS Sent")
                                logger.info(f"Alert notification sent via {channel_name}")
                            elif status == "failed":
                                notification_failed = True
                                logger.warning(f"Alert notification failed via {channel_name}")
                            elif status == "error":
                                notification_failed = True
                                logger.error(f"Alert notification error via {channel_name}: {channel_result.get('error', 'Unknown error')}")
                    except ImportError:
                        pass
                    
                    # Format action_taken
                    if actions:
                        action_taken = ", ".join(actions)
                    elif notification_failed:
                        action_taken = "Alert Triggered (Notifications Failed)"
                    else:
                        action_taken = "Alert Triggered"
                    
                    # Update the alert record with action_taken
                    # We need to update the most recent alert for this metric
                    try:
                        from data_collection.database import connect_to_db
                        update_conn = connect_to_db()
                        if update_conn:
                            update_cursor = update_conn.cursor()
                            # Update the most recent alert with matching metric_name and timestamp
                            update_query = """
                                UPDATE alerts 
                                SET action_taken = %s, severity = %s
                                WHERE id = (
                                    SELECT id FROM alerts 
                                    WHERE metric_name = %s AND timestamp = %s 
                                    ORDER BY id DESC LIMIT 1
                                );
                            """
                            update_cursor.execute(
                                update_query,
                                (action_taken, severity, metric_name, alert_timestamp)
                            )
                            update_conn.commit()
                            update_cursor.close()
                            update_conn.close()
                    except Exception as e:
                        # If update fails, log but don't fail the alert
                        try:
                            from utils.logger import get_logger
                            logger = get_logger(__name__)
                            logger.warning(f"Failed to update alert action_taken: {e}")
                        except ImportError:
                            pass
            except ImportError as e:
                # Notification system not available - log but don't fail
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Notification system not available: {e}")
                except ImportError:
                    print(f"Warning: Notification system not available: {e}")
            except Exception as e:
                # Notification sending failed - log but don't fail alert storage
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"Error sending alert notifications: {e}", exc_info=True)
                except ImportError:
                    print(f"Error sending alert notifications: {e}")
        
        # Return True if alert was stored (even if notifications failed)
        return alert_stored
    
    except ImportError as e:
        # Database module not available
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database module not available for alert storage: {e}")
        except ImportError:
            print(f"Database module not available for alert storage: {e}")
        return False
    except Exception as e:
        # Log error but don't crash
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error triggering alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error triggering alert: {e}")
        return False
