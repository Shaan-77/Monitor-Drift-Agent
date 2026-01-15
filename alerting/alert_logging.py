"""
Logic for logging alerts in the database.

This module provides functionality to log alerts to the database
for persistence and historical tracking.
"""

from typing import Dict, List, Optional
from datetime import datetime


class AlertLogger:
    """Handles logging of alerts to the database."""
    
    def __init__(self, database=None):
        """
        Initialize alert logger.
        
        Args:
            database: Optional database connection for logging
        """
        self.database = database
    
    def log_alert(self, alert: Dict) -> str:
        """
        Log an alert to the database.
        
        Args:
            alert: Dictionary containing alert information
        
        Returns:
            Alert ID of logged alert
        
        TODO: Implement alert logging to database
        """
        if not self.database:
            # TODO: Log warning that database is not available
            return ""
        
        # TODO: Prepare alert data for database insertion
        alert_data = {
            "alert_id": alert.get("alert_id"),
            "alert_type": alert.get("alert_type"),
            "severity": alert.get("severity"),
            "message": alert.get("message"),
            "metadata": alert.get("metadata", {}),
            "status": alert.get("status", "created"),
            "created_at": alert.get("timestamp", datetime.now().isoformat()),
            "channels": alert.get("channels", {})
        }
        
        # TODO: Insert alert into database
        alert_id = self.database.insert_alert(alert_data)
        
        return alert_id
    
    def log_alert_batch(self, alerts: List[Dict]) -> List[str]:
        """
        Log multiple alerts to the database.
        
        Args:
            alerts: List of alert dictionaries
        
        Returns:
            List of alert IDs
        
        TODO: Implement batch alert logging
        """
        alert_ids = []
        for alert in alerts:
            alert_id = self.log_alert(alert)
            if alert_id:
                alert_ids.append(alert_id)
        return alert_ids
    
    def update_alert_status(self, alert_id: str, status: str) -> bool:
        """
        Update the status of a logged alert.
        
        Args:
            alert_id: Unique identifier for the alert
            status: New status (e.g., 'acknowledged', 'resolved', 'dismissed')
        
        Returns:
            True if update successful, False otherwise
        
        TODO: Implement alert status update
        """
        if not self.database:
            return False
        
        # TODO: Update alert status in database
        # self.database.update_alert(alert_id, {"status": status})
        return False


def create_alert_logger(database=None) -> AlertLogger:
    """
    Create an alert logger instance.
    
    Args:
        database: Optional database connection
    
    Returns:
        AlertLogger instance
    
    TODO: Implement alert logger factory
    """
    return AlertLogger(database=database)


def store_alert_in_db(
    metric_name: str,
    value: float,
    timestamp: datetime,
    resource_type: str,
    severity: Optional[str] = None,
    action_taken: Optional[str] = None
) -> bool:
    """
    Store an alert in the PostgreSQL database.
    
    This function stores alert details (metric name, value, timestamp, resource type,
    severity, and action taken) in the PostgreSQL database when an anomaly is detected.
    It wraps the database function from data_collection.database module.
    
    Args:
        metric_name: Name of the metric that triggered the alert (e.g., 'CPU Usage', 'Cloud Cost')
        value: Metric value that exceeded the threshold
        timestamp: Timestamp when the alert was triggered (datetime object)
        resource_type: Resource type identifier (Server, AWS, GCP, Azure)
        severity: Optional severity level (default: "medium" if not provided)
        action_taken: Optional action taken (default: "Alert Triggered" if not provided)
    
    Returns:
        True if storage successful, False otherwise
    
    Example:
        >>> from datetime import datetime
        >>> store_alert_in_db("CPU Usage", 85.5, datetime.utcnow(), "Server", "high", "Email Sent")
        True
    """
    try:
        # Import database function
        from data_collection.database import store_alert_in_db as db_store_alert
        
        # Use defaults if not provided
        if severity is None:
            severity = "medium"
        if action_taken is None:
            action_taken = "Alert Triggered"
        
        # Call the database function
        return db_store_alert(metric_name, value, timestamp, resource_type, severity, action_taken)
    
    except ImportError:
        # Database module not available
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for alert storage")
        except ImportError:
            print("Database module not available for alert storage")
        return False
    except Exception as e:
        # Log error but don't crash
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error storing alert in database: {e}", exc_info=True)
        except ImportError:
            print(f"Error storing alert in database: {e}")
        return False
