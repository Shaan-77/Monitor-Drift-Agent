"""
Logic for retrieving alert history from the database.

This module provides functionality to query and retrieve historical
alert data from the database.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta


class AlertHistory:
    """Handles retrieval of alert history from the database."""
    
    def __init__(self, database=None):
        """
        Initialize alert history manager.
        
        Args:
            database: Optional database connection for querying
        """
        self.database = database
    
    def get_alert(self, alert_id: str) -> Optional[Dict]:
        """
        Retrieve a specific alert by ID.
        
        Args:
            alert_id: Unique identifier for the alert
        
        Returns:
            Alert dictionary or None if not found
        
        TODO: Implement alert retrieval from database
        """
        if not self.database:
            return None
        
        # TODO: Retrieve alert from database
        alert = self.database.get_alert(alert_id)
        return alert
    
    def list_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        List alerts with optional filtering.
        
        Args:
            start_time: Optional start of time range
            end_time: Optional end of time range
            status: Optional filter by alert status
            severity: Optional filter by severity level
            alert_type: Optional filter by alert type
            limit: Maximum number of alerts to return
            offset: Offset for pagination
        
        Returns:
            List of alert dictionaries
        
        TODO: Implement alert listing with filters
        """
        if not self.database:
            return []
        
        # TODO: Build query with filters
        alerts = self.database.list_alerts(
            status=status,
            severity=severity,
            limit=limit
        )
        
        # TODO: Apply additional filters (time range, alert_type) if needed
        filtered_alerts = alerts
        
        if start_time or end_time:
            # TODO: Filter by time range
            pass
        
        if alert_type:
            # TODO: Filter by alert type
            filtered_alerts = [a for a in filtered_alerts if a.get("alert_type") == alert_type]
        
        return filtered_alerts[offset:offset + limit]
    
    def get_alert_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """
        Get statistics about alerts in a time range.
        
        Args:
            start_time: Optional start of time range
            end_time: Optional end of time range
        
        Returns:
            Dictionary containing alert statistics
        
        TODO: Implement alert statistics calculation
        """
        alerts = self.list_alerts(start_time=start_time, end_time=end_time, limit=10000)
        
        stats = {
            "total_alerts": len(alerts),
            "by_severity": {},
            "by_type": {},
            "by_status": {},
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            }
        }
        
        # TODO: Calculate statistics
        for alert in alerts:
            severity = alert.get("severity", "unknown")
            alert_type = alert.get("alert_type", "unknown")
            status = alert.get("status", "unknown")
            
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            stats["by_type"][alert_type] = stats["by_type"].get(alert_type, 0) + 1
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        
        return stats
    
    def search_alerts(self, query: str, limit: int = 100) -> List[Dict]:
        """
        Search alerts by message content or metadata.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        
        Returns:
            List of matching alert dictionaries
        
        TODO: Implement alert search functionality
        """
        if not self.database:
            return []
        
        # TODO: Implement search query
        # This would typically involve full-text search in the database
        all_alerts = self.list_alerts(limit=limit * 10)  # Get more to filter
        
        # Simple text search in message
        matching_alerts = [
            alert for alert in all_alerts
            if query.lower() in alert.get("message", "").lower()
        ]
        
        return matching_alerts[:limit]


def get_alert_history(
    severity: Optional[str] = None,
    resource_type: Optional[str] = None,
    time_range: Optional[Union[datetime, timedelta, int]] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Retrieve alert history from the PostgreSQL database with optional filtering.
    
    This function queries the alerts table and returns a list of alert dictionaries
    matching the specified filters. Results are ordered by timestamp (most recent first).
    
    Args:
        severity: Optional filter by severity level ("critical", "high", "medium", "low")
        resource_type: Optional filter by resource type ("Server", "AWS", "GCP", "Azure", "Cloud")
        time_range: Optional time range filter. Can be:
            - datetime object: alerts >= this timestamp
            - timedelta object: alerts >= (now - timedelta)
            - int (days): alerts >= (now - days)
        limit: Maximum number of alerts to return (default: 100)
        offset: Offset for pagination (default: 0)
    
    Returns:
        List of alert dictionaries with keys: id, metric_name, value, timestamp,
        resource_type, severity, action_taken
    
    Example:
        >>> from datetime import datetime, timedelta
        >>> # Get all alerts
        >>> alerts = get_alert_history()
        >>> # Filter by severity
        >>> alerts = get_alert_history(severity="high")
        >>> # Filter by time range (last 30 days)
        >>> alerts = get_alert_history(time_range=30)
        >>> # Combined filters
        >>> alerts = get_alert_history(severity="critical", resource_type="Server", time_range=timedelta(days=7))
    """
    conn = None
    cursor = None
    
    try:
        # Import database connection function
        from data_collection.database import connect_to_db
        
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to establish database connection for alert history")
            except ImportError:
                print("Error: Failed to establish database connection for alert history")
            return []
        
        cursor = conn.cursor()
        
        # Build WHERE clause dynamically
        where_clauses = []
        params = []
        
        # Handle severity filter
        if severity:
            if not isinstance(severity, str):
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Invalid severity type: {type(severity)}, expected str")
                except ImportError:
                    print(f"Warning: Invalid severity type: {type(severity)}")
            else:
                where_clauses.append("severity = %s")
                params.append(severity)
        
        # Handle resource_type filter
        if resource_type:
            if not isinstance(resource_type, str):
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Invalid resource_type type: {type(resource_type)}, expected str")
                except ImportError:
                    print(f"Warning: Invalid resource_type type: {type(resource_type)}")
            else:
                where_clauses.append("resource_type = %s")
                params.append(resource_type)
        
        # Handle time_range filter
        if time_range is not None:
            try:
                cutoff_timestamp = None
                
                if isinstance(time_range, datetime):
                    # Use datetime directly
                    cutoff_timestamp = time_range
                    # Ensure UTC and timezone-naive
                    if cutoff_timestamp.tzinfo is not None:
                        cutoff_timestamp = cutoff_timestamp.astimezone().replace(tzinfo=None)
                elif isinstance(time_range, timedelta):
                    # Calculate: now - timedelta
                    cutoff_timestamp = datetime.utcnow() - time_range
                elif isinstance(time_range, int):
                    # Treat as days
                    cutoff_timestamp = datetime.utcnow() - timedelta(days=time_range)
                else:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.warning(f"Invalid time_range type: {type(time_range)}, expected datetime, timedelta, or int")
                    except ImportError:
                        print(f"Warning: Invalid time_range type: {type(time_range)}")
                    cutoff_timestamp = None
                
                if cutoff_timestamp:
                    where_clauses.append("timestamp >= %s")
                    params.append(cutoff_timestamp)
            except Exception as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Error processing time_range: {e}")
                except ImportError:
                    print(f"Warning: Error processing time_range: {e}")
        
        # Build query
        query = """
            SELECT id, metric_name, value, timestamp, resource_type, severity, action_taken
            FROM alerts
        """
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += " ORDER BY timestamp DESC"
        query += " LIMIT %s OFFSET %s"
        
        params.extend([limit, offset])
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        alerts = []
        for row in rows:
            alert_dict = {
                'id': row[0],
                'metric_name': row[1],
                'value': float(row[2]) if row[2] is not None else None,
                'timestamp': row[3],
                'resource_type': row[4],
                'severity': row[5],
                'action_taken': row[6]
            }
            alerts.append(alert_dict)
        
        cursor.close()
        conn.close()
        
        return alerts
    
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for alert history retrieval")
        except ImportError:
            print("Error: Database module not available for alert history retrieval")
        return []
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error retrieving alert history: {e}", exc_info=True)
        except ImportError:
            print(f"Error retrieving alert history: {e}")
        return []
    finally:
        # Ensure connection is closed
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def create_alert_history(database=None) -> AlertHistory:
    """
    Create an alert history manager instance.
    
    Args:
        database: Optional database connection
    
    Returns:
        AlertHistory instance
    
    TODO: Implement alert history factory
    """
    return AlertHistory(database=database)
