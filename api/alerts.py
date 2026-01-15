"""
API endpoints for triggering and managing alerts.

This module provides REST API endpoints to create, retrieve,
and manage alerts in the system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

# Import database functions
try:
    from data_collection.database import store_alert_in_db, get_alerts_from_db, connect_to_db, PSYCOPG2_AVAILABLE
except ImportError:
    def store_alert_in_db(*args, **kwargs):
        raise RuntimeError("Database module not available")
    def get_alerts_from_db(*args, **kwargs):
        return {"alerts": [], "total": 0, "error": "Database module not available"}
    PSYCOPG2_AVAILABLE = False

# Import settings for configuration
try:
    from config.settings import get_settings
except ImportError:
    def get_settings():
        return None

# Create router
router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# Pydantic models for request/response
class AlertCreateRequest(BaseModel):
    """Request model for creating an alert."""
    metric_name: str = Field(..., description="Name of the metric that triggered the alert")
    value: float = Field(..., description="Metric value that exceeded the threshold")
    resource_type: str = Field(..., description="Resource type (Server, AWS, GCP, Azure)")
    severity: str = Field(default="medium", description="Alert severity (low, medium, high, critical)")
    action_taken: Optional[str] = Field(default="Alert Triggered", description="Action taken after alert")


class AlertUpdateRequest(BaseModel):
    """Request model for updating alert status."""
    status: Optional[str] = Field(None, description="Alert status (acknowledged, resolved, dismissed)")
    severity: Optional[str] = Field(None, description="Updated severity level")
    action_taken: Optional[str] = Field(None, description="Updated action taken")


# API key verification
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-KEY")) -> bool:
    """
    Verify API key for alert management endpoints.
    
    Args:
        x_api_key: API key from X-API-KEY header
    
    Returns:
        True if authentication passes or is disabled
    
    Raises:
        HTTPException: If API key is invalid or missing when required
    """
    settings = get_settings()
    if not settings or not settings.api_key_enabled:
        return True  # No auth required if disabled
    
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


@router.post("", status_code=201)
async def create_alert(
    request: AlertCreateRequest,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Create a new alert and store it in the database.
    
    Args:
        request: AlertCreateRequest containing alert data
    
    Returns:
        Dictionary containing created alert information
    
    Raises:
        HTTPException: 400 if validation fails
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database storage fails
    """
    if not PSYCOPG2_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Database module not available. Please install psycopg2-binary"
        )
    
    try:
        # Validate severity
        valid_severities = ['low', 'medium', 'high', 'critical']
        if request.severity.lower() not in valid_severities:
            raise HTTPException(
                status_code=400,
                detail=f"severity must be one of: {', '.join(valid_severities)}"
            )
        
        # Validate resource_type
        valid_resource_types = ['Server', 'AWS', 'GCP', 'Azure']
        if request.resource_type not in valid_resource_types:
            raise HTTPException(
                status_code=400,
                detail=f"resource_type must be one of: {', '.join(valid_resource_types)}"
            )
        
        # Store alert in database
        success = store_alert_in_db(
            metric_name=request.metric_name,
            value=request.value,
            timestamp=datetime.utcnow(),
            resource_type=request.resource_type,
            severity=request.severity.lower(),
            action_taken=request.action_taken or "Alert Triggered"
        )
        
        if success:
            return {
                "status": "success",
                "message": "Alert created and stored successfully",
                "alert": {
                    "metric_name": request.metric_name,
                    "value": request.value,
                    "resource_type": request.resource_type,
                    "severity": request.severity.lower(),
                    "action_taken": request.action_taken or "Alert Triggered",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to store alert in database"
            )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating alert: {str(e)}"
        )


@router.get("")
async def list_alerts(
    alert_id: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    severity: Optional[str] = None,
    resource_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    List alerts with optional filtering.
    
    Query Parameters:
        alert_id: Optional specific alert ID to retrieve
        start_time: Optional ISO format timestamp for start of time range
        end_time: Optional ISO format timestamp for end of time range
        severity: Optional filter by severity (low, medium, high, critical)
        resource_type: Optional filter by resource type (Server, AWS, GCP, Azure)
        limit: Maximum number of records to return (default: 100, max: 1000)
        offset: Number of records to skip for pagination (default: 0)
    
    Headers:
        X-API-KEY: API key for authentication (required if API_KEY_ENABLED=true)
    
    Returns:
        JSON response containing:
            - alerts: List of alert dictionaries
            - total: Total number of records matching filters
            - limit: Current limit value
            - offset: Current offset value
    
    Raises:
        HTTPException: 400 if query parameters are invalid
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database query fails
    """
    try:
        # Validate limit
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="limit must be between 1 and 1000"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=400,
                detail="offset must be greater than or equal to 0"
            )
        
        # Validate severity if provided
        if severity:
            valid_severities = ['low', 'medium', 'high', 'critical']
            if severity.lower() not in valid_severities:
                raise HTTPException(
                    status_code=400,
                    detail=f"severity must be one of: {', '.join(valid_severities)}"
                )
        
        # Validate resource_type if provided
        if resource_type:
            valid_resource_types = ['Server', 'AWS', 'GCP', 'Azure']
            if resource_type not in valid_resource_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"resource_type must be one of: {', '.join(valid_resource_types)}"
                )
        
        # Parse timestamps if provided
        start_datetime = None
        end_datetime = None
        
        if start_time:
            try:
                start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                if start_datetime.tzinfo is not None:
                    start_datetime = start_datetime.astimezone().replace(tzinfo=None)
            except (ValueError, AttributeError):
                raise HTTPException(
                    status_code=400,
                    detail="start_time must be in ISO format (e.g., 2024-01-01T00:00:00Z)"
                )
        
        if end_time:
            try:
                end_datetime = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                if end_datetime.tzinfo is not None:
                    end_datetime = end_datetime.astimezone().replace(tzinfo=None)
            except (ValueError, AttributeError):
                raise HTTPException(
                    status_code=400,
                    detail="end_time must be in ISO format (e.g., 2024-01-01T00:00:00Z)"
                )
        
        # Fetch alerts from database
        result = get_alerts_from_db(
            alert_id=alert_id,
            start_time=start_datetime,
            end_time=end_datetime,
            severity=severity.lower() if severity else None,
            resource_type=resource_type,
            limit=limit,
            offset=offset
        )
        
        # Check if result contains an error
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=f"Database query failed: {result.get('error', 'Unknown error')}"
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving alerts: {str(e)}"
        )


@router.get("/{alert_id}")
async def get_alert(
    alert_id: int,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Retrieve a specific alert by ID.
    
    Args:
        alert_id: Unique identifier for the alert
    
    Returns:
        Dictionary containing alert information
    
    Raises:
        HTTPException: 404 if alert not found
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database query fails
    """
    try:
        result = get_alerts_from_db(alert_id=alert_id, limit=1)
        
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=f"Database query failed: {result.get('error', 'Unknown error')}"
            )
        
        if not result.get('alerts') or len(result['alerts']) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Alert with ID {alert_id} not found"
            )
        
        return {
            "alert": result['alerts'][0],
            "total": 1
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving alert: {str(e)}"
        )


@router.put("/{alert_id}")
async def update_alert(
    alert_id: int,
    request: AlertUpdateRequest,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Update an alert (currently only supports updating action_taken via database).
    
    Note: Full update functionality requires additional database schema support.
    This endpoint provides basic update capability.
    
    Args:
        alert_id: Unique identifier for the alert
        request: AlertUpdateRequest containing fields to update
    
    Returns:
        Dictionary containing updated alert information
    
    Raises:
        HTTPException: 404 if alert not found
        HTTPException: 400 if validation fails
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database update fails
    """
    if not PSYCOPG2_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Database module not available. Please install psycopg2-binary"
        )
    
    try:
        # First, check if alert exists
        result = get_alerts_from_db(alert_id=alert_id, limit=1)
        
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=f"Database query failed: {result.get('error', 'Unknown error')}"
            )
        
        if not result.get('alerts') or len(result['alerts']) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Alert with ID {alert_id} not found"
            )
        
        existing_alert = result['alerts'][0]
        
        # Update alert in database
        conn = None
        try:
            from data_collection.database import connect_to_db, create_alerts_schema
            conn = connect_to_db()
            if conn is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to connect to database"
                )
            
            # Ensure schema exists
            create_alerts_schema(conn)
            
            cursor = conn.cursor()
            
            # Build update query
            updates = []
            params = []
            
            if request.action_taken:
                updates.append("action_taken = %s")
                params.append(request.action_taken)
            
            # Note: severity and status updates would require additional schema columns
            # For now, we only support action_taken updates
            
            if not updates:
                cursor.close()
                conn.close()
                return {
                    "status": "success",
                    "message": "No updates provided",
                    "alert": existing_alert
                }
            
            # Execute update
            update_query = f"UPDATE alerts SET {', '.join(updates)} WHERE id = %s"
            params.append(alert_id)
            cursor.execute(update_query, params)
            conn.commit()
            cursor.close()
            conn.close()
            
            # Fetch updated alert
            updated_result = get_alerts_from_db(alert_id=alert_id, limit=1)
            updated_alert = updated_result['alerts'][0] if updated_result.get('alerts') else existing_alert
            
            return {
                "status": "success",
                "message": "Alert updated successfully",
                "alert": updated_alert
            }
        
        except Exception as e:
            if conn:
                conn.close()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update alert: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error updating alert: {str(e)}"
        )


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: int,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Delete an alert from the database.
    
    Args:
        alert_id: Unique identifier for the alert
    
    Returns:
        Dictionary containing deletion confirmation
    
    Raises:
        HTTPException: 404 if alert not found
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database deletion fails
    """
    if not PSYCOPG2_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Database module not available. Please install psycopg2-binary"
        )
    
    try:
        # First, check if alert exists
        result = get_alerts_from_db(alert_id=alert_id, limit=1)
        
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=f"Database query failed: {result.get('error', 'Unknown error')}"
            )
        
        if not result.get('alerts') or len(result['alerts']) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Alert with ID {alert_id} not found"
            )
        
        # Delete alert from database
        conn = None
        try:
            from data_collection.database import connect_to_db
            conn = connect_to_db()
            if conn is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to connect to database"
                )
            
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts WHERE id = %s", (alert_id,))
            conn.commit()
            deleted_count = cursor.rowcount
            cursor.close()
            conn.close()
            
            if deleted_count == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Alert with ID {alert_id} not found"
                )
            
            return {
                "status": "success",
                "message": f"Alert {alert_id} deleted successfully",
                "alert_id": alert_id
            }
        
        except HTTPException:
            raise
        except Exception as e:
            if conn:
                conn.close()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete alert: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error deleting alert: {str(e)}"
        )
