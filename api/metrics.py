"""
API endpoints for collecting system metrics.

This module provides REST API endpoints to collect and retrieve
system metrics including CPU, memory, disk, and network statistics.
"""

from typing import Dict, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse

# Try to import slowapi for rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    Limiter = None
    get_remote_address = None
    RateLimitExceeded = Exception

# Import metric collection functions
try:
    from data_collection.system_metrics import (
        get_cpu_usage,
        get_memory_usage,
        get_network_traffic
    )
except ImportError:
    # Fallback if module not available
    def get_cpu_usage():
        raise RuntimeError("System metrics module not available")
    def get_memory_usage():
        raise RuntimeError("System metrics module not available")
    def get_network_traffic():
        raise RuntimeError("System metrics module not available")

# Import settings for configuration
try:
    from config.settings import get_settings
except ImportError:
    def get_settings():
        return None

# Import database functions
try:
    from data_collection.database import get_metrics_from_db, store_metrics
    from data_collection.system_metrics import collect_and_store_metrics
except ImportError:
    def get_metrics_from_db(*args, **kwargs):
        raise RuntimeError("Database module not available")
    def store_metrics(*args, **kwargs):
        raise RuntimeError("Database module not available")
    def collect_and_store_metrics():
        raise RuntimeError("System metrics module not available")

# Create router
router = APIRouter()

# Rate limiter will be initialized in main.py and passed to router
# For now, we'll use a simple approach that works with FastAPI


def get_rate_limit():
    """
    Get the rate limit string from settings.
    
    Returns:
        Rate limit string or None if rate limiting is disabled
    """
    if not SLOWAPI_AVAILABLE:
        return None
    
    settings = get_settings()
    if settings and settings.api_enable_rate_limit:
        return settings.api_rate_limit
    
    return None


# Apply rate limiting decorator conditionally
def apply_rate_limit(func):
    """Apply rate limiting if enabled."""
    rate_limit = get_rate_limit()
    if rate_limit and limiter:
        return limiter.limit(rate_limit)(func)
    return func


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY")
) -> bool:
    """
    Verify API key from request header.
    
    Args:
        x_api_key: API key from X-API-KEY header
    
    Returns:
        True if API key is valid
    
    Raises:
        HTTPException: 401 if API key is missing or invalid when authentication is enabled
    """
    settings = get_settings()
    if settings is None:
        # If settings not available, allow access (fail open for development)
        return True
    
    # Check if API key authentication is enabled
    if not settings.api_key_enabled:
        # Authentication is disabled, allow access
        return True
    
    # Authentication is enabled, verify API key
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Please provide X-API-KEY header."
        )
    
    # Validate API key
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Access denied."
        )
    
    return True


@router.get("/api/metrics")
@apply_rate_limit
async def get_metrics(request: Request):
    """
    Retrieve the most recent system metrics (CPU, memory, network).
    
    This endpoint is rate-limited based on configuration.
    
    Returns:
        JSON response containing:
            - cpu_usage: CPU metrics dictionary
            - memory_usage: Memory metrics dictionary
            - network_traffic: Network metrics dictionary
            - timestamp: ISO format timestamp
    
    Raises:
        HTTPException: 500 if metric collection fails
        RateLimitExceeded: 429 if rate limit is exceeded (handled by main.py)
    """
    try:
        # Collect all metrics
        cpu_data = get_cpu_usage()
        memory_data = get_memory_usage()
        network_data = get_network_traffic()
        
        # Get unified timestamp
        current_timestamp = datetime.now().isoformat()
        
        # Return structured JSON response
        return {
            "cpu_usage": cpu_data,
            "memory_usage": memory_data,
            "network_traffic": network_data,
            "timestamp": current_timestamp
        }
    
    except RuntimeError as e:
        # Handle runtime errors (e.g., psutil not available)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect metrics: {str(e)}"
        )
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error collecting metrics: {str(e)}"
        )


@router.post("/api/metrics/collect")
@apply_rate_limit
async def collect_and_store_metrics_endpoint(request: Request):
    """
    Collect system metrics and store them in the database.
    
    This endpoint collects current CPU, memory, and network metrics
    and stores them in the PostgreSQL database for historical tracking.
    
    Returns:
        JSON response containing:
            - status: "success" or "error"
            - message: Status message
            - metrics: Collected metrics data
            - timestamp: ISO format timestamp
    
    Raises:
        HTTPException: 500 if metric collection or storage fails
        RateLimitExceeded: 429 if rate limit is exceeded
    """
    try:
        # Collect all metrics
        cpu_data = get_cpu_usage()
        memory_data = get_memory_usage()
        network_data = get_network_traffic()
        
        # Get unified timestamp
        current_timestamp = datetime.now().isoformat()
        
        # Store metrics in database
        storage_success = store_metrics(cpu_data, memory_data, network_data)
        
        if storage_success:
            return {
                "status": "success",
                "message": "Metrics collected and stored successfully",
                "metrics": {
                    "cpu_usage": cpu_data,
                    "memory_usage": memory_data,
                    "network_traffic": network_data
                },
                "timestamp": current_timestamp
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to store metrics in database"
            )
    
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect metrics: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


# Legacy functions kept for backward compatibility
def get_metrics_legacy(metric_type: Optional[str] = None) -> Dict:
    """
    Retrieve system metrics (legacy function).
    
    Args:
        metric_type: Optional filter for specific metric type
                    (e.g., 'cpu', 'memory', 'disk', 'network')
    
    Returns:
        Dictionary containing requested metrics
    
    TODO: Implement actual metric collection logic
    """
    # TODO: Implement metric retrieval from data_collection module
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": {}
    }


def collect_metrics_legacy() -> Dict:
    """
    Trigger immediate metric collection (legacy function).
    
    Returns:
        Dictionary containing collected metrics
    
    TODO: Implement actual metric collection trigger
    """
    # TODO: Implement metric collection trigger
    return {
        "status": "success",
        "message": "Metrics collection triggered",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/api/metrics/cloud-and-server")
@apply_rate_limit
async def get_cloud_and_server_metrics(
    timestamp: Optional[str] = None,
    resource_type: Optional[str] = None,
    metric_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict:
    """
    Retrieve stored cloud and server metrics from the database.
    
    This endpoint retrieves metrics from the cloud_server_metrics table
    with optional filtering by timestamp, resource_type, and metric_type.
    Supports pagination via limit and offset parameters.
    
    Query Parameters:
        timestamp: Optional ISO format timestamp string. Filters rows where timestamp >= timestamp
        resource_type: Optional filter (Server, AWS, GCP, Azure)
        metric_type: Optional filter (CPU, Memory, Network, Cloud Compute, Cloud Storage, Cloud Bandwidth)
        limit: Maximum number of records to return (default: 50, max: 1000)
        offset: Number of records to skip for pagination (default: 0)
    
    Headers:
        X-API-KEY: API key for authentication (required if API_KEY_ENABLED=true)
    
    Returns:
        JSON response containing:
            - metrics: List of metric dictionaries
            - total: Total number of records matching filters
            - limit: Current limit value
            - offset: Current offset value
    
    Raises:
        HTTPException: 400 if query parameters are invalid
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database query fails
    """
    try:
        # Validate limit parameter (must be between 1 and 1000)
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="limit must be between 1 and 1000"
            )
        
        # Validate offset parameter (must be >= 0)
        if offset < 0:
            raise HTTPException(
                status_code=400,
                detail="offset must be greater than or equal to 0"
            )
        
        # Validate resource_type if provided
        valid_resource_types = ['Server', 'AWS', 'GCP', 'Azure']
        if resource_type and resource_type not in valid_resource_types:
            raise HTTPException(
                status_code=400,
                detail=f"resource_type must be one of: {', '.join(valid_resource_types)}"
            )
        
        # Validate metric_type if provided
        valid_metric_types = [
            'CPU', 'Memory', 'Network',
            'Cloud Compute', 'Cloud Storage', 'Cloud Bandwidth'
        ]
        if metric_type and metric_type not in valid_metric_types:
            raise HTTPException(
                status_code=400,
                detail=f"metric_type must be one of: {', '.join(valid_metric_types)}"
            )
        
        # Validate timestamp format if provided
        if timestamp:
            try:
                # Try to parse ISO format timestamp to validate
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                raise HTTPException(
                    status_code=400,
                    detail="timestamp must be in ISO format (e.g., 2024-01-01T00:00:00Z)"
                )
        
        # Fetch metrics from database
        try:
            result = get_metrics_from_db(
                timestamp=timestamp,
                resource_type=resource_type,
                metric_type=metric_type,
                limit=limit,
                offset=offset
            )
        except RuntimeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database module error: {str(e)}"
            )
        
        # Check if result contains an error
        if 'error' in result:
            # Database query failed, but we got a result structure
            raise HTTPException(
                status_code=500,
                detail=f"Database query failed: {result.get('error', 'Unknown error')}"
            )
        
        # Return successful response
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, auth errors)
        raise
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving metrics: {str(e)}"
        )
