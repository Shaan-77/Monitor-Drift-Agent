"""
API endpoints for managing policies and thresholds.

This module provides REST API endpoints to manage policies and thresholds
in the system, including dynamic threshold updates.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

# Import settings for configuration
try:
    from config.settings import get_settings
except ImportError:
    def get_settings():
        return None

# Import policy management functions
try:
    from policy_management.policy_definition import (
        ResourcePolicy,
        store_resource_policy_in_db,
        get_resource_policy_from_db,
        list_resource_policies_from_db,
        update_resource_policy_in_db,
        delete_resource_policy_from_db,
        create_resource_policy,
        PSYCOPG2_AVAILABLE as POLICY_DB_AVAILABLE
    )
except ImportError:
    ResourcePolicy = None
    store_resource_policy_in_db = None
    get_resource_policy_from_db = None
    list_resource_policies_from_db = None
    update_resource_policy_in_db = None
    delete_resource_policy_from_db = None
    create_resource_policy = None
    POLICY_DB_AVAILABLE = False

# Create router
router = APIRouter(prefix="/api/policies", tags=["policies"])


# Pydantic models for request/response
class ThresholdUpdateRequest(BaseModel):
    """Request model for updating thresholds."""
    cpu_usage_threshold: Optional[float] = Field(None, ge=0, le=100, description="CPU usage threshold (0-100%)")
    cpu_threshold_duration: Optional[int] = Field(None, ge=0, description="CPU threshold duration in minutes")
    cloud_cost_threshold: Optional[float] = Field(None, gt=0, description="Cloud cost threshold in dollars per day")
    memory_usage_threshold: Optional[float] = Field(None, ge=0, le=100, description="Memory usage threshold (0-100%)")
    disk_usage_threshold: Optional[float] = Field(None, ge=0, le=100, description="Disk usage threshold (0-100%)")
    network_bandwidth_threshold: Optional[float] = Field(None, gt=0, description="Network bandwidth threshold")


class PolicyCreateRequest(BaseModel):
    """Request model for creating a resource policy."""
    resource_name: str = Field(..., description="Resource name (e.g., 'Server CPU', 'AWS EC2')")
    threshold_value: float = Field(..., gt=0, description="Threshold value")
    threshold_type: str = Field(..., description="Threshold type (usage or cost)")
    duration: int = Field(..., ge=0, description="Duration in minutes (0 = immediate trigger)")
    enabled: bool = Field(default=True, description="Whether the policy is enabled")


class PolicyUpdateRequest(BaseModel):
    """Request model for updating a resource policy."""
    threshold_value: Optional[float] = Field(None, gt=0, description="Updated threshold value")
    duration: Optional[int] = Field(None, ge=0, description="Updated duration in minutes (0 = immediate trigger)")
    enabled: Optional[bool] = Field(None, description="Updated enabled status")


# API key verification (similar to metrics.py)
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-KEY")) -> bool:
    """
    Verify API key for threshold management endpoints.
    
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


@router.get("/thresholds")
async def get_thresholds(api_key_valid: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """
    Get current threshold values.
    
    Returns:
        Dictionary containing all current threshold values
    
    Raises:
        HTTPException: If settings are not available
    """
    settings = get_settings()
    if not settings:
        raise HTTPException(status_code=500, detail="Settings not available")
    return settings.get_thresholds()


@router.post("/thresholds")
async def update_thresholds(
    request: ThresholdUpdateRequest,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Update threshold values with validation.
    
    Args:
        request: ThresholdUpdateRequest containing threshold values to update
    
    Returns:
        Dictionary with success message and updated thresholds
    
    Raises:
        HTTPException: If validation fails or settings are not available
    """
    settings = get_settings()
    if not settings:
        raise HTTPException(status_code=500, detail="Settings not available")
    
    try:
        # Convert Pydantic model to dict, filtering None values
        updates = {k: v for k, v in request.dict().items() if v is not None}
        if not updates:
            raise HTTPException(status_code=400, detail="No threshold values provided")
        
        updated = settings.update_thresholds(**updates)
        return {
            "message": "Thresholds updated successfully",
            "thresholds": updated
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating thresholds: {str(e)}")


@router.get("")
async def list_policies(
    enabled_only: bool = False,
    threshold_type: Optional[str] = None,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    List all resource policies.
    
    Query Parameters:
        enabled_only: If True, only return enabled policies (default: False)
        threshold_type: Optional filter by threshold type (usage or cost)
    
    Headers:
        X-API-KEY: API key for authentication (required if API_KEY_ENABLED=true)
    
    Returns:
        JSON response containing list of policies
    
    Raises:
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database query fails
    """
    if not POLICY_DB_AVAILABLE or list_resource_policies_from_db is None:
        raise HTTPException(
            status_code=500,
            detail="Policy management module not available"
        )
    
    try:
        policies = list_resource_policies_from_db(
            enabled_only=enabled_only,
            threshold_type=threshold_type
        )
        
        return {
            "policies": [policy.to_dict() for policy in policies],
            "total": len(policies)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing policies: {str(e)}"
        )


@router.post("", status_code=201)
async def create_policy(
    request: PolicyCreateRequest,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Create a new resource policy.
    
    Args:
        request: PolicyCreateRequest containing policy data
    
    Returns:
        Dictionary containing created policy information
    
    Raises:
        HTTPException: 400 if validation fails
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database storage fails
    """
    if not POLICY_DB_AVAILABLE or create_resource_policy is None:
        raise HTTPException(
            status_code=500,
            detail="Policy management module not available"
        )
    
    try:
        # Validate threshold_type
        valid_types = ['usage', 'cost']
        if request.threshold_type.lower() not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"threshold_type must be one of: {', '.join(valid_types)}"
            )
        
        # Create policy with enabled status
        policy = create_resource_policy(
            resource_name=request.resource_name,
            threshold_value=request.threshold_value,
            threshold_type=request.threshold_type.lower(),
            duration=request.duration,
            enabled=request.enabled,
            store_in_db=False  # We'll store manually to get the ID
        )
        
        # Store in database
        policy_id = store_resource_policy_in_db(policy)
        
        if policy_id:
            # Retrieve stored policy to return complete data
            stored_policy = get_resource_policy_from_db(policy_id=policy_id)
            if stored_policy:
                return {
                    "status": "success",
                    "message": "Policy created successfully",
                    "policy": stored_policy.to_dict()
                }
        
        raise HTTPException(
            status_code=500,
            detail="Failed to store policy in database"
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating policy: {str(e)}"
        )


@router.get("/{policy_id}")
async def get_policy(
    policy_id: int,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Retrieve a specific policy by ID.
    
    Args:
        policy_id: Unique identifier for the policy
    
    Returns:
        Dictionary containing policy information
    
    Raises:
        HTTPException: 404 if policy not found
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database query fails
    """
    if not POLICY_DB_AVAILABLE or get_resource_policy_from_db is None:
        raise HTTPException(
            status_code=500,
            detail="Policy management module not available"
        )
    
    try:
        policy = get_resource_policy_from_db(policy_id=policy_id)
        
        if policy is None:
            raise HTTPException(
                status_code=404,
                detail=f"Policy with ID {policy_id} not found"
            )
        
        return {
            "policy": policy.to_dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving policy: {str(e)}"
        )


@router.put("/{policy_id}")
async def update_policy(
    policy_id: int,
    request: PolicyUpdateRequest,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Update an existing policy.
    
    Args:
        policy_id: Unique identifier for the policy
        request: PolicyUpdateRequest containing fields to update
    
    Returns:
        Dictionary containing updated policy information
    
    Raises:
        HTTPException: 404 if policy not found
        HTTPException: 400 if validation fails
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database update fails
    """
    if not POLICY_DB_AVAILABLE or update_resource_policy_in_db is None:
        raise HTTPException(
            status_code=500,
            detail="Policy management module not available"
        )
    
    try:
        # Check if at least one field is being updated
        if request.threshold_value is None and request.duration is None and request.enabled is None:
            raise HTTPException(
                status_code=400,
                detail="At least one field (threshold_value, duration, enabled) must be provided"
            )
        
        # Update policy
        updated_policy = update_resource_policy_in_db(
            policy_id=policy_id,
            threshold_value=request.threshold_value,
            duration=request.duration,
            enabled=request.enabled
        )
        
        if updated_policy is None:
            raise HTTPException(
                status_code=404,
                detail=f"Policy with ID {policy_id} not found"
            )
        
        return {
            "status": "success",
            "message": "Policy updated successfully",
            "policy": updated_policy.to_dict()
        }
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error updating policy: {str(e)}"
        )


@router.delete("/{policy_id}")
async def delete_policy(
    policy_id: int,
    api_key_valid: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Delete a policy from the database.
    
    Args:
        policy_id: Unique identifier for the policy
    
    Returns:
        Dictionary containing deletion confirmation
    
    Raises:
        HTTPException: 404 if policy not found
        HTTPException: 401 if API key authentication fails
        HTTPException: 500 if database deletion fails
    """
    if not POLICY_DB_AVAILABLE or delete_resource_policy_from_db is None:
        raise HTTPException(
            status_code=500,
            detail="Policy management module not available"
        )
    
    try:
        success = delete_resource_policy_from_db(policy_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Policy {policy_id} deleted successfully",
                "policy_id": policy_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Policy with ID {policy_id} not found"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error deleting policy: {str(e)}"
        )


# Legacy policy management functions (kept for backward compatibility)
def create_policy(
    name: str,
    description: str,
    rules: Dict,
    enabled: bool = True
) -> Dict:
    """
    Create a new policy.
    
    Args:
        name: Policy name
        description: Policy description
        rules: Policy rules definition
        enabled: Whether the policy is enabled
    
    Returns:
        Dictionary containing created policy information
    
    TODO: Implement policy creation logic
    """
    # TODO: Implement policy creation in policy_management module
    return {
        "policy_id": None,  # TODO: Generate unique policy ID
        "name": name,
        "description": description,
        "rules": rules,
        "enabled": enabled,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def get_policy(policy_id: str) -> Dict:
    """
    Retrieve a specific policy by ID.
    
    Args:
        policy_id: Unique identifier for the policy
    
    Returns:
        Dictionary containing policy information
    
    TODO: Implement policy retrieval from database
    """
    # TODO: Implement policy retrieval
    return {}


def list_policies(enabled_only: bool = False) -> List[Dict]:
    """
    List all policies.
    
    Args:
        enabled_only: If True, only return enabled policies
    
    Returns:
        List of policy dictionaries
    
    TODO: Implement policy listing from database
    """
    # TODO: Implement policy listing
    return []


def update_policy(
    policy_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    rules: Optional[Dict] = None,
    enabled: Optional[bool] = None
) -> Dict:
    """
    Update an existing policy.
    
    Args:
        policy_id: Unique identifier for the policy
        name: Optional new policy name
        description: Optional new policy description
        rules: Optional new policy rules
        enabled: Optional new enabled status
    
    Returns:
        Dictionary containing updated policy information
    
    TODO: Implement policy update logic
    """
    # TODO: Implement policy update
    return {
        "policy_id": policy_id,
        "updated_at": datetime.now().isoformat()
    }


def delete_policy(policy_id: str) -> Dict:
    """
    Delete a policy.
    
    Args:
        policy_id: Unique identifier for the policy
    
    Returns:
        Dictionary containing deletion confirmation
    
    TODO: Implement policy deletion
    """
    # TODO: Implement policy deletion
    return {
        "policy_id": policy_id,
        "status": "deleted",
        "deleted_at": datetime.now().isoformat()
    }


def enforce_policy(policy_id: str, context: Dict) -> Dict:
    """
    Enforce a policy against a given context.
    
    Args:
        policy_id: Unique identifier for the policy
        context: Context data to evaluate policy against
    
    Returns:
        Dictionary containing enforcement result
    
    TODO: Implement policy enforcement logic
    """
    # TODO: Implement policy enforcement in policy_management module
    return {
        "policy_id": policy_id,
        "enforced": False,
        "violations": [],
        "timestamp": datetime.now().isoformat()
    }
