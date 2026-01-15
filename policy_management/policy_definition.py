"""
Logic for defining and storing policies.

This module provides functionality to define policies with rules,
conditions, and actions for the monitoring system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod

# Import database functions
try:
    from data_collection.database import connect_to_db
    from psycopg2 import Error
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    connect_to_db = None
    Error = Exception


class PolicyRule:
    """Represents a single rule within a policy."""
    
    def __init__(
        self,
        condition: Dict,
        action: Dict,
        priority: int = 0
    ):
        """
        Initialize a policy rule.
        
        Args:
            condition: Dictionary defining the condition to evaluate
            action: Dictionary defining the action to take if condition is met
            priority: Rule priority (higher values have higher priority)
        """
        self.condition = condition
        self.action = action
        self.priority = priority
    
    def to_dict(self) -> Dict:
        """Convert rule to dictionary representation."""
        return {
            "condition": self.condition,
            "action": self.action,
            "priority": self.priority
        }


class Policy:
    """Represents a policy with multiple rules."""
    
    def __init__(
        self,
        policy_id: Optional[str],
        name: str,
        description: str,
        rules: List[PolicyRule],
        enabled: bool = True,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        """
        Initialize a policy.
        
        Args:
            policy_id: Unique identifier for the policy
            name: Policy name
            description: Policy description
            rules: List of policy rules
            enabled: Whether the policy is enabled
            created_at: Policy creation timestamp
            updated_at: Policy last update timestamp
        """
        self.policy_id = policy_id
        self.name = name
        self.description = description
        self.rules = rules
        self.enabled = enabled
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert policy to dictionary representation."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules],
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def add_rule(self, rule: PolicyRule):
        """Add a rule to the policy."""
        self.rules.append(rule)
        self.updated_at = datetime.now()
    
    def remove_rule(self, index: int):
        """Remove a rule from the policy by index."""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
            self.updated_at = datetime.now()
    
    def enable(self):
        """Enable the policy."""
        self.enabled = True
        self.updated_at = datetime.now()
    
    def disable(self):
        """Disable the policy."""
        self.enabled = False
        self.updated_at = datetime.now()


class ResourcePolicy:
    """Represents a simple threshold-based policy for resource usage and cost limits."""
    
    def __init__(
        self,
        resource_name: str,
        threshold_value: float,
        threshold_type: str,
        duration: int,
        policy_id: Optional[str] = None,
        enabled: bool = True,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        """
        Initialize a resource policy.
        
        Args:
            resource_name: Name of the resource (e.g., "CPU Usage", "Cloud Cost")
            threshold_value: Threshold value (e.g., 80.0 for CPU usage, 500.0 for cost)
            threshold_type: Type of threshold ("usage" or "cost")
            duration: Duration in minutes for sustained exceedance before violation
            policy_id: Optional unique identifier (set when stored in database)
            enabled: Whether the policy is enabled
            created_at: Policy creation timestamp
            updated_at: Policy last update timestamp
        """
        self.resource_name = resource_name
        self.threshold_value = threshold_value
        self.threshold_type = threshold_type
        self.duration = duration
        self.policy_id = policy_id
        self.enabled = enabled
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        
        # Validate on initialization
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate policy parameters.
        
        Returns:
            True if valid, raises ValueError if invalid
        
        Raises:
            ValueError: If policy parameters are invalid
        """
        if not self.resource_name or not isinstance(self.resource_name, str):
            raise ValueError("resource_name must be a non-empty string")
        
        if not isinstance(self.threshold_value, (int, float)) or self.threshold_value <= 0:
            raise ValueError("threshold_value must be a positive number")
        
        if self.threshold_type not in ["usage", "cost"]:
            raise ValueError("threshold_type must be 'usage' or 'cost'")
        
        if not isinstance(self.duration, int) or self.duration < 0:
            raise ValueError("duration must be a non-negative integer")
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert policy to dictionary representation."""
        return {
            "policy_id": self.policy_id,
            "resource_name": self.resource_name,
            "threshold_value": self.threshold_value,
            "threshold_type": self.threshold_type,
            "duration": self.duration,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }
    
    def enable(self):
        """Enable the policy."""
        self.enabled = True
        self.updated_at = datetime.utcnow()
    
    def disable(self):
        """Disable the policy."""
        self.enabled = False
        self.updated_at = datetime.utcnow()


class PolicyDefinitionManager:
    """Manages policy definitions and storage."""
    
    def __init__(self, database=None):
        """
        Initialize policy definition manager.
        
        Args:
            database: Optional database connection for persistence
        """
        self.database = database
        self.policies: Dict[str, Policy] = {}
    
    def create_policy(
        self,
        name: str,
        description: str,
        rules: List[Dict],
        enabled: bool = True
    ) -> Policy:
        """
        Create a new policy.
        
        Args:
            name: Policy name
            description: Policy description
            rules: List of rule dictionaries
            enabled: Whether the policy is enabled
        
        Returns:
            Created Policy instance
        
        TODO: Implement policy creation with database persistence
        """
        # TODO: Generate unique policy ID
        policy_id = None  # TODO: Generate UUID or use database sequence
        
        # TODO: Convert rule dictionaries to PolicyRule objects
        policy_rules = []
        for rule_dict in rules:
            rule = PolicyRule(
                condition=rule_dict.get("condition", {}),
                action=rule_dict.get("action", {}),
                priority=rule_dict.get("priority", 0)
            )
            policy_rules.append(rule)
        
        policy = Policy(
            policy_id=policy_id,
            name=name,
            description=description,
            rules=policy_rules,
            enabled=enabled
        )
        
        # TODO: Store policy in database if database connection available
        if self.database:
            # TODO: self.database.insert_policy(policy.to_dict())
            pass
        
        if policy_id:
            self.policies[policy_id] = policy
        
        return policy
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """
        Retrieve a policy by ID.
        
        Args:
            policy_id: Unique identifier for the policy
        
        Returns:
            Policy instance or None if not found
        
        TODO: Implement policy retrieval from database
        """
        # TODO: Retrieve from database if not in memory
        if policy_id in self.policies:
            return self.policies[policy_id]
        
        if self.database:
            # TODO: policy_data = self.database.get_policy(policy_id)
            # TODO: Convert to Policy object
            pass
        
        return None
    
    def list_policies(self, enabled_only: bool = False) -> List[Policy]:
        """
        List all policies.
        
        Args:
            enabled_only: If True, only return enabled policies
        
        Returns:
            List of Policy instances
        
        TODO: Implement policy listing from database
        """
        # TODO: Retrieve from database if needed
        policies = list(self.policies.values())
        
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        
        return policies
    
    def update_policy(
        self,
        policy_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        rules: Optional[List[Dict]] = None,
        enabled: Optional[bool] = None
    ) -> Optional[Policy]:
        """
        Update an existing policy.
        
        Args:
            policy_id: Unique identifier for the policy
            name: Optional new policy name
            description: Optional new policy description
            rules: Optional new policy rules
            enabled: Optional new enabled status
        
        Returns:
            Updated Policy instance or None if not found
        
        TODO: Implement policy update with database persistence
        """
        policy = self.get_policy(policy_id)
        if not policy:
            return None
        
        if name is not None:
            policy.name = name
        if description is not None:
            policy.description = description
        if rules is not None:
            # TODO: Convert rule dictionaries to PolicyRule objects
            policy.rules = []
            for rule_dict in rules:
                rule = PolicyRule(
                    condition=rule_dict.get("condition", {}),
                    action=rule_dict.get("action", {}),
                    priority=rule_dict.get("priority", 0)
                )
                policy.rules.append(rule)
        if enabled is not None:
            policy.enabled = enabled
        
        policy.updated_at = datetime.now()
        
        # TODO: Update in database
        if self.database:
            # TODO: self.database.update_policy(policy_id, policy.to_dict())
            pass
        
        return policy
    
    def delete_policy(self, policy_id: str) -> bool:
        """
        Delete a policy.
        
        Args:
            policy_id: Unique identifier for the policy
        
        Returns:
            True if deletion successful, False otherwise
        
        TODO: Implement policy deletion
        """
        if policy_id in self.policies:
            del self.policies[policy_id]
        
        if self.database:
            # TODO: self.database.delete_policy(policy_id)
            pass
        
        return True


def create_resource_policy(
    resource_name: str,
    threshold_value: float,
    threshold_type: str,
    duration: int,
    store_in_db: bool = True
) -> ResourcePolicy:
    """
    Create a resource policy instance.
    
    This function creates a ResourcePolicy for simple threshold-based monitoring
    of resource usage and cost limits.
    
    Args:
        resource_name: Name of the resource (e.g., "CPU Usage", "Cloud Cost")
        threshold_value: Threshold value (e.g., 80.0 for CPU usage, 500.0 for cost)
        threshold_type: Type of threshold ("usage" or "cost")
        duration: Duration in minutes for sustained exceedance before violation
        store_in_db: If True, automatically store policy in database
    
    Returns:
        ResourcePolicy instance
    
    Raises:
        ValueError: If parameters are invalid
    """
    # Create policy instance (validation happens in __init__)
    policy = ResourcePolicy(
        resource_name=resource_name,
        threshold_value=threshold_value,
        threshold_type=threshold_type,
        duration=duration
    )
    
    # Store in database if requested
    if store_in_db:
        try:
            policy_id = store_resource_policy_in_db(policy)
            if policy_id:
                policy.policy_id = str(policy_id)
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Failed to store policy in database: {e}. Policy created but not stored.")
            except ImportError:
                print(f"Warning: Failed to store policy in database: {e}. Policy created but not stored.")
    
    return policy


def create_cpu_usage_policy(
    threshold: float,
    duration: int = 5
) -> ResourcePolicy:
    """
    Create a CPU usage policy.
    
    Convenience function for creating CPU usage policies.
    
    Args:
        threshold: CPU usage threshold percentage (e.g., 80.0 for 80%)
        duration: Duration in minutes for sustained exceedance (default: 5)
    
    Returns:
        ResourcePolicy instance for CPU usage
    """
    return create_resource_policy(
        resource_name="CPU Usage",
        threshold_value=threshold,
        threshold_type="usage",
        duration=duration
    )


def create_cloud_cost_policy(
    threshold: float,
    duration: int = 0
) -> ResourcePolicy:
    """
    Create a cloud cost policy.
    
    Convenience function for creating cloud cost limit policies.
    
    Args:
        threshold: Cost threshold in dollars (e.g., 500.0 for $500/day)
        duration: Duration in minutes for sustained exceedance (default: 0 for immediate)
    
    Returns:
        ResourcePolicy instance for cloud cost
    """
    return create_resource_policy(
        resource_name="Cloud Cost",
        threshold_value=threshold,
        threshold_type="cost",
        duration=duration
    )


def create_memory_usage_policy(
    threshold: float,
    duration: int = 5
) -> ResourcePolicy:
    """
    Create a memory usage policy.
    
    Convenience function for creating memory usage policies.
    
    Args:
        threshold: Memory usage threshold percentage (e.g., 80.0 for 80%)
        duration: Duration in minutes for sustained exceedance (default: 5)
    
    Returns:
        ResourcePolicy instance for memory usage
    """
    return create_resource_policy(
        resource_name="Memory Usage",
        threshold_value=threshold,
        threshold_type="usage",
        duration=duration
    )


def store_resource_policy_in_db(policy: ResourcePolicy) -> Optional[int]:
    """
    Store a resource policy in the PostgreSQL database.
    
    Args:
        policy: ResourcePolicy instance to store
    
    Returns:
        Policy ID (from database) if successful, None otherwise
    """
    if not PSYCOPG2_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("psycopg2 not available. Cannot store policy in database.")
        except ImportError:
            print("Error: psycopg2 not available. Cannot store policy in database.")
        return None
    
    # Validate policy before storing
    try:
        policy.validate()
    except ValueError as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Policy validation failed: {e}")
        except ImportError:
            print(f"Error: Policy validation failed: {e}")
        return None
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to connect to database for storing policy")
            except ImportError:
                print("Error: Failed to connect to database for storing policy")
            return None
        
        cursor = conn.cursor()
        
        # Insert policy into database
        insert_query = """
            INSERT INTO resource_policies (
                resource_name, threshold_value, threshold_type, duration, enabled, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        
        cursor.execute(
            insert_query,
            (
                policy.resource_name,
                float(policy.threshold_value),
                policy.threshold_type,
                int(policy.duration),
                policy.enabled,
                datetime.utcnow()
            )
        )
        
        # Get the inserted ID
        policy_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"Successfully stored resource policy: {policy.resource_name} (ID: {policy_id})")
        except ImportError:
            print(f"Successfully stored resource policy: {policy.resource_name} (ID: {policy_id})")
        
        return policy_id
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error storing resource policy: {e}")
        except ImportError:
            print(f"Error storing resource policy in database: {e}")
        if conn:
            conn.rollback()
        return None
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error storing resource policy: {e}", exc_info=True)
        except ImportError:
            print(f"Error storing resource policy: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()


def get_resource_policy_from_db(
    policy_id: Optional[int] = None,
    resource_name: Optional[str] = None
) -> Optional[ResourcePolicy]:
    """
    Retrieve a resource policy from the database.
    
    Args:
        policy_id: Optional policy ID to retrieve by
        resource_name: Optional resource name to retrieve by
    
    Returns:
        ResourcePolicy instance if found, None otherwise
    
    Note:
        Either policy_id or resource_name must be provided. If both are provided,
        policy_id takes precedence.
    """
    if not PSYCOPG2_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("psycopg2 not available. Cannot retrieve policy from database.")
        except ImportError:
            print("Error: psycopg2 not available. Cannot retrieve policy from database.")
        return None
    
    if policy_id is None and resource_name is None:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Either policy_id or resource_name must be provided")
        except ImportError:
            print("Error: Either policy_id or resource_name must be provided")
        return None
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            return None
        
        cursor = conn.cursor()
        
        if policy_id is not None:
            query = """
                SELECT id, resource_name, threshold_value, threshold_type,
                       duration, enabled, timestamp
                FROM resource_policies
                WHERE id = %s;
            """
            cursor.execute(query, (policy_id,))
        else:
            query = """
                SELECT id, resource_name, threshold_value, threshold_type,
                       duration, enabled, timestamp
                FROM resource_policies
                WHERE resource_name = %s
                ORDER BY timestamp DESC
                LIMIT 1;
            """
            cursor.execute(query, (resource_name,))
        
        result = cursor.fetchone()
        cursor.close()
        
        if result is None:
            return None
        
        # Convert database result to ResourcePolicy instance
        policy = ResourcePolicy(
            resource_name=result[1],
            threshold_value=float(result[2]),
            threshold_type=result[3],
            duration=int(result[4]),
            policy_id=str(result[0]),
            enabled=bool(result[5]),
            created_at=result[6],
            updated_at=result[6]
        )
        
        return policy
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error retrieving resource policy: {e}")
        except ImportError:
            print(f"Error retrieving resource policy from database: {e}")
        return None
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error retrieving resource policy: {e}", exc_info=True)
        except ImportError:
            print(f"Error retrieving resource policy: {e}")
        return None
    finally:
        if conn:
            conn.close()


def list_resource_policies_from_db(
    enabled_only: bool = False,
    threshold_type: Optional[str] = None
) -> List[ResourcePolicy]:
    """
    List all resource policies from the database.
    
    Args:
        enabled_only: If True, only return enabled policies
        threshold_type: Optional filter by threshold type ("usage" or "cost")
    
    Returns:
        List of ResourcePolicy instances
    """
    if not PSYCOPG2_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("psycopg2 not available. Cannot list policies from database.")
        except ImportError:
            print("Error: psycopg2 not available. Cannot list policies from database.")
        return []
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            return []
        
        cursor = conn.cursor()
        
        # Build query with optional filters
        query = """
            SELECT id, resource_name, threshold_value, threshold_type,
                   duration, enabled, timestamp
            FROM resource_policies
            WHERE 1=1
        """
        params = []
        
        if enabled_only:
            query += " AND enabled = %s"
            params.append(True)
        
        if threshold_type:
            query += " AND threshold_type = %s"
            params.append(threshold_type)
        
        query += " ORDER BY timestamp DESC;"
        
        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
        cursor.close()
        
        # Convert results to ResourcePolicy instances
        policies = []
        for row in results:
            policy = ResourcePolicy(
                resource_name=row[1],
                threshold_value=float(row[2]),
                threshold_type=row[3],
                duration=int(row[4]),
                policy_id=str(row[0]),
                enabled=bool(row[5]),
                created_at=row[6],
                updated_at=row[6]
            )
            policies.append(policy)
        
        return policies
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error listing resource policies: {e}")
        except ImportError:
            print(f"Error listing resource policies from database: {e}")
        return []
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error listing resource policies: {e}", exc_info=True)
        except ImportError:
            print(f"Error listing resource policies: {e}")
        return []
    finally:
        if conn:
            conn.close()


def update_resource_policy_in_db(
    policy_id: int,
    threshold_value: Optional[float] = None,
    duration: Optional[int] = None,
    enabled: Optional[bool] = None
) -> Optional[ResourcePolicy]:
    """
    Update a resource policy in the database.
    
    Args:
        policy_id: ID of the policy to update
        threshold_value: Optional new threshold value
        duration: Optional new duration
        enabled: Optional new enabled status
    
    Returns:
        Updated ResourcePolicy instance if successful, None otherwise
    """
    if not PSYCOPG2_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("psycopg2 not available. Cannot update policy in database.")
        except ImportError:
            print("Error: psycopg2 not available. Cannot update policy in database.")
        return None
    
    # Check if at least one field is being updated
    if threshold_value is None and duration is None and enabled is None:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning("No fields provided for update")
        except ImportError:
            print("Warning: No fields provided for update")
        return None
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            return None
        
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        if threshold_value is not None:
            update_fields.append("threshold_value = %s")
            params.append(float(threshold_value))
        
        if duration is not None:
            update_fields.append("duration = %s")
            params.append(int(duration))
        
        if enabled is not None:
            update_fields.append("enabled = %s")
            params.append(bool(enabled))
        
        # Add policy_id to params
        params.append(policy_id)
        
        query = f"""
            UPDATE resource_policies
            SET {', '.join(update_fields)}
            WHERE id = %s
            RETURNING id, resource_name, threshold_value, threshold_type,
                      duration, enabled, timestamp;
        """
        
        cursor.execute(query, tuple(params))
        result = cursor.fetchone()
        
        if result is None:
            cursor.close()
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Policy with ID {policy_id} not found")
            except ImportError:
                print(f"Warning: Policy with ID {policy_id} not found")
            return None
        
        conn.commit()
        cursor.close()
        
        # Create updated ResourcePolicy instance
        policy = ResourcePolicy(
            resource_name=result[1],
            threshold_value=float(result[2]),
            threshold_type=result[3],
            duration=int(result[4]),
            policy_id=str(result[0]),
            enabled=bool(result[5]),
            created_at=result[6],
            updated_at=datetime.utcnow()
        )
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"Successfully updated resource policy: {policy.resource_name} (ID: {policy_id})")
        except ImportError:
            print(f"Successfully updated resource policy: {policy.resource_name} (ID: {policy_id})")
        
        return policy
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error updating resource policy: {e}")
        except ImportError:
            print(f"Error updating resource policy in database: {e}")
        if conn:
            conn.rollback()
        return None
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error updating resource policy: {e}", exc_info=True)
        except ImportError:
            print(f"Error updating resource policy: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()


def delete_resource_policy_from_db(policy_id: int) -> bool:
    """
    Delete a resource policy from the database.
    
    Args:
        policy_id: ID of the policy to delete
    
    Returns:
        True if deletion successful, False otherwise
    """
    if not PSYCOPG2_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("psycopg2 not available. Cannot delete policy from database.")
        except ImportError:
            print("Error: psycopg2 not available. Cannot delete policy from database.")
        return False
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        
        delete_query = "DELETE FROM resource_policies WHERE id = %s;"
        cursor.execute(delete_query, (policy_id,))
        
        # Check if any row was deleted
        rows_deleted = cursor.rowcount
        
        conn.commit()
        cursor.close()
        
        if rows_deleted > 0:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"Successfully deleted resource policy (ID: {policy_id})")
            except ImportError:
                print(f"Successfully deleted resource policy (ID: {policy_id})")
            return True
        else:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Policy with ID {policy_id} not found for deletion")
            except ImportError:
                print(f"Warning: Policy with ID {policy_id} not found for deletion")
            return False
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error deleting resource policy: {e}")
        except ImportError:
            print(f"Error deleting resource policy from database: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error deleting resource policy: {e}", exc_info=True)
        except ImportError:
            print(f"Error deleting resource policy: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def create_policy_manager(database=None) -> PolicyDefinitionManager:
    """
    Create a policy definition manager instance.
    
    Args:
        database: Optional database connection
    
    Returns:
        PolicyDefinitionManager instance
    
    TODO: Implement manager factory
    """
    return PolicyDefinitionManager(database=database)
