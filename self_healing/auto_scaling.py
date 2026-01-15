"""
Logic for automatic scaling and resource management.

This module provides functionality to automatically scale down resources
or shut down services when cost spikes or anomalies are detected.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
import os

# Try to import boto3 for AWS
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    BotoCoreError = Exception

# Try to import Azure SDK
try:
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.storage import StorageManagementClient
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import AzureError
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    ComputeManagementClient = None
    StorageManagementClient = None
    DefaultAzureCredential = Exception
    AzureError = Exception

# Try to import GCP SDK
try:
    from google.cloud import compute_v1
    from google.cloud import storage as gcp_storage
    from google.auth.exceptions import GoogleAuthError
    from google.api_core import exceptions as gcp_exceptions
    GCP_SDK_AVAILABLE = True
except ImportError:
    GCP_SDK_AVAILABLE = False
    compute_v1 = None
    gcp_storage = None
    GoogleAuthError = Exception
    gcp_exceptions = None


class ScalingAction(ABC):
    """Abstract base class for scaling actions."""
    
    @abstractmethod
    def execute(self, context: Dict) -> Dict:
        """Execute the scaling action."""
        pass
    
    @abstractmethod
    def can_execute(self, context: Dict) -> bool:
        """Check if the action can be executed."""
        pass


class ScaleDownAction(ScalingAction):
    """Action to scale down resources."""
    
    def __init__(self, target_resource: str, scale_factor: float = 0.5):
        """
        Initialize scale down action.
        
        Args:
            target_resource: Resource identifier to scale down
            scale_factor: Factor by which to scale (0.5 = reduce by 50%)
        """
        self.target_resource = target_resource
        self.scale_factor = scale_factor
    
    def execute(self, context: Dict) -> Dict:
        """
        Execute scale down action.
        
        Args:
            context: Context data containing resource information
        
        Returns:
            Dictionary containing execution result
        
        TODO: Implement actual scale down logic
        """
        if not self.can_execute(context):
            return {
                "action": "scale_down",
                "status": "failed",
                "reason": "Cannot execute scale down",
                "timestamp": datetime.now().isoformat()
            }
        
        # TODO: Implement scale down logic
        # This would typically involve:
        # - Identifying resources to scale down
        # - Calling cloud provider APIs (AWS, Azure, GCP)
        # - Verifying scale down completion
        
        return {
            "action": "scale_down",
            "target_resource": self.target_resource,
            "scale_factor": self.scale_factor,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
    
    def can_execute(self, context: Dict) -> bool:
        """
        Check if scale down can be executed.
        
        Args:
            context: Context data
        
        Returns:
            True if action can be executed, False otherwise
        
        TODO: Implement validation logic
        """
        # TODO: Validate that:
        # - Resource exists
        # - Current scale is above minimum threshold
        # - No critical processes are running
        return True


class ShutdownAction(ScalingAction):
    """Action to shut down services or resources."""
    
    def __init__(self, target_service: str, graceful: bool = True):
        """
        Initialize shutdown action.
        
        Args:
            target_service: Service identifier to shut down
            graceful: Whether to perform graceful shutdown
        """
        self.target_service = target_service
        self.graceful = graceful
    
    def execute(self, context: Dict) -> Dict:
        """
        Execute shutdown action.
        
        Args:
            context: Context data containing service information
        
        Returns:
            Dictionary containing execution result
        
        TODO: Implement actual shutdown logic
        """
        if not self.can_execute(context):
            return {
                "action": "shutdown",
                "status": "failed",
                "reason": "Cannot execute shutdown",
                "timestamp": datetime.now().isoformat()
            }
        
        # TODO: Implement shutdown logic
        # This would typically involve:
        # - Identifying services to shut down
        # - Performing graceful shutdown if requested
        # - Terminating resources via cloud provider APIs
        # - Verifying shutdown completion
        
        return {
            "action": "shutdown",
            "target_service": self.target_service,
            "graceful": self.graceful,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
    
    def can_execute(self, context: Dict) -> bool:
        """
        Check if shutdown can be executed.
        
        Args:
            context: Context data
        
        Returns:
            True if action can be executed, False otherwise
        
        TODO: Implement validation logic
        """
        # TODO: Validate that:
        # - Service exists
        # - No critical dependencies
        # - Shutdown is safe to perform
        return True


class AutoScaler:
    """Manages automatic scaling actions based on detected anomalies."""
    
    def __init__(self, actions: List[ScalingAction] = None):
        """
        Initialize auto scaler.
        
        Args:
            actions: List of available scaling actions
        """
        self.actions = actions or []
    
    def add_action(self, action: ScalingAction):
        """Add a scaling action."""
        self.actions.append(action)
    
    def remove_action(self, action: ScalingAction):
        """Remove a scaling action."""
        if action in self.actions:
            self.actions.remove(action)
    
    def handle_cost_spike(self, cost_data: Dict, threshold: float) -> List[Dict]:
        """
        Handle cost spike by executing appropriate scaling actions.
        
        Args:
            cost_data: Dictionary containing cost information
            threshold: Cost threshold that was exceeded
        
        Returns:
            List of action execution results
        
        TODO: Implement cost spike handling logic
        """
        current_cost = cost_data.get("current_cost", 0.0)
        if current_cost <= threshold:
            return []
        
        # TODO: Determine which resources to scale down
        # TODO: Select appropriate actions based on cost spike severity
        results = []
        
        for action in self.actions:
            if action.can_execute(cost_data):
                result = action.execute(cost_data)
                results.append(result)
        
        return results
    
    def handle_anomaly(self, anomaly: Dict, policy: Optional[Dict] = None) -> List[Dict]:
        """
        Handle detected anomaly by executing appropriate scaling actions.
        
        Args:
            anomaly: Dictionary containing anomaly information
            policy: Optional policy dictating response actions
        
        Returns:
            List of action execution results
        
        TODO: Implement anomaly handling logic
        """
        # TODO: Determine appropriate response based on anomaly type and severity
        # TODO: Check policy for allowed actions
        results = []
        
        severity = anomaly.get("severity", "medium")
        anomaly_type = anomaly.get("type", "unknown")
        
        # Only take action for high severity anomalies
        if severity in ["high", "critical"]:
            for action in self.actions:
                if action.can_execute(anomaly):
                    result = action.execute(anomaly)
                    results.append(result)
        
        return results


def log_self_healing_action(
    resource_name: str,
    action_taken: str,
    cost_at_action: Optional[float] = None,
    status: str = "success",
    error_message: Optional[str] = None
) -> bool:
    """
    Log a self-healing action to the PostgreSQL database.
    
    This function stores self-healing action details (resource name, action taken,
    cost at action, status, error message) in the self_healing_log table.
    
    Args:
        resource_name: Name of the resource that was acted upon (e.g., 'AWS EC2', 'S3 Storage')
        action_taken: Description of the action taken (e.g., 'Stopped EC2 instance i-1234567890')
        cost_at_action: Optional cost value when action was taken
        status: Status of the action ("success", "failed", "skipped")
        error_message: Optional error message if action failed
    
    Returns:
        True if logging successful, False otherwise
    """
    conn = None
    try:
        from data_collection.database import connect_to_db, create_self_healing_log_schema
        
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to establish database connection for self-healing log")
            except ImportError:
                print("Error: Failed to establish database connection for self-healing log")
            return False
        
        # Ensure schema exists
        if not create_self_healing_log_schema(conn):
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning("Failed to create/verify self_healing_log schema")
            except ImportError:
                print("Warning: Failed to create/verify self_healing_log schema")
            # Continue anyway - schema might already exist
        
        cursor = conn.cursor()
        
        # Validate inputs
        if not resource_name or not isinstance(resource_name, str):
            print("Error: resource_name must be a non-empty string")
            cursor.close()
            conn.close()
            return False
        
        if not action_taken or not isinstance(action_taken, str):
            print("Error: action_taken must be a non-empty string")
            cursor.close()
            conn.close()
            return False
        
        # Insert query
        insert_query = """
            INSERT INTO self_healing_log (resource_name, action_taken, timestamp, cost_at_action, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        timestamp = datetime.utcnow()
        
        cursor.execute(
            insert_query,
            (
                resource_name,
                action_taken,
                timestamp,
                float(cost_at_action) if cost_at_action is not None else None,
                status,
                error_message
            )
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for self-healing log")
        except ImportError:
            print("Error: Database module not available for self-healing log")
        return False
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error logging self-healing action: {e}", exc_info=True)
        except ImportError:
            print(f"Error logging self-healing action: {e}")
        if conn:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
        return False


def shutdown_ec2_instance(instance_id: str) -> bool:
    """
    Stop an AWS EC2 instance.
    
    Args:
        instance_id: EC2 instance ID (e.g., 'i-1234567890abcdef0')
    
    Returns:
        True if instance stopped successfully, False otherwise
    """
    if not BOTO3_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("boto3 library not available for EC2 operations")
        except ImportError:
            print("Error: boto3 library not available for EC2 operations")
        return False
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        if settings is None:
            return False
        
        aws_config = settings.get_cloud_config('aws')
        if not aws_config.get('access_key') or not aws_config.get('secret_key'):
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("AWS credentials not configured")
            except ImportError:
                print("Error: AWS credentials not configured")
            return False
        
        # Create EC2 client
        ec2_client = boto3.client(
            'ec2',
            aws_access_key_id=aws_config['access_key'],
            aws_secret_access_key=aws_config['secret_key'],
            region_name=aws_config.get('region', 'us-east-1')
        )
        
        # Check if instance exists and is running
        try:
            response = ec2_client.describe_instances(InstanceIds=[instance_id])
            if not response.get('Reservations'):
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"EC2 instance {instance_id} not found")
                except ImportError:
                    print(f"Warning: EC2 instance {instance_id} not found")
                return False
            
            instance = response['Reservations'][0]['Instances'][0]
            state = instance['State']['Name']
            
            # Check for production/critical tags
            tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
            protected_tags = settings.self_healing_protected_tags if hasattr(settings, 'self_healing_protected_tags') else ['production', 'critical']
            
            for tag_key, tag_value in tags.items():
                if tag_key.lower() in [t.lower() for t in protected_tags] or tag_value.lower() in ['production', 'critical', 'true']:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"Skipping EC2 instance {instance_id} - protected by tag {tag_key}={tag_value}")
                    except ImportError:
                        print(f"Info: Skipping EC2 instance {instance_id} - protected by tag")
                    return False
            
            # Only stop if running
            if state == 'running':
                # Check if dry-run mode
                if hasattr(settings, 'self_healing_dry_run') and settings.self_healing_dry_run:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"DRY RUN: Would stop EC2 instance {instance_id}")
                    except ImportError:
                        print(f"DRY RUN: Would stop EC2 instance {instance_id}")
                    return True
                
                # Stop the instance
                ec2_client.stop_instances(InstanceIds=[instance_id])
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(f"Stopped EC2 instance {instance_id}")
                except ImportError:
                    print(f"Stopped EC2 instance {instance_id}")
                return True
            else:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(f"EC2 instance {instance_id} is not running (state: {state})")
                except ImportError:
                    print(f"Info: EC2 instance {instance_id} is not running (state: {state})")
                return False
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = str(e)
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error stopping EC2 instance {instance_id}: {error_code} - {error_msg}")
            except ImportError:
                print(f"Error stopping EC2 instance {instance_id}: {error_code} - {error_msg}")
            return False
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Unexpected error stopping EC2 instance {instance_id}: {e}", exc_info=True)
        except ImportError:
            print(f"Error stopping EC2 instance {instance_id}: {e}")
        return False


def delete_s3_bucket(bucket_name: str) -> bool:
    """
    Delete an AWS S3 bucket (only if empty or unused).
    
    Args:
        bucket_name: S3 bucket name
    
    Returns:
        True if bucket deleted successfully, False otherwise
    """
    if not BOTO3_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("boto3 library not available for S3 operations")
        except ImportError:
            print("Error: boto3 library not available for S3 operations")
        return False
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        if settings is None:
            return False
        
        aws_config = settings.get_cloud_config('aws')
        if not aws_config.get('access_key') or not aws_config.get('secret_key'):
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("AWS credentials not configured")
            except ImportError:
                print("Error: AWS credentials not configured")
            return False
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_config['access_key'],
            aws_secret_access_key=aws_config['secret_key'],
            region_name=aws_config.get('region', 'us-east-1')
        )
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"S3 bucket {bucket_name} not found")
                except ImportError:
                    print(f"Warning: S3 bucket {bucket_name} not found")
                return False
            else:
                raise
        
        # Check bucket tags for protection
        try:
            tags_response = s3_client.get_bucket_tagging(Bucket=bucket_name)
            tags = {tag['Key']: tag['Value'] for tag in tags_response.get('TagSet', [])}
            protected_tags = settings.self_healing_protected_tags if hasattr(settings, 'self_healing_protected_tags') else ['production', 'critical']
            
            for tag_key, tag_value in tags.items():
                if tag_key.lower() in [t.lower() for t in protected_tags] or tag_value.lower() in ['production', 'critical', 'true']:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"Skipping S3 bucket {bucket_name} - protected by tag {tag_key}={tag_value}")
                    except ImportError:
                        print(f"Info: Skipping S3 bucket {bucket_name} - protected by tag")
                    return False
        except ClientError:
            # No tags or tag access denied - continue
            pass
        
        # Check if bucket is empty
        try:
            objects = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            if objects.get('KeyCount', 0) > 0:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"S3 bucket {bucket_name} is not empty - skipping deletion")
                except ImportError:
                    print(f"Warning: S3 bucket {bucket_name} is not empty - skipping deletion")
                return False
        except ClientError as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Error checking S3 bucket contents: {e}")
            except ImportError:
                print(f"Warning: Error checking S3 bucket contents: {e}")
            return False
        
        # Check if dry-run mode
        if hasattr(settings, 'self_healing_dry_run') and settings.self_healing_dry_run:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"DRY RUN: Would delete S3 bucket {bucket_name}")
            except ImportError:
                print(f"DRY RUN: Would delete S3 bucket {bucket_name}")
            return True
        
        # Delete the bucket
        s3_client.delete_bucket(Bucket=bucket_name)
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"Deleted S3 bucket {bucket_name}")
        except ImportError:
            print(f"Deleted S3 bucket {bucket_name}")
        return True
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = str(e)
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error deleting S3 bucket {bucket_name}: {error_code} - {error_msg}")
        except ImportError:
            print(f"Error deleting S3 bucket {bucket_name}: {error_code} - {error_msg}")
        return False
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Unexpected error deleting S3 bucket {bucket_name}: {e}", exc_info=True)
        except ImportError:
            print(f"Error deleting S3 bucket {bucket_name}: {e}")
        return False


def stop_azure_vm(vm_name: str, resource_group: str) -> bool:
    """
    Stop an Azure Virtual Machine.
    
    Args:
        vm_name: Azure VM name
        resource_group: Azure resource group name
    
    Returns:
        True if VM stopped successfully, False otherwise
    """
    if not AZURE_SDK_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Azure SDK not available for VM operations")
        except ImportError:
            print("Error: Azure SDK not available for VM operations")
        return False
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        if settings is None:
            return False
        
        azure_config = settings.get_cloud_config('azure')
        if not azure_config.get('subscription_id'):
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Azure subscription ID not configured")
            except ImportError:
                print("Error: Azure subscription ID not configured")
            return False
        
        # Authenticate and create compute client
        credential = DefaultAzureCredential()
        compute_client = ComputeManagementClient(
            credential,
            azure_config['subscription_id']
        )
        
        # Check if VM exists and get its state
        try:
            vm = compute_client.virtual_machines.get(resource_group, vm_name)
            
            # Check tags for protection
            tags = vm.tags or {}
            protected_tags = settings.self_healing_protected_tags if hasattr(settings, 'self_healing_protected_tags') else ['production', 'critical']
            
            for tag_key, tag_value in tags.items():
                if tag_key.lower() in [t.lower() for t in protected_tags] or str(tag_value).lower() in ['production', 'critical', 'true']:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"Skipping Azure VM {vm_name} - protected by tag {tag_key}={tag_value}")
                    except ImportError:
                        print(f"Info: Skipping Azure VM {vm_name} - protected by tag")
                    return False
            
            # Check instance view to get power state
            instance_view = compute_client.virtual_machines.instance_view(resource_group, vm_name)
            power_state = None
            for status in instance_view.statuses:
                if status.code.startswith('PowerState/'):
                    power_state = status.code.split('/')[1]
                    break
            
            # Only stop if running
            if power_state == 'running':
                # Check if dry-run mode
                if hasattr(settings, 'self_healing_dry_run') and settings.self_healing_dry_run:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"DRY RUN: Would stop Azure VM {vm_name}")
                    except ImportError:
                        print(f"DRY RUN: Would stop Azure VM {vm_name}")
                    return True
                
                # Stop the VM
                compute_client.virtual_machines.begin_power_off(resource_group, vm_name).wait()
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(f"Stopped Azure VM {vm_name}")
                except ImportError:
                    print(f"Stopped Azure VM {vm_name}")
                return True
            else:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(f"Azure VM {vm_name} is not running (state: {power_state})")
                except ImportError:
                    print(f"Info: Azure VM {vm_name} is not running (state: {power_state})")
                return False
        
        except AzureError as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error stopping Azure VM {vm_name}: {e}")
            except ImportError:
                print(f"Error stopping Azure VM {vm_name}: {e}")
            return False
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Unexpected error stopping Azure VM {vm_name}: {e}", exc_info=True)
        except ImportError:
            print(f"Error stopping Azure VM {vm_name}: {e}")
        return False


def stop_gcp_instance(instance_name: str, zone: str, project_id: str) -> bool:
    """
    Stop a GCP Compute Engine instance.
    
    Args:
        instance_name: GCP instance name
        zone: GCP zone (e.g., 'us-central1-a')
        project_id: GCP project ID
    
    Returns:
        True if instance stopped successfully, False otherwise
    """
    if not GCP_SDK_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("GCP SDK not available for Compute Engine operations")
        except ImportError:
            print("Error: GCP SDK not available for Compute Engine operations")
        return False
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        if settings is None:
            return False
        
        if not project_id:
            gcp_config = settings.get_cloud_config('gcp')
            project_id = gcp_config.get('project_id', '')
            if not project_id:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error("GCP project ID not configured")
                except ImportError:
                    print("Error: GCP project ID not configured")
                return False
        
        # Create instances client
        instances_client = compute_v1.InstancesClient()
        
        # Get instance to check state and labels
        try:
            instance = instances_client.get(
                project=project_id,
                zone=zone,
                instance=instance_name
            )
            
            # Check labels for protection
            labels = instance.labels or {}
            protected_tags = settings.self_healing_protected_tags if hasattr(settings, 'self_healing_protected_tags') else ['production', 'critical']
            
            for label_key, label_value in labels.items():
                if label_key.lower() in [t.lower() for t in protected_tags] or label_value.lower() in ['production', 'critical', 'true']:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"Skipping GCP instance {instance_name} - protected by label {label_key}={label_value}")
                    except ImportError:
                        print(f"Info: Skipping GCP instance {instance_name} - protected by label")
                    return False
            
            # Only stop if running
            if instance.status == 'RUNNING':
                # Check if dry-run mode
                if hasattr(settings, 'self_healing_dry_run') and settings.self_healing_dry_run:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"DRY RUN: Would stop GCP instance {instance_name}")
                    except ImportError:
                        print(f"DRY RUN: Would stop GCP instance {instance_name}")
                    return True
                
                # Stop the instance
                operation = instances_client.stop(
                    project=project_id,
                    zone=zone,
                    instance=instance_name
                )
                # Wait for operation to complete
                operation.result()
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(f"Stopped GCP instance {instance_name}")
                except ImportError:
                    print(f"Stopped GCP instance {instance_name}")
                return True
            else:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(f"GCP instance {instance_name} is not running (status: {instance.status})")
                except ImportError:
                    print(f"Info: GCP instance {instance_name} is not running (status: {instance.status})")
                return False
        
        except gcp_exceptions.NotFound:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"GCP instance {instance_name} not found")
            except ImportError:
                print(f"Warning: GCP instance {instance_name} not found")
            return False
        except GoogleAuthError as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"GCP authentication error: {e}")
            except ImportError:
                print(f"Error: GCP authentication error: {e}")
            return False
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Unexpected error stopping GCP instance {instance_name}: {e}", exc_info=True)
        except ImportError:
            print(f"Error stopping GCP instance {instance_name}: {e}")
        return False


def scale_down_resources(
    resource_name: str,
    current_cost: float,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Scale down cloud resources or shut down services when a cost spike is detected.
    
    This function identifies the cloud provider and resource type from resource_name,
    then executes the appropriate scaling action based on configuration.
    
    Args:
        resource_name: Name of the resource (e.g., 'AWS EC2', 'S3 Storage', 'Azure VM')
        current_cost: Current cost of the resource
        threshold: Optional cost threshold (if None, uses settings.cloud_cost_threshold)
    
    Returns:
        Dictionary containing:
        {
            'resource_name': str,
            'action_taken': str,
            'status': str,  # 'success', 'failed', 'skipped'
            'cost_at_action': float,
            'error_message': Optional[str]
        }
    """
    result = {
        'resource_name': resource_name,
        'action_taken': '',
        'status': 'skipped',
        'cost_at_action': current_cost,
        'error_message': None
    }
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        if settings is None:
            result['error_message'] = "Settings not available"
            return result
        
        # Check if self-healing is enabled
        if not settings.self_healing_enabled:
            result['action_taken'] = "Self-healing disabled"
            result['status'] = 'skipped'
            log_self_healing_action(
                resource_name=resource_name,
                action_taken="Self-healing disabled",
                cost_at_action=current_cost,
                status='skipped'
            )
            return result
        
        # Determine threshold
        if threshold is None:
            threshold = settings.cloud_cost_threshold
        
        # Check if cost exceeds threshold
        if current_cost <= threshold:
            result['action_taken'] = f"Cost {current_cost} below threshold {threshold}"
            result['status'] = 'skipped'
            log_self_healing_action(
                resource_name=resource_name,
                action_taken=f"Cost below threshold",
                cost_at_action=current_cost,
                status='skipped'
            )
            return result
        
        # Extract provider and resource type from resource_name
        resource_lower = resource_name.lower()
        provider = None
        resource_type = None
        
        if 'aws' in resource_lower or 'ec2' in resource_lower or 's3' in resource_lower:
            provider = 'AWS'
            if 'ec2' in resource_lower:
                resource_type = 'EC2'
            elif 's3' in resource_lower or 'storage' in resource_lower:
                resource_type = 'S3'
        elif 'azure' in resource_lower:
            provider = 'Azure'
            if 'vm' in resource_lower or 'virtual' in resource_lower:
                resource_type = 'VM'
            elif 'storage' in resource_lower:
                resource_type = 'Storage'
        elif 'gcp' in resource_lower or 'google' in resource_lower or 'compute engine' in resource_lower:
            provider = 'GCP'
            if 'compute' in resource_lower or 'instance' in resource_lower:
                resource_type = 'Compute Engine'
            elif 'storage' in resource_lower:
                resource_type = 'Storage'
        
        if not provider or not resource_type:
            result['error_message'] = f"Unknown resource type: {resource_name}"
            result['status'] = 'failed'
            log_self_healing_action(
                resource_name=resource_name,
                action_taken="Unknown resource type",
                cost_at_action=current_cost,
                status='failed',
                error_message=result['error_message']
            )
            return result
        
        # Get scaling action configuration
        scaling_actions = getattr(settings, 'scaling_actions', {})
        if not scaling_actions:
            # Default actions
            scaling_actions = {
                'AWS EC2': 'shutdown',
                'AWS S3': 'delete',
                'Azure VM': 'stop',
                'GCP Compute Engine': 'stop'
            }
        
        action_key = f"{provider} {resource_type}"
        action_type = scaling_actions.get(action_key, 'shutdown')
        
        # Execute appropriate action
        action_success = False
        action_description = ""
        error_msg = None
        
        if provider == 'AWS':
            if resource_type == 'EC2':
                # For EC2, we need to identify which instance to stop
                # For now, we'll try to get instance IDs from settings or query
                instance_ids = getattr(settings, 'aws_instance_ids', '').strip()
                if instance_ids:
                    instance_list = [i.strip() for i in instance_ids.split(",") if i.strip()]
                    # Stop the first non-protected instance
                    for instance_id in instance_list:
                        if shutdown_ec2_instance(instance_id):
                            action_success = True
                            action_description = f"Stopped EC2 instance {instance_id}"
                            break
                    if not action_success:
                        error_msg = "Failed to stop any EC2 instances"
                else:
                    error_msg = "No EC2 instance IDs configured"
            elif resource_type == 'S3':
                # For S3, we need bucket names
                bucket_names = getattr(settings, 'aws_s3_bucket_names', '').strip()
                if bucket_names:
                    bucket_list = [b.strip() for b in bucket_names.split(",") if b.strip()]
                    # Delete the first empty bucket
                    for bucket_name in bucket_list:
                        if delete_s3_bucket(bucket_name):
                            action_success = True
                            action_description = f"Deleted S3 bucket {bucket_name}"
                            break
                    if not action_success:
                        error_msg = "Failed to delete any S3 buckets"
                else:
                    error_msg = "No S3 bucket names configured"
        
        elif provider == 'Azure':
            if resource_type == 'VM':
                # For Azure VM, we need VM name and resource group
                vm_names = getattr(settings, 'azure_vm_names', '').strip()
                if vm_names:
                    vm_list = [v.strip() for v in vm_names.split(",") if v.strip()]
                    # Get resource group from settings or use default
                    resource_group = getattr(settings, 'azure_resource_group', '')
                    if not resource_group:
                        resource_group = 'default'  # Fallback
                    
                    for vm_name in vm_list:
                        if stop_azure_vm(vm_name, resource_group):
                            action_success = True
                            action_description = f"Stopped Azure VM {vm_name}"
                            break
                    if not action_success:
                        error_msg = "Failed to stop any Azure VMs"
                else:
                    error_msg = "No Azure VM names configured"
        
        elif provider == 'GCP':
            if resource_type == 'Compute Engine':
                # For GCP, we need instance name, zone, and project
                instance_names = getattr(settings, 'gcp_instance_names', '').strip()
                if instance_names:
                    instance_list = [i.strip() for i in instance_names.split(",") if i.strip()]
                    project_id = settings.gcp_project_id
                    # Get zone from settings or use default
                    zone = getattr(settings, 'gcp_default_zone', 'us-central1-a')
                    
                    for instance_name in instance_list:
                        if stop_gcp_instance(instance_name, zone, project_id):
                            action_success = True
                            action_description = f"Stopped GCP instance {instance_name}"
                            break
                    if not action_success:
                        error_msg = "Failed to stop any GCP instances"
                else:
                    error_msg = "No GCP instance names configured"
        
        # Update result
        if action_success:
            result['action_taken'] = action_description
            result['status'] = 'success'
        else:
            result['action_taken'] = f"Attempted to scale down {resource_name}"
            result['status'] = 'failed'
            result['error_message'] = error_msg or "Scaling action failed"
        
        # Log the action
        log_self_healing_action(
            resource_name=resource_name,
            action_taken=result['action_taken'],
            cost_at_action=current_cost,
            status=result['status'],
            error_message=result['error_message']
        )
        
        return result
    
    except Exception as e:
        error_msg = str(e)
        result['status'] = 'failed'
        result['error_message'] = error_msg
        result['action_taken'] = f"Error scaling down {resource_name}"
        
        # Log the error
        log_self_healing_action(
            resource_name=resource_name,
            action_taken=result['action_taken'],
            cost_at_action=current_cost,
            status='failed',
            error_message=error_msg
        )
        
        return result


def create_auto_scaler(actions: List[ScalingAction] = None) -> AutoScaler:
    """
    Create an auto scaler instance.
    
    Args:
        actions: Optional list of scaling actions
    
    Returns:
        AutoScaler instance
    
    TODO: Implement auto scaler factory
    """
    return AutoScaler(actions=actions)
