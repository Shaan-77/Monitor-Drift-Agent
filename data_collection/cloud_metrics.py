"""
Logic for collecting cloud usage metrics.

This module provides functionality to collect cloud usage metrics from
various cloud providers (AWS, Azure, GCP) including cost, resource usage,
and billing information.
"""

import os
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Try to import boto3, handle gracefully if not available
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    BotoCoreError = Exception

# Try to import Google Cloud Monitoring, handle gracefully if not available
try:
    from google.cloud import monitoring_v3
    from google.cloud import compute_v1
    from google.cloud import storage as gcp_storage
    from google.auth.exceptions import GoogleAuthError
    from google.api_core import exceptions as gcp_exceptions
    GCP_MONITORING_AVAILABLE = True
except ImportError:
    GCP_MONITORING_AVAILABLE = False
    monitoring_v3 = None
    compute_v1 = None
    gcp_storage = None
    GoogleAuthError = Exception
    gcp_exceptions = None

# Try to import Azure Monitor, handle gracefully if not available
try:
    from azure.mgmt.monitor import MonitorManagementClient
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.storage import StorageManagementClient
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.core.exceptions import AzureError
    AZURE_MONITORING_AVAILABLE = True
except ImportError:
    AZURE_MONITORING_AVAILABLE = False
    MonitorManagementClient = None
    ComputeManagementClient = None
    StorageManagementClient = None
    DefaultAzureCredential = None
    ClientSecretCredential = None
    AzureError = Exception

# Import settings and logger
try:
    from config.settings import get_settings
    from utils.logger import get_logger
except ImportError:
    def get_settings():
        return None
    def get_logger(name):
        import logging
        return logging.getLogger(name)


def standardize_metrics(
    provider: str,
    metric_name: str,
    metric_value: float,
    timestamp: datetime,
    resource_type: str
) -> Dict:
    """
    Standardize cloud metrics into a consistent format.
    
    Args:
        provider: Cloud provider name ('aws', 'azure', 'gcp')
        metric_name: Name of the metric (e.g., 'CPUUtilization', 'DiskWrite')
        metric_value: Metric value (numeric)
        timestamp: Timestamp when metric was collected
        resource_type: Type of resource ('Compute', 'Storage', 'Network')
    
    Returns:
        Standardized dictionary with provider, metric_name, metric_value, timestamp, resource_type
    """
    return {
        'provider': provider,
        'metric_name': metric_name,
        'metric_value': float(metric_value) if metric_value is not None else 0.0,
        'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
        'resource_type': resource_type
    }


class CloudProvider(ABC):
    """Abstract base class for cloud provider integrations."""
    
    @abstractmethod
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with cloud provider."""
        pass
    
    @abstractmethod
    def collect_cost_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Collect cost metrics for a date range."""
        pass
    
    @abstractmethod
    def collect_resource_metrics(self) -> Dict:
        """Collect current resource usage metrics."""
        pass


class AWSMetricsCollector(CloudProvider):
    """Collector for AWS cloud metrics."""
    
    def __init__(self):
        """Initialize AWS metrics collector."""
        self.cloudwatch_client = None
        self.ec2_client = None
        self.s3_client = None
        self.logger = get_logger(__name__)
    
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with AWS using provided credentials.
        
        Args:
            credentials: Dictionary containing AWS credentials (access_key, secret_key, region)
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not BOTO3_AVAILABLE:
            self.logger.error("boto3 library is not available")
            return False
        
        try:
            access_key = credentials.get('access_key')
            secret_key = credentials.get('secret_key')
            region = credentials.get('region', 'us-east-1')
            
            if not access_key or not secret_key:
                self.logger.error("AWS credentials (access_key, secret_key) are required")
                return False
            
            # Create clients with explicit credentials
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            
            self.cloudwatch_client = session.client('cloudwatch')
            self.ec2_client = session.client('ec2')
            self.s3_client = session.client('s3')
            
            # Test authentication by making a simple API call
            self.ec2_client.describe_regions(MaxResults=1)
            
            return True
        except (ClientError, BotoCoreError) as e:
            self.logger.error(f"AWS authentication failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during AWS authentication: {str(e)}")
            return False
    
    def get_aws_metrics(self) -> List[Dict]:
        """
        Fetch AWS cloud usage metrics (compute, storage, network).
        
        Collects metrics from:
        - EC2 instances: CPU utilization
        - S3 buckets: Storage consumption
        - EC2 instances: Network bandwidth (NetworkIn, NetworkOut)
        
        Returns:
            List of standardized metric dictionaries
        
        Raises:
            RuntimeError: If boto3 not available or authentication fails
        """
        if not BOTO3_AVAILABLE:
            raise RuntimeError(
                "boto3 library is not available. Please install it using: pip install boto3"
            )
        
        try:
            settings = get_settings()
            if settings is None:
                raise RuntimeError("Settings not available")
            
            # Get AWS configuration
            aws_config = settings.get_cloud_config('aws')
            if not aws_config.get('access_key') or not aws_config.get('secret_key'):
                raise RuntimeError(
                    "AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                )
            
            # Authenticate if not already done
            if self.cloudwatch_client is None:
                if not self.authenticate(aws_config):
                    raise RuntimeError("AWS authentication failed")
            
            # Get time window from settings
            time_window_minutes = int(
                os.getenv("CLOUD_METRICS_TIME_WINDOW_MINUTES", "5")
            )
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            metrics = []
            
            # Get specific instance IDs from config if provided
            instance_ids_filter = os.getenv("AWS_INSTANCE_IDS", "").strip()
            instance_ids = [i.strip() for i in instance_ids_filter.split(",") if i.strip()] if instance_ids_filter else None
            
            # Discover EC2 instances
            try:
                if instance_ids:
                    response = self.ec2_client.describe_instances(InstanceIds=instance_ids)
                else:
                    response = self.ec2_client.describe_instances()
                
                instance_list = []
                for reservation in response.get('Reservations', []):
                    for instance in reservation.get('Instances', []):
                        if instance['State']['Name'] == 'running':
                            instance_list.append(instance['InstanceId'])
                
                # Collect CPU utilization for each EC2 instance
                for instance_id in instance_list:
                    try:
                        cpu_response = self.cloudwatch_client.get_metric_statistics(
                            Namespace='AWS/EC2',
                            MetricName='CPUUtilization',
                            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                            StartTime=start_time,
                            EndTime=end_time,
                            Period=60,
                            Statistics=['Average']
                        )
                        
                        if cpu_response.get('Datapoints'):
                            latest_point = max(cpu_response['Datapoints'], key=lambda x: x['Timestamp'])
                            metrics.append(standardize_metrics(
                                provider='aws',
                                metric_name='CPUUtilization',
                                metric_value=latest_point['Average'],
                                timestamp=latest_point['Timestamp'],
                                resource_type='Compute'
                            ))
                    except (ClientError, BotoCoreError) as e:
                        self.logger.warning(f"Failed to get CPU metrics for instance {instance_id}: {str(e)}")
                        continue
                
                # Collect network metrics for each EC2 instance
                for instance_id in instance_list:
                    try:
                        # Network In
                        net_in_response = self.cloudwatch_client.get_metric_statistics(
                            Namespace='AWS/EC2',
                            MetricName='NetworkIn',
                            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                            StartTime=start_time,
                            EndTime=end_time,
                            Period=60,
                            Statistics=['Sum']
                        )
                        
                        if net_in_response.get('Datapoints'):
                            latest_point = max(net_in_response['Datapoints'], key=lambda x: x['Timestamp'])
                            metrics.append(standardize_metrics(
                                provider='aws',
                                metric_name='NetworkIn',
                                metric_value=latest_point['Sum'],
                                timestamp=latest_point['Timestamp'],
                                resource_type='Network'
                            ))
                        
                        # Network Out
                        net_out_response = self.cloudwatch_client.get_metric_statistics(
                            Namespace='AWS/EC2',
                            MetricName='NetworkOut',
                            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                            StartTime=start_time,
                            EndTime=end_time,
                            Period=60,
                            Statistics=['Sum']
                        )
                        
                        if net_out_response.get('Datapoints'):
                            latest_point = max(net_out_response['Datapoints'], key=lambda x: x['Timestamp'])
                            metrics.append(standardize_metrics(
                                provider='aws',
                                metric_name='NetworkOut',
                                metric_value=latest_point['Sum'],
                                timestamp=latest_point['Timestamp'],
                                resource_type='Network'
                            ))
                    except (ClientError, BotoCoreError) as e:
                        self.logger.warning(f"Failed to get network metrics for instance {instance_id}: {str(e)}")
                        continue
                        
            except (ClientError, BotoCoreError) as e:
                self.logger.warning(f"Failed to discover EC2 instances: {str(e)}")
            
            # Discover S3 buckets and collect storage metrics
            try:
                bucket_names_filter = os.getenv("AWS_S3_BUCKET_NAMES", "").strip()
                bucket_names = [b.strip() for b in bucket_names_filter.split(",") if b.strip()] if bucket_names_filter else None
                
                if bucket_names:
                    buckets = bucket_names
                else:
                    response = self.s3_client.list_buckets()
                    buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
                
                for bucket_name in buckets:
                    try:
                        # Get bucket size using CloudWatch
                        size_response = self.cloudwatch_client.get_metric_statistics(
                            Namespace='AWS/S3',
                            MetricName='BucketSizeBytes',
                            Dimensions=[
                                {'Name': 'BucketName', 'Value': bucket_name},
                                {'Name': 'StorageType', 'Value': 'StandardStorage'}
                            ],
                            StartTime=start_time,
                            EndTime=end_time,
                            Period=86400,  # Daily metric
                            Statistics=['Average']
                        )
                        
                        if size_response.get('Datapoints'):
                            latest_point = max(size_response['Datapoints'], key=lambda x: x['Timestamp'])
                            metrics.append(standardize_metrics(
                                provider='aws',
                                metric_name='BucketSizeBytes',
                                metric_value=latest_point['Average'],
                                timestamp=latest_point['Timestamp'],
                                resource_type='Storage'
                            ))
                        
                        # Get object count
                        obj_count_response = self.cloudwatch_client.get_metric_statistics(
                            Namespace='AWS/S3',
                            MetricName='NumberOfObjects',
                            Dimensions=[
                                {'Name': 'BucketName', 'Value': bucket_name},
                                {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
                            ],
                            StartTime=start_time,
                            EndTime=end_time,
                            Period=86400,
                            Statistics=['Average']
                        )
                        
                        if obj_count_response.get('Datapoints'):
                            latest_point = max(obj_count_response['Datapoints'], key=lambda x: x['Timestamp'])
                            metrics.append(standardize_metrics(
                                provider='aws',
                                metric_name='NumberOfObjects',
                                metric_value=latest_point['Average'],
                                timestamp=latest_point['Timestamp'],
                                resource_type='Storage'
                            ))
                    except (ClientError, BotoCoreError) as e:
                        self.logger.warning(f"Failed to get storage metrics for bucket {bucket_name}: {str(e)}")
                        continue
                        
            except (ClientError, BotoCoreError) as e:
                self.logger.warning(f"Failed to discover S3 buckets: {str(e)}")
            
            return metrics
            
        except RuntimeError:
            raise
        except (ClientError, BotoCoreError) as e:
            error_msg = f"Failed to collect AWS metrics: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error collecting AWS metrics: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    def collect_cost_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect AWS cost metrics for a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
        
        Returns:
            Dictionary containing cost metrics
        
        TODO: Implement AWS Cost Explorer API integration
        """
        # TODO: Implement AWS cost collection
        return {
            "provider": "aws",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_cost": 0.0,
            "services": {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def collect_resource_metrics(self) -> Dict:
        """
        Collect AWS resource usage metrics.
        
        Returns:
            Dictionary containing resource metrics
        
        TODO: Implement AWS resource metric collection
        """
        # TODO: Implement AWS resource collection
        return {
            "provider": "aws",
            "ec2_instances": 0,
            "s3_buckets": 0,
            "rds_instances": 0,
            "lambda_functions": 0,
            "timestamp": datetime.utcnow().isoformat()
        }


class AzureMetricsCollector(CloudProvider):
    """Collector for Azure cloud metrics."""
    
    def __init__(self):
        """Initialize Azure metrics collector."""
        self.monitor_client = None
        self.compute_client = None
        self.storage_client = None
        self.subscription_id = None
        self.credential = None
        self.logger = get_logger(__name__)
    
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with Azure using provided credentials.
        
        Args:
            credentials: Dictionary containing Azure credentials (subscription_id, client_id, client_secret, tenant_id)
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not AZURE_MONITORING_AVAILABLE:
            self.logger.error("Azure Monitor library is not available")
            return False
        
        try:
            subscription_id = credentials.get('subscription_id')
            client_id = credentials.get('client_id')
            client_secret = credentials.get('client_secret')
            tenant_id = credentials.get('tenant_id')
            
            if not subscription_id:
                self.logger.error("Azure subscription_id is required")
                return False
            
            self.subscription_id = subscription_id
            
            # Use service principal if credentials provided, otherwise use DefaultAzureCredential
            if client_id and client_secret and tenant_id:
                self.credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            else:
                # Try DefaultAzureCredential (for managed identity, environment variables, etc.)
                self.credential = DefaultAzureCredential()
            
            # Initialize clients
            self.monitor_client = MonitorManagementClient(
                self.credential,
                subscription_id
            )
            self.compute_client = ComputeManagementClient(
                self.credential,
                subscription_id
            )
            self.storage_client = StorageManagementClient(
                self.credential,
                subscription_id
            )
            
            # Test authentication by making a simple API call
            list(self.compute_client.virtual_machines.list_all()[:1])
            
            return True
        except AzureError as e:
            self.logger.error(f"Azure authentication failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during Azure authentication: {str(e)}")
            return False
    
    def get_azure_metrics(self) -> List[Dict]:
        """
        Fetch Azure cloud usage metrics (compute, storage, network).
        
        Collects metrics from:
        - Virtual Machines: CPU utilization (Percentage CPU)
        - Blob Storage: Storage consumption
        - Virtual Machines: Network bandwidth
        
        Returns:
            List of standardized metric dictionaries
        
        Raises:
            RuntimeError: If Azure Monitor not available or authentication fails
        """
        if not AZURE_MONITORING_AVAILABLE:
            raise RuntimeError(
                "Azure Monitor library is not available. "
                "Please install it using: pip install azure-mgmt-monitor azure-mgmt-compute azure-mgmt-storage azure-identity"
            )
        
        try:
            settings = get_settings()
            if settings is None:
                raise RuntimeError("Settings not available")
            
            # Get Azure configuration
            azure_config = settings.get_cloud_config('azure')
            if not azure_config.get('subscription_id'):
                raise RuntimeError(
                    "Azure subscription_id not configured. Set AZURE_SUBSCRIPTION_ID"
                )
            
            # Authenticate if not already done
            if self.monitor_client is None:
                if not self.authenticate(azure_config):
                    raise RuntimeError("Azure authentication failed")
            
            # Get time window from settings
            time_window_minutes = int(
                os.getenv("CLOUD_METRICS_TIME_WINDOW_MINUTES", "5")
            )
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            # Format timespan for Azure API (ISO 8601 format)
            timespan = f"{start_time.isoformat()}/{end_time.isoformat()}"
            
            metrics = []
            
            # Get specific VM names from config if provided
            vm_names_filter = os.getenv("AZURE_VM_NAMES", "").strip()
            vm_names = [v.strip() for v in vm_names_filter.split(",") if v.strip()] if vm_names_filter else None
            
            # Discover Virtual Machines
            try:
                vm_list = []
                vms = self.compute_client.virtual_machines.list_all()
                
                for vm in vms:
                    if vm_names is None or vm.name in vm_names:
                        # Get resource group from ID
                        resource_group = vm.id.split('/')[4]
                        vm_list.append({
                            'name': vm.name,
                            'resource_group': resource_group,
                            'resource_id': vm.id
                        })
                
                # Collect CPU utilization for each Virtual Machine
                for vm in vm_list:
                    try:
                        resource_uri = vm['resource_id']
                        
                        metrics_data = self.monitor_client.metrics.list(
                            resource_uri=resource_uri,
                            timespan=timespan,
                            metricnames="Percentage CPU",
                            aggregation="Average",
                            interval=timedelta(minutes=1)
                        )
                        
                        for metric in metrics_data.value:
                            for timeseries in metric.timeseries:
                                for data_point in timeseries.data:
                                    if data_point.average is not None:
                                        metrics.append(standardize_metrics(
                                            provider='azure',
                                            metric_name='Percentage CPU',
                                            metric_value=data_point.average,
                                            timestamp=data_point.time_stamp,
                                            resource_type='Compute'
                                        ))
                    except AzureError as e:
                        self.logger.warning(f"Failed to get CPU metrics for VM {vm['name']}: {str(e)}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Unexpected error getting CPU metrics for VM {vm['name']}: {str(e)}")
                        continue
                
                # Collect network metrics for each Virtual Machine
                for vm in vm_list:
                    try:
                        resource_uri = vm['resource_id']
                        
                        # Network In
                        metrics_data = self.monitor_client.metrics.list(
                            resource_uri=resource_uri,
                            timespan=timespan,
                            metricnames="Network In Total",
                            aggregation="Total",
                            interval=timedelta(minutes=1)
                        )
                        
                        for metric in metrics_data.value:
                            for timeseries in metric.timeseries:
                                for data_point in timeseries.data:
                                    if data_point.total is not None:
                                        metrics.append(standardize_metrics(
                                            provider='azure',
                                            metric_name='NetworkIn',
                                            metric_value=data_point.total,
                                            timestamp=data_point.time_stamp,
                                            resource_type='Network'
                                        ))
                        
                        # Network Out
                        metrics_data = self.monitor_client.metrics.list(
                            resource_uri=resource_uri,
                            timespan=timespan,
                            metricnames="Network Out Total",
                            aggregation="Total",
                            interval=timedelta(minutes=1)
                        )
                        
                        for metric in metrics_data.value:
                            for timeseries in metric.timeseries:
                                for data_point in timeseries.data:
                                    if data_point.total is not None:
                                        metrics.append(standardize_metrics(
                                            provider='azure',
                                            metric_name='NetworkOut',
                                            metric_value=data_point.total,
                                            timestamp=data_point.time_stamp,
                                            resource_type='Network'
                                        ))
                    except AzureError as e:
                        self.logger.warning(f"Failed to get network metrics for VM {vm['name']}: {str(e)}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Unexpected error getting network metrics for VM {vm['name']}: {str(e)}")
                        continue
                        
            except AzureError as e:
                self.logger.warning(f"Failed to discover Virtual Machines: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Unexpected error discovering Virtual Machines: {str(e)}")
            
            # Discover Storage Accounts and collect storage metrics
            try:
                storage_accounts = self.storage_client.storage_accounts.list()
                
                for account in storage_accounts:
                    try:
                        resource_group = account.id.split('/')[4]
                        account_name = account.name
                        
                        # Get storage account metrics
                        resource_uri = f"/subscriptions/{self.subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Storage/storageAccounts/{account_name}"
                        
                        metrics_data = self.monitor_client.metrics.list(
                            resource_uri=resource_uri,
                            timespan=timespan,
                            metricnames="UsedCapacity",
                            aggregation="Average",
                            interval=timedelta(minutes=1)
                        )
                        
                        for metric in metrics_data.value:
                            for timeseries in metric.timeseries:
                                for data_point in timeseries.data:
                                    if data_point.average is not None:
                                        metrics.append(standardize_metrics(
                                            provider='azure',
                                            metric_name='UsedCapacity',
                                            metric_value=data_point.average,
                                            timestamp=data_point.time_stamp,
                                            resource_type='Storage'
                                        ))
                    except AzureError as e:
                        self.logger.warning(f"Failed to get storage metrics for account {account.name}: {str(e)}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Unexpected error getting storage metrics for account {account.name}: {str(e)}")
                        continue
                        
            except AzureError as e:
                self.logger.warning(f"Failed to discover Storage Accounts: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Unexpected error discovering Storage Accounts: {str(e)}")
            
            return metrics
            
        except RuntimeError:
            raise
        except AzureError as e:
            error_msg = f"Azure API error: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error collecting Azure metrics: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    def collect_cost_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect Azure cost metrics for a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
        
        Returns:
            Dictionary containing cost metrics
        
        TODO: Implement Azure Cost Management API integration
        """
        # TODO: Implement Azure cost collection
        return {
            "provider": "azure",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_cost": 0.0,
            "resources": {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def collect_resource_metrics(self) -> Dict:
        """
        Collect Azure resource usage metrics.
        
        Returns:
            Dictionary containing resource metrics
        
        TODO: Implement Azure resource metric collection
        """
        # TODO: Implement Azure resource collection
        return {
            "provider": "azure",
            "virtual_machines": 0,
            "storage_accounts": 0,
            "sql_databases": 0,
            "timestamp": datetime.utcnow().isoformat()
        }


class GCPMetricsCollector(CloudProvider):
    """Collector for GCP cloud metrics."""
    
    def __init__(self):
        """Initialize GCP metrics collector."""
        self.monitoring_client = None
        self.compute_client = None
        self.storage_client = None
        self.project_id = None
        self.logger = get_logger(__name__)
    
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with GCP using provided credentials.
        
        Args:
            credentials: Dictionary containing GCP credentials (project_id, credentials_path)
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not GCP_MONITORING_AVAILABLE:
            self.logger.error("Google Cloud Monitoring library is not available")
            return False
        
        try:
            project_id = credentials.get('project_id')
            credentials_path = credentials.get('credentials_path')
            
            if not project_id:
                self.logger.error("GCP project_id is required")
                return False
            
            self.project_id = project_id
            
            # Set credentials path if provided
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            # Initialize clients
            self.monitoring_client = monitoring_v3.MetricServiceClient()
            self.compute_client = compute_v1.InstancesClient()
            self.storage_client = gcp_storage.Client(project=project_id)
            
            # Test authentication by making a simple API call
            project_name = f"projects/{project_id}"
            list(self.monitoring_client.list_metric_descriptors(name=project_name, page_size=1))
            
            return True
        except GoogleAuthError as e:
            self.logger.error(f"GCP authentication failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during GCP authentication: {str(e)}")
            return False
    
    def get_gcp_metrics(self) -> List[Dict]:
        """
        Fetch GCP cloud usage metrics (compute, storage, network).
        
        Collects metrics from:
        - Compute Engine instances: CPU utilization
        - Cloud Storage buckets: Storage consumption
        - Compute Engine instances: Network egress
        
        Returns:
            List of standardized metric dictionaries
        
        Raises:
            RuntimeError: If Google Cloud Monitoring not available or authentication fails
        """
        if not GCP_MONITORING_AVAILABLE:
            raise RuntimeError(
                "Google Cloud Monitoring library is not available. "
                "Please install it using: pip install google-cloud-monitoring google-cloud-compute google-cloud-storage"
            )
        
        try:
            settings = get_settings()
            if settings is None:
                raise RuntimeError("Settings not available")
            
            # Get GCP configuration
            gcp_config = settings.get_cloud_config('gcp')
            if not gcp_config.get('project_id'):
                raise RuntimeError(
                    "GCP project_id not configured. Set GCP_PROJECT_ID"
                )
            
            # Authenticate if not already done
            if self.monitoring_client is None:
                if not self.authenticate(gcp_config):
                    raise RuntimeError("GCP authentication failed")
            
            project_id = self.project_id
            project_name = f"projects/{project_id}"
            
            # Get time window from settings
            time_window_minutes = int(
                os.getenv("CLOUD_METRICS_TIME_WINDOW_MINUTES", "5")
            )
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            # Convert to protobuf Timestamp format
            from google.protobuf.timestamp_pb2 import Timestamp
            interval_start = Timestamp()
            interval_start.FromDatetime(start_time)
            interval_end = Timestamp()
            interval_end.FromDatetime(end_time)
            
            metrics = []
            
            # Get specific instance names from config if provided
            instance_names_filter = os.getenv("GCP_INSTANCE_NAMES", "").strip()
            instance_names = [i.strip() for i in instance_names_filter.split(",") if i.strip()] if instance_names_filter else None
            
            # Discover Compute Engine instances
            try:
                zone_list = []
                # List all zones
                try:
                    zones_client = compute_v1.ZonesClient()
                    zones = zones_client.list(project=project_id)
                    zone_list = [zone.name for zone in zones]
                except Exception as e:
                    self.logger.warning(f"Failed to list zones, using default: {str(e)}")
                    zone_list = ['us-central1-a']  # Default zone
                
                instance_list = []
                for zone in zone_list:
                    try:
                        instances = self.compute_client.list(project=project_id, zone=zone)
                        for instance in instances:
                            if instance.status == 'RUNNING':
                                if instance_names is None or instance.name in instance_names:
                                    instance_list.append({
                                        'name': instance.name,
                                        'zone': zone,
                                        'id': instance.id
                                    })
                    except Exception as e:
                        self.logger.warning(f"Failed to list instances in zone {zone}: {str(e)}")
                        continue
                
                # Collect CPU utilization for each Compute Engine instance
                for instance in instance_list:
                    try:
                        filter_str = (
                            f'metric.type="compute.googleapis.com/instance/cpu/utilization" '
                            f'AND resource.labels.instance_id="{instance["id"]}"'
                        )
                        
                        results = self.monitoring_client.list_time_series(
                            name=project_name,
                            filter=filter_str,
                            interval={
                                'start_time': interval_start,
                                'end_time': interval_end
                            },
                            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                        )
                        
                        for result in results:
                            if result.points:
                                latest_point = result.points[-1]
                                metrics.append(standardize_metrics(
                                    provider='gcp',
                                    metric_name='CPUUtilization',
                                    metric_value=latest_point.value.double_value if latest_point.value.HasField('double_value') else 0.0,
                                    timestamp=datetime.fromtimestamp(
                                        latest_point.interval.end_time.seconds + 
                                        latest_point.interval.end_time.nanos / 1e9
                                    ),
                                    resource_type='Compute'
                                ))
                    except Exception as e:
                        self.logger.warning(f"Failed to get CPU metrics for instance {instance['name']}: {str(e)}")
                        continue
                
                # Collect network egress for each Compute Engine instance
                for instance in instance_list:
                    try:
                        filter_str = (
                            f'metric.type="compute.googleapis.com/instance/network/sent_bytes_count" '
                            f'AND resource.labels.instance_id="{instance["id"]}"'
                        )
                        
                        results = self.monitoring_client.list_time_series(
                            name=project_name,
                            filter=filter_str,
                            interval={
                                'start_time': interval_start,
                                'end_time': interval_end
                            },
                            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                        )
                        
                        for result in results:
                            if result.points:
                                latest_point = result.points[-1]
                                metrics.append(standardize_metrics(
                                    provider='gcp',
                                    metric_name='NetworkEgress',
                                    metric_value=latest_point.value.int64_value if latest_point.value.HasField('int64_value') else latest_point.value.double_value if latest_point.value.HasField('double_value') else 0.0,
                                    timestamp=datetime.fromtimestamp(
                                        latest_point.interval.end_time.seconds + 
                                        latest_point.interval.end_time.nanos / 1e9
                                    ),
                                    resource_type='Network'
                                ))
                    except Exception as e:
                        self.logger.warning(f"Failed to get network metrics for instance {instance['name']}: {str(e)}")
                        continue
                        
            except Exception as e:
                self.logger.warning(f"Failed to discover Compute Engine instances: {str(e)}")
            
            # Discover Cloud Storage buckets and collect storage metrics
            try:
                bucket_names_filter = os.getenv("GCP_BUCKET_NAMES", "").strip()
                bucket_names = [b.strip() for b in bucket_names_filter.split(",") if b.strip()] if bucket_names_filter else None
                
                if bucket_names:
                    buckets = bucket_names
                else:
                    buckets = [bucket.name for bucket in self.storage_client.list_buckets()]
                
                for bucket_name in buckets:
                    try:
                        bucket = self.storage_client.bucket(bucket_name)
                        
                        # Get bucket size
                        total_size = 0
                        object_count = 0
                        for blob in bucket.list_blobs():
                            total_size += blob.size
                            object_count += 1
                        
                        if total_size > 0:
                            metrics.append(standardize_metrics(
                                provider='gcp',
                                metric_name='BucketSizeBytes',
                                metric_value=total_size,
                                timestamp=datetime.utcnow(),
                                resource_type='Storage'
                            ))
                        
                        if object_count > 0:
                            metrics.append(standardize_metrics(
                                provider='gcp',
                                metric_name='NumberOfObjects',
                                metric_value=object_count,
                                timestamp=datetime.utcnow(),
                                resource_type='Storage'
                            ))
                    except Exception as e:
                        self.logger.warning(f"Failed to get storage metrics for bucket {bucket_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                self.logger.warning(f"Failed to discover Cloud Storage buckets: {str(e)}")
            
            return metrics
            
        except RuntimeError:
            raise
        except GoogleAuthError as e:
            error_msg = f"GCP authentication error: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error collecting GCP metrics: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    def collect_cost_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect GCP cost metrics for a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
        
        Returns:
            Dictionary containing cost metrics
        
        TODO: Implement GCP Billing API integration
        """
        # TODO: Implement GCP cost collection
        return {
            "provider": "gcp",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_cost": 0.0,
            "projects": {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def collect_resource_metrics(self) -> Dict:
        """
        Collect GCP resource usage metrics.
        
        Returns:
            Dictionary containing resource metrics
        
        TODO: Implement GCP resource metric collection
        """
        # TODO: Implement GCP resource collection
        return {
            "provider": "gcp",
            "compute_instances": 0,
            "storage_buckets": 0,
            "sql_instances": 0,
            "timestamp": datetime.utcnow().isoformat()
        }


def get_aws_metrics() -> List[Dict]:
    """
    Fetch AWS cloud usage metrics (compute, storage, network).
    
    This is a standalone function that collects metrics from AWS CloudWatch
    for all EC2 instances (CPU utilization), S3 buckets (storage), and
    network bandwidth metrics.
    
    Returns:
        List of standardized metric dictionaries with format:
        {
            'provider': 'aws',
            'metric_name': str,
            'metric_value': float,
            'timestamp': str (ISO format),
            'resource_type': str ('Compute', 'Storage', 'Network')
        }
    
    Raises:
        RuntimeError: If boto3 not available or authentication fails
    """
    if not BOTO3_AVAILABLE:
        raise RuntimeError(
            "boto3 library is not available. Please install it using: pip install boto3"
        )
    
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available")
    
    aws_config = settings.get_cloud_config('aws')
    collector = AWSMetricsCollector()
    
    if not collector.authenticate(aws_config):
        raise RuntimeError("AWS authentication failed. Check your AWS credentials.")
    
    return collector.get_aws_metrics()


def get_gcp_metrics() -> List[Dict]:
    """
    Fetch GCP cloud usage metrics (compute, storage, network).
    
    This is a standalone function that collects metrics from Google Cloud
    Monitoring for all Compute Engine instances (CPU utilization), Cloud
    Storage buckets (storage), and network egress metrics.
    
    Returns:
        List of standardized metric dictionaries with format:
        {
            'provider': 'gcp',
            'metric_name': str,
            'metric_value': float,
            'timestamp': str (ISO format),
            'resource_type': str ('Compute', 'Storage', 'Network')
        }
    
    Raises:
        RuntimeError: If Google Cloud Monitoring not available or authentication fails
    """
    if not GCP_MONITORING_AVAILABLE:
        raise RuntimeError(
            "Google Cloud Monitoring library is not available. "
            "Please install it using: pip install google-cloud-monitoring"
        )
    
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available")
    
    gcp_config = settings.get_cloud_config('gcp')
    collector = GCPMetricsCollector()
    
    if not collector.authenticate(gcp_config):
        raise RuntimeError("GCP authentication failed. Check your GCP credentials.")
    
    return collector.get_gcp_metrics()


def get_azure_metrics() -> List[Dict]:
    """
    Fetch Azure cloud usage metrics (compute, storage, network).
    
    This is a standalone function that collects metrics from Azure Monitor
    for all Virtual Machines (CPU utilization), Blob Storage accounts
    (storage consumption), and network bandwidth metrics.
    
    Returns:
        List of standardized metric dictionaries with format:
        {
            'provider': 'azure',
            'metric_name': str,
            'metric_value': float,
            'timestamp': str (ISO format),
            'resource_type': str ('Compute', 'Storage', 'Network')
        }
    
    Raises:
        RuntimeError: If Azure Monitor not available or authentication fails
    """
    if not AZURE_MONITORING_AVAILABLE:
        raise RuntimeError(
            "Azure Monitor library is not available. "
            "Please install it using: pip install azure-mgmt-monitor"
        )
    
    settings = get_settings()
    if settings is None:
        raise RuntimeError("Settings not available")
    
    azure_config = settings.get_cloud_config('azure')
    collector = AzureMetricsCollector()
    
    if not collector.authenticate(azure_config):
        raise RuntimeError("Azure authentication failed. Check your Azure credentials.")
    
    return collector.get_azure_metrics()


def collect_cloud_metrics(provider: str, credentials: Dict) -> Dict:
    """
    Collect cloud metrics for a specific provider.
    
    Args:
        provider: Cloud provider name ('aws', 'azure', 'gcp')
        credentials: Provider-specific credentials
    
    Returns:
        Dictionary containing cloud metrics
    
    TODO: Implement provider selection and metric collection
    """
    # TODO: Implement provider selection and collection
    collectors = {
        "aws": AWSMetricsCollector(),
        "azure": AzureMetricsCollector(),
        "gcp": GCPMetricsCollector()
    }
    
    if provider not in collectors:
        raise ValueError(f"Unsupported cloud provider: {provider}")
    
    collector = collectors[provider]
    if not collector.authenticate(credentials):
        raise ValueError(f"Authentication failed for {provider}")
    
    return {
        "cost_metrics": collector.collect_cost_metrics(
            datetime.utcnow(), datetime.utcnow()
        ),
        "resource_metrics": collector.collect_resource_metrics()
    }


def calculate_aws_cost(usage_metrics: List[Dict]) -> List[Dict]:
    """
    Calculate AWS costs based on resource usage metrics.
    
    Uses example pricing:
    - EC2 CPU: $0.01 per percent per hour (example)
    - S3 Storage: $0.023 per GB per month
    - Network: $0.09 per GB
    
    Args:
        usage_metrics: List of standardized metric dictionaries from get_aws_metrics()
    
    Returns:
        List of dictionaries with format:
        {
            'resource_name': str,      # e.g., 'AWS EC2', 'S3 Bucket'
            'resource_usage': float,  # Usage amount
            'cost': float,            # Calculated cost
            'timestamp': datetime     # Timestamp from metric
        }
    """
    cost_data = []
    current_time = datetime.utcnow()
    
    # Pricing constants (example values - can be made configurable)
    EC2_CPU_COST_PER_PERCENT_PER_HOUR = 0.01
    S3_STORAGE_COST_PER_GB_PER_MONTH = 0.023
    NETWORK_COST_PER_GB = 0.09
    
    for metric in usage_metrics:
        resource_type = metric.get('resource_type', '')
        metric_name = metric.get('metric_name', '')
        metric_value = metric.get('metric_value', 0.0)
        timestamp_str = metric.get('timestamp', '')
        
        # Parse timestamp
        try:
            if isinstance(timestamp_str, str):
                # Try parsing ISO format
                if 'T' in timestamp_str or '+' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Convert to UTC naive
                    if timestamp.tzinfo:
                        timestamp = timestamp.replace(tzinfo=None)
                else:
                    timestamp = current_time
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
                if timestamp.tzinfo:
                    timestamp = timestamp.replace(tzinfo=None)
            else:
                timestamp = current_time
        except Exception:
            timestamp = current_time
        
        cost = 0.0
        resource_name = ""
        resource_usage = 0.0
        
        if resource_type == 'Compute' and metric_name == 'CPUUtilization':
            # CPU cost: percentage * cost per percent per hour
            # Assuming 1 hour window for real-time calculation
            resource_usage = metric_value  # CPU percentage
            cost = metric_value * EC2_CPU_COST_PER_PERCENT_PER_HOUR
            resource_name = "AWS EC2"
        
        elif resource_type == 'Storage':
            if metric_name == 'BucketSizeBytes':
                # Storage cost: convert bytes to GB, then apply monthly cost
                # For real-time, we calculate hourly cost (monthly / 730 hours)
                resource_usage = metric_value / (1024 ** 3)  # Convert bytes to GB
                cost = resource_usage * (S3_STORAGE_COST_PER_GB_PER_MONTH / 730.0)  # Hourly cost
                resource_name = "S3 Storage"
            elif metric_name == 'NumberOfObjects':
                # Object count - minimal cost, skip for now or add object-based pricing
                continue
        
        elif resource_type == 'Network':
            if metric_name in ['NetworkIn', 'NetworkOut']:
                # Network cost: convert bytes to GB, then apply cost per GB
                resource_usage = metric_value / (1024 ** 3)  # Convert bytes to GB
                cost = resource_usage * NETWORK_COST_PER_GB
                resource_name = f"AWS Network ({metric_name})"
        
        if cost > 0 and resource_name:
            cost_data.append({
                'resource_name': resource_name,
                'resource_usage': resource_usage,
                'cost': cost,
                'timestamp': timestamp
            })
    
    return cost_data


def calculate_azure_cost(usage_metrics: List[Dict]) -> List[Dict]:
    """
    Calculate Azure costs based on resource usage metrics.
    
    Uses example pricing:
    - VM CPU: $0.012 per percent per hour (example)
    - Blob Storage: $0.018 per GB per month
    - Network: $0.05 per GB
    
    Args:
        usage_metrics: List of standardized metric dictionaries from get_azure_metrics()
    
    Returns:
        List of dictionaries with format:
        {
            'resource_name': str,      # e.g., 'Azure VM', 'Blob Storage'
            'resource_usage': float,   # Usage amount
            'cost': float,             # Calculated cost
            'timestamp': datetime      # Timestamp from metric
        }
    """
    cost_data = []
    current_time = datetime.utcnow()
    
    # Pricing constants (example values - can be made configurable)
    VM_CPU_COST_PER_PERCENT_PER_HOUR = 0.012
    BLOB_STORAGE_COST_PER_GB_PER_MONTH = 0.018
    NETWORK_COST_PER_GB = 0.05
    
    for metric in usage_metrics:
        resource_type = metric.get('resource_type', '')
        metric_name = metric.get('metric_name', '')
        metric_value = metric.get('metric_value', 0.0)
        timestamp_str = metric.get('timestamp', '')
        
        # Parse timestamp
        try:
            if isinstance(timestamp_str, str):
                if 'T' in timestamp_str or '+' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp.tzinfo:
                        timestamp = timestamp.replace(tzinfo=None)
                else:
                    timestamp = current_time
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
                if timestamp.tzinfo:
                    timestamp = timestamp.replace(tzinfo=None)
            else:
                timestamp = current_time
        except Exception:
            timestamp = current_time
        
        cost = 0.0
        resource_name = ""
        resource_usage = 0.0
        
        if resource_type == 'Compute' and metric_name == 'Percentage CPU':
            # CPU cost: percentage * cost per percent per hour
            resource_usage = metric_value  # CPU percentage
            cost = metric_value * VM_CPU_COST_PER_PERCENT_PER_HOUR
            resource_name = "Azure VM"
        
        elif resource_type == 'Storage' and metric_name == 'UsedCapacity':
            # Storage cost: convert bytes to GB, then apply monthly cost
            # For real-time, we calculate hourly cost (monthly / 730 hours)
            resource_usage = metric_value / (1024 ** 3)  # Convert bytes to GB
            cost = resource_usage * (BLOB_STORAGE_COST_PER_GB_PER_MONTH / 730.0)  # Hourly cost
            resource_name = "Azure Blob Storage"
        
        elif resource_type == 'Network':
            if metric_name in ['NetworkIn', 'NetworkOut']:
                # Network cost: convert bytes to GB, then apply cost per GB
                resource_usage = metric_value / (1024 ** 3)  # Convert bytes to GB
                cost = resource_usage * NETWORK_COST_PER_GB
                resource_name = f"Azure Network ({metric_name})"
        
        if cost > 0 and resource_name:
            cost_data.append({
                'resource_name': resource_name,
                'resource_usage': resource_usage,
                'cost': cost,
                'timestamp': timestamp
            })
    
    return cost_data


def calculate_gcp_cost(usage_metrics: List[Dict]) -> List[Dict]:
    """
    Calculate GCP costs based on resource usage metrics.
    
    Uses example pricing:
    - Compute Engine CPU: $0.01 per percent per hour (example)
    - Cloud Storage: $0.020 per GB per month
    - Network: $0.12 per GB
    
    Args:
        usage_metrics: List of standardized metric dictionaries from get_gcp_metrics()
    
    Returns:
        List of dictionaries with format:
        {
            'resource_name': str,      # e.g., 'GCP Compute Engine', 'Cloud Storage'
            'resource_usage': float,   # Usage amount
            'cost': float,             # Calculated cost
            'timestamp': datetime      # Timestamp from metric
        }
    """
    cost_data = []
    current_time = datetime.utcnow()
    
    # Pricing constants (example values - can be made configurable)
    COMPUTE_CPU_COST_PER_PERCENT_PER_HOUR = 0.01
    CLOUD_STORAGE_COST_PER_GB_PER_MONTH = 0.020
    NETWORK_COST_PER_GB = 0.12
    
    for metric in usage_metrics:
        resource_type = metric.get('resource_type', '')
        metric_name = metric.get('metric_name', '')
        metric_value = metric.get('metric_value', 0.0)
        timestamp_str = metric.get('timestamp', '')
        
        # Parse timestamp
        try:
            if isinstance(timestamp_str, str):
                if 'T' in timestamp_str or '+' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp.tzinfo:
                        timestamp = timestamp.replace(tzinfo=None)
                else:
                    timestamp = current_time
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
                if timestamp.tzinfo:
                    timestamp = timestamp.replace(tzinfo=None)
            else:
                timestamp = current_time
        except Exception:
            timestamp = current_time
        
        cost = 0.0
        resource_name = ""
        resource_usage = 0.0
        
        if resource_type == 'Compute' and metric_name == 'CPUUtilization':
            # CPU cost: percentage * cost per percent per hour
            resource_usage = metric_value  # CPU percentage
            cost = metric_value * COMPUTE_CPU_COST_PER_PERCENT_PER_HOUR
            resource_name = "GCP Compute Engine"
        
        elif resource_type == 'Storage':
            if metric_name == 'BucketSizeBytes':
                # Storage cost: convert bytes to GB, then apply monthly cost
                # For real-time, we calculate hourly cost (monthly / 730 hours)
                resource_usage = metric_value / (1024 ** 3)  # Convert bytes to GB
                cost = resource_usage * (CLOUD_STORAGE_COST_PER_GB_PER_MONTH / 730.0)  # Hourly cost
                resource_name = "GCP Cloud Storage"
            elif metric_name == 'NumberOfObjects':
                # Object count - minimal cost, skip for now
                continue
        
        elif resource_type == 'Network' and metric_name == 'NetworkEgress':
            # Network cost: convert bytes to GB, then apply cost per GB
            resource_usage = metric_value / (1024 ** 3)  # Convert bytes to GB
            cost = resource_usage * NETWORK_COST_PER_GB
            resource_name = "GCP Network Egress"
        
        if cost > 0 and resource_name:
            cost_data.append({
                'resource_name': resource_name,
                'resource_usage': resource_usage,
                'cost': cost,
                'timestamp': timestamp
            })
    
    return cost_data


def get_cloud_costs(provider: Optional[str] = None, store_in_db: bool = True) -> Dict:
    """
    Fetch real-time cloud resource usage and calculate associated costs.
    
    This function collects usage metrics from cloud providers, calculates costs
    based on usage, and optionally stores the data in the database.
    
    Args:
        provider: Optional provider name ('aws', 'azure', 'gcp'). 
                  If None, collects from all enabled providers.
        store_in_db: If True, automatically store cost data in database (default: True)
    
    Returns:
        Dictionary with format:
        {
            'cloud_usage': List[Dict],  # Usage metrics from providers
            'cloud_cost': List[Dict],    # Cost data with resource_name, resource_usage, cost, timestamp
            'total_cost': float,         # Total cost across all resources
            'timestamp': datetime        # Current timestamp
        }
    
    Raises:
        RuntimeError: If settings not available or cloud metrics not enabled
    """
    logger = get_logger(__name__)
    settings = get_settings()
    
    if settings is None:
        raise RuntimeError("Settings not available. Cannot determine cloud configuration.")
    
    if not settings.enable_cloud_metrics:
        logger.warning("Cloud metrics collection is disabled in settings")
        return {
            'cloud_usage': [],
            'cloud_cost': [],
            'total_cost': 0.0,
            'timestamp': datetime.utcnow()
        }
    
    all_usage_metrics = []
    all_cost_data = []
    current_time = datetime.utcnow()
    
    # Determine which providers to collect from
    providers_to_collect = []
    if provider:
        if provider.lower() in ['aws', 'azure', 'gcp']:
            providers_to_collect = [provider.lower()]
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
    else:
        # Collect from all enabled providers
        providers_to_collect = ['aws', 'azure', 'gcp']
    
    # Collect usage metrics and calculate costs for each provider
    for prov in providers_to_collect:
        try:
            # Fetch usage metrics
            if prov == 'aws':
                usage_metrics = get_aws_metrics()
                cost_data = calculate_aws_cost(usage_metrics)
            elif prov == 'azure':
                usage_metrics = get_azure_metrics()
                cost_data = calculate_azure_cost(usage_metrics)
            elif prov == 'gcp':
                usage_metrics = get_gcp_metrics()
                cost_data = calculate_gcp_cost(usage_metrics)
            else:
                continue
            
            all_usage_metrics.extend(usage_metrics)
            all_cost_data.extend(cost_data)
            
        except RuntimeError as e:
            logger.warning(f"Failed to collect metrics from {prov}: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error collecting metrics from {prov}: {str(e)}", exc_info=True)
            continue
    
    # Calculate total cost
    total_cost = sum(item.get('cost', 0.0) for item in all_cost_data)
    
    # Store in database if requested
    if store_in_db and all_cost_data:
        try:
            from data_collection.database import store_cloud_cost_data
            for cost_item in all_cost_data:
                store_cloud_cost_data(
                    resource_name=cost_item['resource_name'],
                    resource_usage=cost_item['resource_usage'],
                    cost=cost_item['cost'],
                    timestamp=cost_item['timestamp']
                )
        except ImportError:
            logger.warning("Database module not available for storing cloud cost data")
        except Exception as e:
            logger.warning(f"Failed to store cloud cost data: {str(e)}")
    
    return {
        'cloud_usage': all_usage_metrics,
        'cloud_cost': all_cost_data,
        'total_cost': total_cost,
        'timestamp': current_time
    }


def generate_simulated_cloud_cost(
    resource_name: str,
    provider: str,
    base_cost: float,
    variation: float,
    timestamp: datetime
) -> Dict:
    """
    Generate simulated cloud cost data.
    
    Args:
        resource_name: Name of the cloud resource (e.g., 'AWS EC2', 'S3 Storage')
        provider: Cloud provider ('aws', 'azure', 'gcp')
        base_cost: Base cost in dollars
        variation: Maximum variation as fraction (e.g., 0.1 for 10%)
        timestamp: Timestamp for the metric
    
    Returns:
        Dictionary matching cloud cost format with resource_name, resource_usage, cost, timestamp
    """
    # Use resource_name hash for deterministic variation
    random.seed(hash(resource_name + str(timestamp)) % 1000)
    cost_variation = random.uniform(-variation, variation)
    cost = max(0.0, base_cost * (1 + cost_variation))
    
    # Calculate resource usage based on cost (simplified model)
    # Different resource types have different usage-to-cost ratios
    if 'EC2' in resource_name or 'VM' in resource_name or 'Compute' in resource_name:
        # Compute resources: usage in CPU-hours
        resource_usage = cost * 10.0  # $1 = 10 CPU-hours
    elif 'Storage' in resource_name or 'S3' in resource_name:
        # Storage resources: usage in GB
        resource_usage = cost * 100.0  # $1 = 100 GB
    else:
        # Network or other: usage in GB transferred
        resource_usage = cost * 50.0  # $1 = 50 GB
    
    return {
        "resource_name": resource_name,
        "resource_usage": resource_usage,
        "cost": cost,
        "timestamp": timestamp
    }
