"""
Configuration settings for Monitor/Drift Agent.

This module provides configuration management for the agent including
thresholds, alert channels, database connections, and other settings.
"""

import os
from typing import Dict, Optional, Any

# Try to load environment variables from .env file
# This allows users to create a .env file in the project root
# with their configuration instead of setting environment variables manually
try:
    from dotenv import load_dotenv
    # Load .env file from project root (parent of config directory)
    # This is the recommended location: project_root/.env
    project_root = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(project_root, '.env')
    
    # Try to load from project root first
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        # Fallback: try loading from current working directory
        # This allows .env file to be in the directory where the script is run from
        load_dotenv(override=False)
except ImportError:
    # python-dotenv is not installed, skip .env file loading
    # Environment variables must be set manually or via system
    # To enable .env support: pip install python-dotenv
    pass


class Settings:
    """Manages application settings."""
    
    def __init__(self):
        """Initialize settings from environment variables and defaults."""
        # Database settings
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://user:password@localhost:5432/monitor_drift"
        )
        
        # Metric collection settings
        self.metric_collection_interval = int(
            os.getenv("METRIC_COLLECTION_INTERVAL", "60")
        )  # seconds
        
        # Threshold settings
        self.default_cpu_threshold = float(
            os.getenv("DEFAULT_CPU_THRESHOLD", "80.0")
        )
        self.default_memory_threshold = float(
            os.getenv("DEFAULT_MEMORY_THRESHOLD", "80.0")
        )
        self.default_disk_threshold = float(
            os.getenv("DEFAULT_DISK_THRESHOLD", "80.0")
        )
        
        # Anomaly detection thresholds
        self.cpu_usage_threshold = float(
            os.getenv("CPU_USAGE_THRESHOLD", "80.0")
        )
        self.cpu_threshold_duration = int(
            os.getenv("CPU_THRESHOLD_DURATION", "5")
        )  # minutes
        self.cloud_cost_threshold = float(
            os.getenv("CLOUD_COST_THRESHOLD", "500.0")
        )  # dollars per day
        self.cost_spike_threshold_percent = float(
            os.getenv("COST_SPIKE_THRESHOLD_PERCENT", "20.0")
        )  # Percentage above historical average (default: 20%)
        self.historical_cost_lookback_days = int(
            os.getenv("HISTORICAL_COST_LOOKBACK_DAYS", "7")
        )  # Days to look back for historical data (default: 7)
        self.sustained_cost_spike_days = int(
            os.getenv("SUSTAINED_COST_SPIKE_DAYS", "3")
        )  # Number of consecutive days required to trigger alert (default: 3)
        
        # Monitoring control settings
        self.cost_spike_monitoring_enabled = os.getenv(
            "COST_SPIKE_MONITORING_ENABLED", "true"
        ).lower() == "true"  # Enable/disable automatic cost spike monitoring (default: true)
        self.system_metrics_monitoring_enabled = os.getenv(
            "SYSTEM_METRICS_MONITORING_ENABLED", "true"
        ).lower() == "true"  # Enable/disable automatic system metrics monitoring (default: true)
        
        # Machine Learning settings
        self.ml_training_lookback_days = int(
            os.getenv("ML_TRAINING_LOOKBACK_DAYS", "30")
        )  # Days of historical data for training (default: 30)
        self.ml_contamination = float(
            os.getenv("ML_CONTAMINATION", "0.05")
        )  # Expected proportion of anomalies (default: 0.05)
        self.ml_n_estimators = int(
            os.getenv("ML_N_ESTIMATORS", "100")
        )  # Number of trees in Isolation Forest (default: 100)
        self.ml_model_path = os.getenv(
            "ML_MODEL_PATH", "models/isolation_forest_model.pkl"
        )  # Path to save/load trained model
        self.ml_enabled = os.getenv("ML_ENABLED", "false").lower() == "true"  # Enable/disable ML-based detection
        self.ml_anomaly_score_threshold = float(
            os.getenv("ML_ANOMALY_SCORE_THRESHOLD", "-0.5")
        )  # Anomaly score threshold for alerting (default: -0.5, lower = more anomalous)
        
        # ML Retraining settings
        self.ml_retraining_enabled = os.getenv("ML_RETRAINING_ENABLED", "false").lower() == "true"  # Enable/disable automatic retraining
        self.ml_retraining_check_interval_hours = int(
            os.getenv("ML_RETRAINING_CHECK_INTERVAL_HOURS", "24")
        )  # Hours between performance checks (default: 24)
        self.ml_min_detection_rate = float(
            os.getenv("ML_MIN_DETECTION_RATE", "0.01")
        )  # Minimum acceptable detection rate (default: 0.01, i.e., 1%)
        self.ml_max_detection_rate = float(
            os.getenv("ML_MAX_DETECTION_RATE", "0.20")
        )  # Maximum acceptable detection rate (default: 0.20, i.e., 20%)
        self.ml_min_alert_rate = float(
            os.getenv("ML_MIN_ALERT_RATE", "0.50")
        )  # Minimum alert rate when anomalies detected (default: 0.50, i.e., 50%)
        self.ml_performance_lookback_days = int(
            os.getenv("ML_PERFORMANCE_LOOKBACK_DAYS", "7")
        )  # Days of performance data to analyze (default: 7)
        self.ml_retraining_threshold_accuracy = float(
            os.getenv("ML_RETRAINING_THRESHOLD_ACCURACY", "0.80")
        )  # Accuracy threshold for retraining (default: 0.80, for future use with labeled data)
        
        self.memory_usage_threshold = float(
            os.getenv("MEMORY_USAGE_THRESHOLD", "80.0")
        )
        self.disk_usage_threshold = float(
            os.getenv("DISK_USAGE_THRESHOLD", "80.0")
        )
        network_bandwidth_threshold_str = os.getenv("NETWORK_BANDWIDTH_THRESHOLD")
        if network_bandwidth_threshold_str:
            self.network_bandwidth_threshold = float(network_bandwidth_threshold_str)
        else:
            self.network_bandwidth_threshold = None
        
        # Cloud provider settings
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        self.azure_subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID", "")
        self.azure_client_id = os.getenv("AZURE_CLIENT_ID", "")
        self.azure_client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
        self.azure_tenant_id = os.getenv("AZURE_TENANT_ID", "")
        
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID", "")
        self.gcp_credentials_path = os.getenv("GCP_CREDENTIALS_PATH", "")
        
        # Cloud metrics collection settings
        self.cloud_metrics_time_window_minutes = int(
            os.getenv("CLOUD_METRICS_TIME_WINDOW_MINUTES", "5")
        )
        self.enable_cloud_metrics = os.getenv(
            "ENABLE_CLOUD_METRICS", "false"
        ).lower() == "true"
        self.aws_instance_ids = os.getenv("AWS_INSTANCE_IDS", "").strip()
        self.aws_s3_bucket_names = os.getenv("AWS_S3_BUCKET_NAMES", "").strip()
        self.gcp_instance_names = os.getenv("GCP_INSTANCE_NAMES", "").strip()
        self.gcp_bucket_names = os.getenv("GCP_BUCKET_NAMES", "").strip()
        self.azure_vm_names = os.getenv("AZURE_VM_NAMES", "").strip()
        
        # Alert channel settings
        self.email_smtp_server = os.getenv("EMAIL_SMTP_SERVER", "")
        self.email_smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.email_username = os.getenv("EMAIL_USERNAME", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        email_recipients_str = os.getenv("EMAIL_RECIPIENTS", "")
        self.email_recipients = [r.strip() for r in email_recipients_str.split(",") if r.strip()] if email_recipients_str else []
        self.email_use_sendgrid = os.getenv("EMAIL_USE_SENDGRID", "false").lower() == "true"
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY", "")
        
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        
        self.sms_api_key = os.getenv("SMS_API_KEY", "")
        self.sms_api_secret = os.getenv("SMS_API_SECRET", "")
        self.sms_from_number = os.getenv("SMS_FROM_NUMBER", "")
        self.sms_use_twilio = os.getenv("SMS_USE_TWILIO", "true").lower() == "true"
        self.sms_api_url = os.getenv("SMS_API_URL", "")
        self.sms_api_method = os.getenv("SMS_API_METHOD", "POST")
        
        # Alert preferences
        self.alert_preferences = self._parse_alert_preferences()
        
        # Anomaly detection settings
        self.anomaly_detection_enabled = os.getenv(
            "ANOMALY_DETECTION_ENABLED", "true"
        ).lower() == "true"
        self.ml_detection_enabled = os.getenv(
            "ML_DETECTION_ENABLED", "false"
        ).lower() == "true"
        # Note: ml_model_path is already defined above in ML settings section
        
        # Self-healing settings
        self.self_healing_enabled = os.getenv(
            "SELF_HEALING_ENABLED", "false"
        ).lower() == "true"
        self.min_scale_factor = float(os.getenv("MIN_SCALE_FACTOR", "0.1"))
        self.self_healing_dry_run = os.getenv(
            "SELF_HEALING_DRY_RUN", "false"
        ).lower() == "true"
        
        # Protected tags that prevent scaling
        protected_tags_str = os.getenv("SELF_HEALING_PROTECTED_TAGS", "production,critical")
        self.self_healing_protected_tags = [t.strip() for t in protected_tags_str.split(",") if t.strip()]
        
        # Scaling actions configuration (JSON string or defaults)
        import json
        scaling_actions_str = os.getenv("SCALING_ACTIONS", "")
        if scaling_actions_str:
            try:
                self.scaling_actions = json.loads(scaling_actions_str)
            except (json.JSONDecodeError, ValueError):
                # Use defaults if JSON parsing fails
                self.scaling_actions = {
                    'AWS EC2': 'shutdown',
                    'AWS S3': 'delete',
                    'Azure VM': 'stop',
                    'GCP Compute Engine': 'stop'
                }
        else:
            # Default scaling actions
            self.scaling_actions = {
                'AWS EC2': 'shutdown',
                'AWS S3': 'delete',
                'Azure VM': 'stop',
                'GCP Compute Engine': 'stop'
            }
        
        # API settings
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_debug = os.getenv("API_DEBUG", "false").lower() == "true"
        self.api_rate_limit = os.getenv("API_RATE_LIMIT", "1 per minute")
        self.api_enable_rate_limit = os.getenv(
            "API_ENABLE_RATE_LIMIT", "true"
        ).lower() == "true"
        self.api_key = os.getenv("API_KEY", "")
        self.api_key_enabled = os.getenv(
            "API_KEY_ENABLED", "false"
        ).lower() == "true"
        
        # Performance threshold settings
        self.max_collection_latency_seconds = float(
            os.getenv("MAX_COLLECTION_LATENCY_SECONDS", "5.0")
        )  # Maximum time for metric collection (default: 5.0 seconds)
        self.max_storage_latency_seconds = float(
            os.getenv("MAX_STORAGE_LATENCY_SECONDS", "10.0")
        )  # Maximum time for database storage (default: 10.0 seconds)
        self.max_processing_latency_per_metric_ms = float(
            os.getenv("MAX_PROCESSING_LATENCY_PER_METRIC_MS", "100.0")
        )  # Maximum processing time per metric in milliseconds (default: 100.0 ms)
        self.min_throughput_metrics_per_second = float(
            os.getenv("MIN_THROUGHPUT_METRICS_PER_SECOND", "100.0")
        )  # Minimum throughput requirement (default: 100.0 metrics/second)
        
        # Alert performance threshold settings
        self.max_alert_generation_latency_seconds = float(
            os.getenv("MAX_ALERT_GENERATION_LATENCY_SECONDS", "1.0")
        )  # Maximum time for alert generation (default: 1.0 seconds)
        self.max_alert_logging_latency_ms = float(
            os.getenv("MAX_ALERT_LOGGING_LATENCY_MS", "100.0")
        )  # Maximum time for database logging per alert (default: 100.0 ms)
        self.min_alert_throughput_per_second = float(
            os.getenv("MIN_ALERT_THROUGHPUT_PER_SECOND", "50.0")
        )  # Minimum alerts processed per second (default: 50.0 alerts/second)
        
        # Database performance threshold settings
        self.max_database_insertion_latency_ms = float(
            os.getenv("MAX_DATABASE_INSERTION_LATENCY_MS", "100.0")
        )  # Maximum time for single alert insertion (default: 100.0 ms)
        self.max_database_retrieval_latency_ms = float(
            os.getenv("MAX_DATABASE_RETRIEVAL_LATENCY_MS", "500.0")
        )  # Maximum time for alert retrieval query (default: 500.0 ms)
        self.min_database_throughput_alerts_per_second = float(
            os.getenv("MIN_DATABASE_THROUGHPUT_ALERTS_PER_SECOND", "50.0")
        )  # Minimum alerts inserted per second (default: 50.0)
        self.max_concurrent_operations_latency_seconds = float(
            os.getenv("MAX_CONCURRENT_OPERATIONS_LATENCY_SECONDS", "30.0")
        )  # Maximum time for concurrent operations (default: 30.0 seconds)
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration."""
        return {
            "url": self.database_url
        }
    
    def get_cloud_config(self, provider: str) -> Dict[str, str]:
        """
        Get cloud provider configuration.
        
        Args:
            provider: Cloud provider name ('aws', 'azure', 'gcp')
        
        Returns:
            Dictionary containing provider-specific configuration
        """
        configs = {
            "aws": {
                "access_key": self.aws_access_key,
                "secret_key": self.aws_secret_key,
                "region": self.aws_region
            },
            "azure": {
                "subscription_id": self.azure_subscription_id,
                "client_id": self.azure_client_id,
                "client_secret": self.azure_client_secret,
                "tenant_id": self.azure_tenant_id
            },
            "gcp": {
                "project_id": self.gcp_project_id,
                "credentials_path": self.gcp_credentials_path
            }
        }
        return configs.get(provider, {})
    
    def get_alert_channel_config(self, channel: str) -> Dict[str, Any]:
        """
        Get alert channel configuration.
        
        Args:
            channel: Channel name ('email', 'slack', 'sms')
        
        Returns:
            Dictionary containing channel-specific configuration
        """
        configs = {
            "email": {
                "smtp_server": self.email_smtp_server,
                "smtp_port": self.email_smtp_port,
                "username": self.email_username,
                "password": self.email_password,
                "recipients": self.email_recipients
            },
            "slack": {
                "webhook_url": self.slack_webhook_url
            },
            "sms": {
                "api_key": self.sms_api_key,
                "api_secret": self.sms_api_secret,
                "from_number": self.sms_from_number
            }
        }
        return configs.get(channel, {})
    
    def get_thresholds(self) -> Dict[str, Any]:
        """
        Get all threshold values.
        
        Returns:
            Dictionary containing all threshold values
        """
        return {
            "cpu_usage_threshold": self.cpu_usage_threshold,
            "cpu_threshold_duration": self.cpu_threshold_duration,
            "cloud_cost_threshold": self.cloud_cost_threshold,
            "memory_usage_threshold": self.memory_usage_threshold,
            "disk_usage_threshold": self.disk_usage_threshold,
            "network_bandwidth_threshold": self.network_bandwidth_threshold
        }
    
    def update_thresholds(self, **kwargs) -> Dict[str, Any]:
        """
        Update threshold values dynamically.
        
        Args:
            **kwargs: Threshold values to update (cpu_usage_threshold, cpu_threshold_duration, etc.)
        
        Returns:
            Dictionary of updated thresholds
        
        Raises:
            ValueError: If threshold values are invalid
        """
        # Validate and update each threshold
        if 'cpu_usage_threshold' in kwargs:
            val = float(kwargs['cpu_usage_threshold'])
            if not 0 <= val <= 100:
                raise ValueError("CPU usage threshold must be between 0 and 100")
            self.cpu_usage_threshold = val
        
        if 'cpu_threshold_duration' in kwargs:
            val = int(kwargs['cpu_threshold_duration'])
            if val < 0:
                raise ValueError("CPU threshold duration must be non-negative")
            self.cpu_threshold_duration = val
        
        if 'cloud_cost_threshold' in kwargs:
            val = float(kwargs['cloud_cost_threshold'])
            if val <= 0:
                raise ValueError("Cloud cost threshold must be positive")
            self.cloud_cost_threshold = val
        
        if 'memory_usage_threshold' in kwargs:
            val = float(kwargs['memory_usage_threshold'])
            if not 0 <= val <= 100:
                raise ValueError("Memory usage threshold must be between 0 and 100")
            self.memory_usage_threshold = val
        
        if 'disk_usage_threshold' in kwargs:
            val = float(kwargs['disk_usage_threshold'])
            if not 0 <= val <= 100:
                raise ValueError("Disk usage threshold must be between 0 and 100")
            self.disk_usage_threshold = val
        
        if 'network_bandwidth_threshold' in kwargs:
            val = kwargs['network_bandwidth_threshold']
            if val is None:
                self.network_bandwidth_threshold = None
            else:
                val = float(val)
                if val <= 0:
                    raise ValueError("Network bandwidth threshold must be positive")
                self.network_bandwidth_threshold = val
        
        return self.get_thresholds()
    
    def _parse_alert_preferences(self) -> Dict[str, Any]:
        """
        Parse alert preferences from environment variable or use defaults.
        
        Returns:
            Dictionary containing alert preference rules
        """
        import json
        
        alert_prefs_str = os.getenv("ALERT_PREFERENCES", "")
        
        if alert_prefs_str:
            try:
                prefs = json.loads(alert_prefs_str)
                # Validate structure
                if isinstance(prefs, dict) and "rules" in prefs:
                    return prefs
                else:
                    # Invalid structure, use defaults
                    return self._get_default_alert_preferences()
            except (json.JSONDecodeError, ValueError):
                # Invalid JSON, use defaults
                return self._get_default_alert_preferences()
        else:
            return self._get_default_alert_preferences()
    
    def _get_default_alert_preferences(self) -> Dict[str, Any]:
        """
        Get default alert preferences.
        
        Returns:
            Dictionary with default alert preference rules
        """
        return {
            'rules': [
                {
                    'condition': {'metric_type': 'cost', 'operator': '>', 'value': 500},
                    'channels': ['sms', 'email'],
                    'severity': 'critical'
                },
                {
                    'condition': {'metric_type': 'cost', 'operator': '>', 'value': 200},
                    'channels': ['email'],
                    'severity': 'high'
                },
                {
                    'condition': {'metric_type': 'usage', 'operator': '>', 'value': 80},
                    'channels': ['slack'],
                    'severity': 'high'
                },
                {
                    'condition': {'metric_type': 'usage', 'operator': '>', 'value': 60},
                    'channels': ['email'],
                    'severity': 'medium'
                }
            ],
            'default_channels': ['email']
        }
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        
        TODO: Implement configuration validation
        """
        # TODO: Validate required settings
        # - Database URL format
        # - Cloud provider credentials if enabled
        # - Alert channel configuration if enabled
        return True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    return settings
