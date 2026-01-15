"""
Database interactions for Monitor/Drift Agent.

This module provides functionality to interact with PostgreSQL database
for storing and retrieving metrics, alerts, and policy data.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
from urllib.parse import urlparse

# Try to import psycopg2, handle gracefully if not available
try:
    import psycopg2
    from psycopg2 import OperationalError, Error
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    OperationalError = Exception
    Error = Exception

# Import settings for database configuration
try:
    from config.settings import get_settings
except ImportError:
    # Fallback if config module is not available
    def get_settings():
        return None


class DatabaseConnection:
    """Manages database connection and operations."""
    
    def __init__(self, connection_string: str):
        """
        Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string
        
        TODO: Implement database connection initialization
        """
        self.connection_string = connection_string
        self.connection = None
        # TODO: Initialize database connection using psycopg2 or SQLAlchemy
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful, False otherwise
        
        TODO: Implement connection establishment
        """
        # TODO: Implement database connection
        return False
    
    def disconnect(self):
        """Close database connection."""
        # TODO: Implement connection cleanup
        if self.connection:
            # TODO: Close connection
            self.connection = None
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Optional query parameters
        
        Returns:
            List of result dictionaries
        
        TODO: Implement query execution
        """
        # TODO: Implement query execution with parameter binding
        return []
    
    def insert_metrics(self, metrics: Dict) -> bool:
        """
        Insert metrics into database.
        
        Args:
            metrics: Dictionary containing metrics data
        
        Returns:
            True if insertion successful, False otherwise
        
        TODO: Implement metrics insertion
        """
        # TODO: Implement metrics insertion into metrics table
        return False
    
    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metric_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve metrics from database within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            metric_type: Optional filter for specific metric type
        
        Returns:
            List of metric dictionaries
        
        TODO: Implement metrics retrieval query
        """
        # TODO: Implement metrics retrieval with time range filtering
        return []
    
    def insert_alert(self, alert: Dict) -> str:
        """
        Insert an alert into database.
        
        Args:
            alert: Dictionary containing alert data
        
        Returns:
            Alert ID of inserted alert
        
        TODO: Implement alert insertion
        """
        # TODO: Implement alert insertion into alerts table
        return ""
    
    def get_alert(self, alert_id: str) -> Optional[Dict]:
        """
        Retrieve an alert by ID.
        
        Args:
            alert_id: Unique identifier for the alert
        
        Returns:
            Alert dictionary or None if not found
        
        TODO: Implement alert retrieval
        """
        # TODO: Implement alert retrieval query
        return None
    
    def list_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        List alerts with optional filtering.
        
        Args:
            status: Optional filter by alert status
            severity: Optional filter by severity level
            limit: Maximum number of alerts to return
        
        Returns:
            List of alert dictionaries
        
        TODO: Implement alert listing with filters
        """
        # TODO: Implement alert listing query with filters
        return []
    
    def insert_policy(self, policy: Dict) -> str:
        """
        Insert a policy into database.
        
        Args:
            policy: Dictionary containing policy data
        
        Returns:
            Policy ID of inserted policy
        
        TODO: Implement policy insertion
        """
        # TODO: Implement policy insertion into policies table
        return ""
    
    def get_policy(self, policy_id: str) -> Optional[Dict]:
        """
        Retrieve a policy by ID.
        
        Args:
            policy_id: Unique identifier for the policy
        
        Returns:
            Policy dictionary or None if not found
        
        TODO: Implement policy retrieval
        """
        # TODO: Implement policy retrieval query
        return None
    
    def list_policies(self, enabled_only: bool = False) -> List[Dict]:
        """
        List all policies.
        
        Args:
            enabled_only: If True, only return enabled policies
        
        Returns:
            List of policy dictionaries
        
        TODO: Implement policy listing
        """
        # TODO: Implement policy listing query
        return []
    
    def update_policy(self, policy_id: str, updates: Dict) -> bool:
        """
        Update an existing policy.
        
        Args:
            policy_id: Unique identifier for the policy
            updates: Dictionary containing fields to update
        
        Returns:
            True if update successful, False otherwise
        
        TODO: Implement policy update
        """
        # TODO: Implement policy update query
        return False
    
    def delete_policy(self, policy_id: str) -> bool:
        """
        Delete a policy.
        
        Args:
            policy_id: Unique identifier for the policy
        
        Returns:
            True if deletion successful, False otherwise
        
        TODO: Implement policy deletion
        """
        # TODO: Implement policy deletion query
        return False


def connect_to_db():
    """
    Establish a connection to the PostgreSQL database.
    
    Uses database credentials from environment variables via config.settings.
    Connection string format: postgresql://user:password@host:port/dbname
    
    Returns:
        psycopg2 connection object if successful, None otherwise
    
    Raises:
        RuntimeError: If psycopg2 is not available
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError(
            "psycopg2 library is not available. Please install it using: pip install psycopg2-binary"
        )
    
    try:
        # Get database URL from settings
        settings = get_settings()
        if settings is None:
            raise RuntimeError("Settings module is not available")
        
        database_url = settings.database_url
        
        # Parse the connection string
        # Format: postgresql://user:password@host:port/dbname
        parsed = urlparse(database_url)
        
        # Extract connection parameters
        dbname = parsed.path.lstrip('/') if parsed.path else None
        user = parsed.username
        password = parsed.password
        host = parsed.hostname or 'localhost'
        port = parsed.port or 5432
        
        if not dbname or not user or not password:
            raise ValueError(
                "Invalid database URL format. Expected: postgresql://user:password@host:port/dbname"
            )
        
        # Establish connection
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        
        return conn
    
    except OperationalError as e:
        print(f"Error connecting to database: {e}")
        return None
    except (ValueError, AttributeError) as e:
        print(f"Error parsing database URL: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error connecting to database: {e}")
        return None


def create_cloud_metrics_schema(conn) -> bool:
    """
    Create the database schema for storing cloud metrics.
    
    Creates a table called cloud_metrics with columns for provider,
    metric_name, metric_value, timestamp, and resource_type.
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create cloud_metrics table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cloud_metrics (
            id SERIAL PRIMARY KEY,
            provider TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value NUMERIC,
            timestamp TIMESTAMP,
            resource_type TEXT NOT NULL
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating cloud metrics schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating cloud metrics schema: {e}")
        if conn:
            conn.rollback()
        return False


def create_alerts_schema(conn) -> bool:
    """
    Create the database schema for storing alerts.
    
    Creates a table called alerts with columns for metric_name, value,
    timestamp, resource_type, severity, and action_taken.
    Also handles migration for existing tables by adding missing columns.
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create table with new schema (if table doesn't exist)
        create_table_query = """
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            metric_name TEXT NOT NULL,
            value NUMERIC NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resource_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'medium',
            action_taken TEXT NOT NULL DEFAULT 'Alert Triggered'
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        
        # Migration: Add columns if they don't exist (for existing tables)
        migration_query = """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='alerts' AND column_name='severity'
            ) THEN
                ALTER TABLE alerts ADD COLUMN severity TEXT NOT NULL DEFAULT 'medium';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='alerts' AND column_name='action_taken'
            ) THEN
                ALTER TABLE alerts ADD COLUMN action_taken TEXT NOT NULL DEFAULT 'Alert Triggered';
            END IF;
        END $$;
        """
        
        cursor.execute(migration_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating alerts schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating alerts schema: {e}")
        if conn:
            conn.rollback()
        return False


def create_self_healing_log_schema(conn) -> bool:
    """
    Create the database schema for storing self-healing action logs.
    
    Creates a table called self_healing_log with columns for resource_name,
    action_taken, timestamp, cost_at_action, status, and error_message.
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS self_healing_log (
            id SERIAL PRIMARY KEY,
            resource_name TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cost_at_action NUMERIC,
            status TEXT,
            error_message TEXT
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating self_healing_log schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating self_healing_log schema: {e}")
        if conn:
            conn.rollback()
        return False


def create_cloud_resource_costs_schema(conn) -> bool:
    """
    Create the database schema for storing cloud resource cost data.
    
    Creates a table called cloud_resource_costs with columns:
    - id: SERIAL PRIMARY KEY
    - resource_name: TEXT NOT NULL (e.g., 'AWS EC2', 'S3')
    - resource_usage: NUMERIC NOT NULL (usage amount)
    - cost: NUMERIC NOT NULL (calculated cost)
    - timestamp: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create cloud_resource_costs table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cloud_resource_costs (
            id SERIAL PRIMARY KEY,
            resource_name TEXT NOT NULL,
            resource_usage NUMERIC NOT NULL,
            cost NUMERIC NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating cloud_resource_costs schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating cloud_resource_costs schema: {e}")
        if conn:
            conn.rollback()
        return False


def create_model_performance_schema(conn) -> bool:
    """
    Create the database schema for storing ML model performance metrics.
    
    Creates a table called model_performance with columns for tracking
    model performance metrics such as accuracy, detection rate, and alert rate.
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create model_performance table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_version TEXT,
            total_predictions INTEGER NOT NULL,
            anomalies_detected INTEGER NOT NULL,
            alerts_triggered INTEGER NOT NULL,
            avg_anomaly_score NUMERIC,
            detection_rate NUMERIC NOT NULL,
            alert_rate NUMERIC,
            false_positive_rate NUMERIC,
            evaluation_period_start TIMESTAMP,
            evaluation_period_end TIMESTAMP,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating model_performance schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating model_performance schema: {e}")
        if conn:
            conn.rollback()
        return False


def create_resource_policies_schema(conn) -> bool:
    """
    Create the database schema for storing resource policies.
    
    Creates a table called resource_policies with columns for resource name,
    threshold value, threshold type, duration, and enabled status.
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create resource_policies table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS resource_policies (
            id SERIAL PRIMARY KEY,
            resource_name TEXT NOT NULL,
            threshold_value NUMERIC NOT NULL,
            threshold_type TEXT NOT NULL,
            duration INTEGER NOT NULL,
            enabled BOOLEAN DEFAULT TRUE,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating resource_policies schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating resource_policies schema: {e}")
        if conn:
            conn.rollback()
        return False


def create_unified_metrics_schema(conn) -> bool:
    """
    Create the database schema for storing unified cloud and server metrics.
    
    Creates a table called cloud_server_metrics with columns for both
    server metrics (CPU, memory, network) and cloud metrics (compute,
    storage, bandwidth) in a single row.
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create cloud_server_metrics table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cloud_server_metrics (
            id SERIAL PRIMARY KEY,
            cpu_usage NUMERIC NOT NULL,
            memory_usage NUMERIC NOT NULL,
            network_traffic NUMERIC NOT NULL,
            cloud_compute NUMERIC,
            cloud_storage NUMERIC,
            cloud_bandwidth NUMERIC,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resource_type TEXT NOT NULL
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating unified metrics schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating unified metrics schema: {e}")
        if conn:
            conn.rollback()
        return False


def create_schema(conn) -> bool:
    """
    Define and create the database schema for storing system metrics.
    
    Creates a table called system_metrics with columns for CPU, memory,
    network metrics, timestamp, and resource type.
    Also creates the cloud_metrics table.
    
    Args:
        conn: psycopg2 connection object
    
    Returns:
        True if schema creation successful, False otherwise
    """
    if conn is None:
        print("Error: Database connection is None")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create system_metrics table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS system_metrics (
            id SERIAL PRIMARY KEY,
            cpu_usage NUMERIC,
            memory_usage NUMERIC,
            network_traffic NUMERIC,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resource_type TEXT NOT NULL
        );
        """
        
        cursor.execute(create_table_query)
        
        # Also create cloud_metrics table
        create_cloud_table_query = """
        CREATE TABLE IF NOT EXISTS cloud_metrics (
            id SERIAL PRIMARY KEY,
            provider TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value NUMERIC,
            timestamp TIMESTAMP,
            resource_type TEXT NOT NULL
        );
        """
        
        cursor.execute(create_cloud_table_query)
        
        # Also create unified cloud_server_metrics table
        create_unified_table_query = """
        CREATE TABLE IF NOT EXISTS cloud_server_metrics (
            id SERIAL PRIMARY KEY,
            cpu_usage NUMERIC NOT NULL,
            memory_usage NUMERIC NOT NULL,
            network_traffic NUMERIC NOT NULL,
            cloud_compute NUMERIC,
            cloud_storage NUMERIC,
            cloud_bandwidth NUMERIC,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resource_type TEXT NOT NULL
        );
        """
        
        cursor.execute(create_unified_table_query)
        
        # Also create alerts table
        create_alerts_table_query = """
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            metric_name TEXT NOT NULL,
            value NUMERIC NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resource_type TEXT NOT NULL
        );
        """
        
        cursor.execute(create_alerts_table_query)
        
        # Also create cloud_resource_costs table
        create_cloud_resource_costs_table_query = """
        CREATE TABLE IF NOT EXISTS cloud_resource_costs (
            id SERIAL PRIMARY KEY,
            resource_name TEXT NOT NULL,
            resource_usage NUMERIC NOT NULL,
            cost NUMERIC NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_cloud_resource_costs_table_query)
        
        # Also create model_performance table
        create_model_performance_table_query = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_version TEXT,
            total_predictions INTEGER NOT NULL,
            anomalies_detected INTEGER NOT NULL,
            alerts_triggered INTEGER NOT NULL,
            avg_anomaly_score NUMERIC,
            detection_rate NUMERIC NOT NULL,
            alert_rate NUMERIC,
            false_positive_rate NUMERIC,
            evaluation_period_start TIMESTAMP,
            evaluation_period_end TIMESTAMP,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_model_performance_table_query)
        
        # Also create resource_policies table
        create_resource_policies_table_query = """
        CREATE TABLE IF NOT EXISTS resource_policies (
            id SERIAL PRIMARY KEY,
            resource_name TEXT NOT NULL,
            threshold_value NUMERIC NOT NULL,
            threshold_type TEXT NOT NULL,
            duration INTEGER NOT NULL,
            enabled BOOLEAN DEFAULT TRUE,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_resource_policies_table_query)
        
        # Also create self_healing_log table
        create_self_healing_log_table_query = """
        CREATE TABLE IF NOT EXISTS self_healing_log (
            id SERIAL PRIMARY KEY,
            resource_name TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cost_at_action NUMERIC,
            status TEXT,
            error_message TEXT
        );
        """
        
        cursor.execute(create_self_healing_log_table_query)
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error creating database schema: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error creating schema: {e}")
        if conn:
            conn.rollback()
        return False


def store_metrics(cpu_data: Dict, memory_data: Dict, network_data: Dict) -> bool:
    """
    Store system metrics (CPU, memory, network) in the PostgreSQL database.
    
    Each metric type is stored as a separate row with the appropriate
    resource_type. The function uses a single transaction to ensure
    all three inserts succeed or fail together.
    
    Args:
        cpu_data: Dictionary from get_cpu_usage() containing cpu_percent and timestamp
        memory_data: Dictionary from get_memory_usage() containing percent/used and timestamp
        network_data: Dictionary from get_network_traffic() containing bytes_sent/bytes_recv and timestamp
    
    Returns:
        True if all metrics stored successfully, False otherwise
    """
    conn = None
    try:
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            print("Failed to establish database connection")
            return False
        
        cursor = conn.cursor()
        
        # Extract values from metric dictionaries
        # CPU metric
        cpu_usage = cpu_data.get('cpu_percent', 0.0)
        cpu_timestamp = cpu_data.get('timestamp')
        if isinstance(cpu_timestamp, datetime):
            cpu_timestamp = cpu_timestamp
            # If timezone-aware, convert to UTC naive
            if cpu_timestamp.tzinfo is not None:
                cpu_timestamp = cpu_timestamp.astimezone().replace(tzinfo=None)
        else:
            cpu_timestamp = datetime.utcnow()
        
        # Memory metric - use percent if available, otherwise use used bytes
        memory_usage = memory_data.get('percent')
        if memory_usage is None:
            memory_usage = memory_data.get('used', 0)
        memory_timestamp = memory_data.get('timestamp')
        if isinstance(memory_timestamp, datetime):
            memory_timestamp = memory_timestamp
            # If timezone-aware, convert to UTC naive
            if memory_timestamp.tzinfo is not None:
                memory_timestamp = memory_timestamp.astimezone().replace(tzinfo=None)
        else:
            memory_timestamp = datetime.utcnow()
        
        # Network metric - sum of bytes sent and received
        bytes_sent = network_data.get('bytes_sent', 0)
        bytes_recv = network_data.get('bytes_recv', 0)
        network_traffic = bytes_sent + bytes_recv
        network_timestamp = network_data.get('timestamp')
        if isinstance(network_timestamp, datetime):
            network_timestamp = network_timestamp
            # If timezone-aware, convert to UTC naive
            if network_timestamp.tzinfo is not None:
                network_timestamp = network_timestamp.astimezone().replace(tzinfo=None)
        else:
            network_timestamp = datetime.utcnow()
        
        # Use the earliest timestamp from all metrics (or current UTC time)
        # This ensures all three rows have the same timestamp
        timestamps = [cpu_timestamp, memory_timestamp, network_timestamp]
        metric_timestamp = min(timestamps) if timestamps else datetime.utcnow()
        
        # Insert query template
        insert_query = """
            INSERT INTO system_metrics (cpu_usage, memory_usage, network_traffic, timestamp, resource_type)
            VALUES (%s, %s, %s, %s, %s);
        """
        
        # Insert CPU metric (check for duplicate first)
        if check_duplicate_metrics(metric_timestamp, 'CPU', 'system_metrics'):
            print(f"Duplicate CPU metric detected for timestamp {metric_timestamp}. Skipping insert.")
        else:
            cursor.execute(
                insert_query,
                (cpu_usage, None, None, metric_timestamp, 'CPU')
            )
        
        # Insert Memory metric (check for duplicate first)
        if check_duplicate_metrics(metric_timestamp, 'Memory', 'system_metrics'):
            print(f"Duplicate Memory metric detected for timestamp {metric_timestamp}. Skipping insert.")
        else:
            cursor.execute(
                insert_query,
                (None, memory_usage, None, metric_timestamp, 'Memory')
            )
        
        # Insert Network metric (check for duplicate first)
        if check_duplicate_metrics(metric_timestamp, 'Network', 'system_metrics'):
            print(f"Duplicate Network metric detected for timestamp {metric_timestamp}. Skipping insert.")
        else:
            cursor.execute(
                insert_query,
                (None, None, network_traffic, metric_timestamp, 'Network')
            )
        
        # Commit transaction only after all three inserts succeed
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error storing metrics in database: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error storing metrics: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()


def store_all_metrics(all_metrics: Dict) -> bool:
    """
    Store unified metrics (server + cloud) in the database.
    
    This function stores both server metrics and cloud metrics from the
    unified structure returned by collect_all_metrics(). Server metrics
    are stored in the system_metrics table, and cloud metrics are stored
    in the cloud_metrics table.
    
    Args:
        all_metrics: Dictionary from collect_all_metrics() containing:
            - timestamp: str (ISO format)
            - server_metrics: Dict (CPU and memory usage with timestamp)
            - cloud_metrics: Dict (AWS, GCP, Azure metrics by provider)
    
    Returns:
        True if storage successful (at least some metrics stored), False otherwise
    """
    conn = None
    storage_success = False
    
    try:
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            print("Failed to establish database connection")
            return False
        
        # Ensure schemas exist
        if not create_schema(conn):
            print("Failed to create/verify database schemas")
            return False
        
        cursor = conn.cursor()
        
        # Store server metrics if available
        server_metrics = all_metrics.get('server_metrics', {})
        if server_metrics and 'error' not in server_metrics:
            try:
                cpu_usage = server_metrics.get('cpu_usage')
                memory_usage = server_metrics.get('memory_usage')
                server_timestamp = server_metrics.get('timestamp')
                
                # Convert timestamp if needed (ensure UTC)
                if isinstance(server_timestamp, str):
                    try:
                        # Try ISO format parsing (Python 3.7+)
                        server_timestamp = datetime.fromisoformat(server_timestamp.replace('Z', '+00:00'))
                        # If timezone-aware, convert to UTC naive
                        if server_timestamp.tzinfo is not None:
                            server_timestamp = server_timestamp.astimezone().replace(tzinfo=None)
                    except (ValueError, AttributeError):
                        server_timestamp = datetime.utcnow()
                elif not isinstance(server_timestamp, datetime):
                    server_timestamp = datetime.utcnow()
                elif server_timestamp.tzinfo is not None:
                    # If timezone-aware, convert to UTC naive
                    server_timestamp = server_timestamp.astimezone().replace(tzinfo=None)
                
                # Insert server metrics into system_metrics table
                insert_query = """
                    INSERT INTO system_metrics (cpu_usage, memory_usage, network_traffic, timestamp, resource_type)
                    VALUES (%s, %s, %s, %s, %s);
                """
                
                # Insert CPU metric (check for duplicate first)
                if cpu_usage is not None:
                    if check_duplicate_metrics(server_timestamp, 'CPU', 'system_metrics'):
                        print(f"Duplicate CPU metric detected for timestamp {server_timestamp}. Skipping insert.")
                    else:
                        cursor.execute(
                            insert_query,
                            (float(cpu_usage), None, None, server_timestamp, 'CPU')
                        )
                        storage_success = True
                
                # Insert Memory metric (check for duplicate first)
                if memory_usage is not None:
                    if check_duplicate_metrics(server_timestamp, 'Memory', 'system_metrics'):
                        print(f"Duplicate Memory metric detected for timestamp {server_timestamp}. Skipping insert.")
                    else:
                        cursor.execute(
                            insert_query,
                            (None, float(memory_usage), None, server_timestamp, 'Memory')
                        )
                        storage_success = True
                
                storage_success = True
            except Exception as e:
                print(f"Error storing server metrics: {e}")
                # Continue with cloud metrics even if server metrics fail
        
        # Store cloud metrics if available
        cloud_metrics = all_metrics.get('cloud_metrics', {})
        if cloud_metrics and 'error' not in cloud_metrics:
            # Insert query for cloud metrics
            cloud_insert_query = """
                INSERT INTO cloud_metrics (provider, metric_name, metric_value, timestamp, resource_type)
                VALUES (%s, %s, %s, %s, %s);
            """
            
            # Process each provider's metrics
            for provider, metrics_list in cloud_metrics.items():
                if not isinstance(metrics_list, list):
                    continue
                
                try:
                    for metric in metrics_list:
                        if not isinstance(metric, dict):
                            continue
                        
                        provider_name = metric.get('provider', provider)
                        metric_name = metric.get('metric_name', '')
                        metric_value = metric.get('metric_value')
                        metric_timestamp = metric.get('timestamp')
                        resource_type = metric.get('resource_type', 'Unknown')
                        
                        # Convert timestamp if needed (ensure UTC)
                        if isinstance(metric_timestamp, str):
                            try:
                                # Try ISO format parsing (Python 3.7+)
                                metric_timestamp = datetime.fromisoformat(metric_timestamp.replace('Z', '+00:00'))
                                # If timezone-aware, convert to UTC naive
                                if metric_timestamp.tzinfo is not None:
                                    metric_timestamp = metric_timestamp.astimezone().replace(tzinfo=None)
                            except (ValueError, AttributeError):
                                metric_timestamp = datetime.utcnow()
                        elif not isinstance(metric_timestamp, datetime):
                            metric_timestamp = datetime.utcnow()
                        elif metric_timestamp.tzinfo is not None:
                            # If timezone-aware, convert to UTC naive
                            metric_timestamp = metric_timestamp.astimezone().replace(tzinfo=None)
                        
                        # Insert cloud metric
                        if metric_value is not None and metric_name:
                            cursor.execute(
                                cloud_insert_query,
                                (
                                    provider_name,
                                    metric_name,
                                    float(metric_value),
                                    metric_timestamp,
                                    resource_type
                                )
                            )
                
                except Exception as e:
                    print(f"Error storing {provider} cloud metrics: {e}")
                    # Continue with other providers even if one fails
        
        # Commit transaction only if at least some metrics were stored
        if storage_success:
            conn.commit()
        else:
            conn.rollback()
        
        cursor.close()
        return storage_success
    
    except Error as e:
        print(f"Error storing all metrics in database: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error storing all metrics: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()


def check_duplicate_metrics(
    timestamp: datetime,
    resource_type: str,
    table_name: str = 'cloud_server_metrics'
) -> bool:
    """
    Check if a metric with the same timestamp and resource_type already exists.
    
    This function queries the database to check if a record with the same
    timestamp and resource_type already exists, helping prevent duplicate entries.
    
    Args:
        timestamp: Timestamp to check for duplicates
        resource_type: Resource type to check for duplicates
        table_name: Name of the table to check (default: 'cloud_server_metrics')
    
    Returns:
        True if duplicate exists, False otherwise
    
    Raises:
        RuntimeError: If psycopg2 is not available
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError(
            "psycopg2 library is not available. Please install it using: pip install psycopg2-binary"
        )
    
    conn = None
    cursor = None
    
    try:
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            print("Failed to establish database connection for duplicate check")
            return False
        
        cursor = conn.cursor()
        
        # Query to check for duplicates
        # Use a small time window (e.g., 1 second) to account for slight timestamp differences
        # This ensures we catch duplicates even if timestamps differ by milliseconds
        query = """
            SELECT COUNT(*) 
            FROM {} 
            WHERE resource_type = %s 
            AND timestamp >= %s - INTERVAL '1 second'
            AND timestamp <= %s + INTERVAL '1 second'
        """.format(table_name)
        
        params = [resource_type, timestamp, timestamp]
        
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        
        cursor.close()
        
        # Return True if duplicate exists (count > 0)
        return count > 0
    
    except Error as e:
        print(f"Error checking for duplicate metrics: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error checking for duplicate metrics: {e}")
        return False
    finally:
        # Ensure cursor and connection are closed
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def check_duplicate_cloud_metric(
    timestamp: datetime,
    provider: str,
    metric_name: str,
    resource_type: str
) -> bool:
    """
    Check if a cloud metric with the same timestamp, provider, metric_name, and resource_type already exists.
    
    This function queries the cloud_metrics table to check if a record with the same
    timestamp, provider, metric_name, and resource_type already exists, helping prevent duplicate entries.
    
    Args:
        timestamp: Timestamp to check for duplicates
        provider: Cloud provider name (aws, gcp, azure)
        metric_name: Name of the metric
        resource_type: Resource type (Compute, Storage, Network)
    
    Returns:
        True if duplicate exists, False otherwise
    
    Raises:
        RuntimeError: If psycopg2 is not available
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError(
            "psycopg2 library is not available. Please install it using: pip install psycopg2-binary"
        )
    
    conn = None
    cursor = None
    
    try:
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            print("Failed to establish database connection for duplicate check")
            return False
        
        cursor = conn.cursor()
        
        # Query to check for duplicates in cloud_metrics table
        # Use a small time window (e.g., 1 second) to account for slight timestamp differences
        query = """
            SELECT COUNT(*) 
            FROM cloud_metrics 
            WHERE provider = %s 
            AND metric_name = %s 
            AND resource_type = %s 
            AND timestamp >= %s - INTERVAL '1 second'
            AND timestamp <= %s + INTERVAL '1 second'
        """
        
        params = [provider, metric_name, resource_type, timestamp, timestamp]
        
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        
        cursor.close()
        
        # Return True if duplicate exists (count > 0)
        return count > 0
    
    except Error as e:
        print(f"Error checking for duplicate cloud metric: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error checking for duplicate cloud metric: {e}")
        return False
    finally:
        # Ensure cursor and connection are closed
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def store_unified_metrics(
    cpu_usage: float,
    memory_usage: float,
    network_traffic: float,
    resource_type: str,
    cloud_compute: Optional[float] = None,
    cloud_storage: Optional[float] = None,
    cloud_bandwidth: Optional[float] = None,
    timestamp: Optional[datetime] = None
) -> bool:
    """
    Store unified metrics (server + cloud) in the cloud_server_metrics table.
    
    This function stores both server metrics and optional cloud metrics in
    a single row in the unified cloud_server_metrics table.
    
    Args:
        cpu_usage: Server CPU usage as percentage (required)
        memory_usage: Server memory usage as percentage or bytes (required)
        network_traffic: Server network bandwidth in bytes (required)
        resource_type: Resource type identifier (Server, AWS, GCP, Azure) (required)
        cloud_compute: Cloud compute usage (optional)
        cloud_storage: Cloud storage usage (optional)
        cloud_bandwidth: Cloud bandwidth usage (optional)
        timestamp: Timestamp when metrics were collected (optional, defaults to current time)
    
    Returns:
        True if storage successful, False otherwise
    """
    conn = None
    
    try:
        # Validate required parameters
        if cpu_usage is None or memory_usage is None or network_traffic is None:
            print("Error: cpu_usage, memory_usage, and network_traffic are required")
            return False
        
        if not resource_type:
            print("Error: resource_type is required")
            return False
        
        # Validate resource_type
        valid_resource_types = ['Server', 'AWS', 'GCP', 'Azure']
        if resource_type not in valid_resource_types:
            print(f"Error: resource_type must be one of {valid_resource_types}")
            return False
        
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            print("Failed to establish database connection")
            return False
        
        # Ensure schema exists
        if not create_unified_metrics_schema(conn):
            print("Failed to create/verify unified metrics schema")
            return False
        
        cursor = conn.cursor()
        
        # Use provided timestamp or default to current UTC time
        if timestamp is None:
            metric_timestamp = datetime.utcnow()
        elif isinstance(timestamp, str):
            try:
                metric_timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                # Ensure UTC if timezone-aware, or assume UTC if naive
                if metric_timestamp.tzinfo is None:
                    # Naive datetime, assume UTC
                    metric_timestamp = metric_timestamp.replace(tzinfo=None)
            except (ValueError, AttributeError):
                metric_timestamp = datetime.utcnow()
        elif isinstance(timestamp, datetime):
            metric_timestamp = timestamp
            # If timezone-aware, convert to UTC naive
            if metric_timestamp.tzinfo is not None:
                metric_timestamp = metric_timestamp.astimezone().replace(tzinfo=None)
        else:
            metric_timestamp = datetime.utcnow()
        
        # Check for duplicate metrics before inserting
        if check_duplicate_metrics(metric_timestamp, resource_type):
            print(f"Duplicate entry detected for timestamp {metric_timestamp} and resource_type {resource_type}. Metrics will not be stored.")
            cursor.close()
            conn.close()
            return False
        
        # Insert query
        insert_query = """
            INSERT INTO cloud_server_metrics (
                cpu_usage, memory_usage, network_traffic,
                cloud_compute, cloud_storage, cloud_bandwidth,
                timestamp, resource_type
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        # Convert to appropriate types
        cpu_usage_float = float(cpu_usage)
        memory_usage_float = float(memory_usage)
        network_traffic_float = float(network_traffic)
        
        # Execute insert
        cursor.execute(
            insert_query,
            (
                cpu_usage_float,
                memory_usage_float,
                network_traffic_float,
                float(cloud_compute) if cloud_compute is not None else None,
                float(cloud_storage) if cloud_storage is not None else None,
                float(cloud_bandwidth) if cloud_bandwidth is not None else None,
                metric_timestamp,
                resource_type
            )
        )
        
        conn.commit()
        cursor.close()
        
        return True
    
    except (Error, ValueError, TypeError) as e:
        print(f"Error storing unified metrics in database: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error storing unified metrics: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()


def store_all_metrics_unified(all_metrics: Dict) -> bool:
    """
    Store unified metrics from collect_all_metrics() structure into cloud_server_metrics table.
    
    This function aggregates server and cloud metrics and stores them in the
    unified cloud_server_metrics table, creating one row per resource type
    (Server, AWS, GCP, Azure).
    
    Args:
        all_metrics: Dictionary from collect_all_metrics() containing:
            - timestamp: str (ISO format)
            - server_metrics: Dict (CPU and memory usage with timestamp)
            - cloud_metrics: Dict (AWS, GCP, Azure metrics by provider)
    
    Returns:
        True if storage successful (at least some metrics stored), False otherwise
    """
    storage_success = False
    
    try:
        # Extract server metrics
        server_metrics = all_metrics.get('server_metrics', {})
        if not server_metrics or 'error' in server_metrics:
            print("Server metrics not available or contain errors")
            return False
        
        # Get server metric values
        cpu_usage = server_metrics.get('cpu_usage')
        memory_usage = server_metrics.get('memory_usage')
        server_timestamp = server_metrics.get('timestamp')
        
        # Convert server timestamp if needed (ensure UTC)
        if isinstance(server_timestamp, str):
            try:
                server_timestamp = datetime.fromisoformat(server_timestamp.replace('Z', '+00:00'))
                # If timezone-aware, convert to UTC naive
                if server_timestamp.tzinfo is not None:
                    server_timestamp = server_timestamp.astimezone().replace(tzinfo=None)
            except (ValueError, AttributeError):
                server_timestamp = datetime.utcnow()
        elif not isinstance(server_timestamp, datetime):
            server_timestamp = datetime.utcnow()
        elif server_timestamp.tzinfo is not None:
            # If timezone-aware, convert to UTC naive
            server_timestamp = server_timestamp.astimezone().replace(tzinfo=None)
        
        # For network_traffic, we need to get it from server metrics or use 0
        # Since get_server_metrics() doesn't include network, we'll use 0 or fetch separately
        network_traffic = 0.0  # Default to 0, could be enhanced to get from network metrics
        
        # Store Server row (server metrics only)
        if cpu_usage is not None and memory_usage is not None:
            success = store_unified_metrics(
                cpu_usage=float(cpu_usage),
                memory_usage=float(memory_usage),
                network_traffic=network_traffic,
                resource_type='Server',
                timestamp=server_timestamp
            )
            if success:
                storage_success = True
        
        # Process cloud metrics by provider
        cloud_metrics = all_metrics.get('cloud_metrics', {})
        if cloud_metrics and 'error' not in cloud_metrics:
            # Aggregate metrics for each provider
            for provider, metrics_list in cloud_metrics.items():
                if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                    continue
                
                try:
                    # Initialize aggregation variables
                    cloud_compute_sum = 0.0
                    cloud_storage_sum = 0.0
                    cloud_bandwidth_sum = 0.0
                    provider_timestamp = server_timestamp  # Default to server timestamp
                    
                    # Aggregate cloud metrics by resource type
                    for metric in metrics_list:
                        if not isinstance(metric, dict):
                            continue
                        
                        resource_type = metric.get('resource_type', '').lower()
                        metric_name = metric.get('metric_name', '').lower()
                        metric_value = metric.get('metric_value', 0.0)
                        
                        # Get timestamp from first metric (or use server timestamp)
                        metric_timestamp = metric.get('timestamp')
                        if metric_timestamp and provider_timestamp == server_timestamp:
                            if isinstance(metric_timestamp, str):
                                try:
                                    provider_timestamp = datetime.fromisoformat(metric_timestamp.replace('Z', '+00:00'))
                                    # If timezone-aware, convert to UTC naive
                                    if provider_timestamp.tzinfo is not None:
                                        provider_timestamp = provider_timestamp.astimezone().replace(tzinfo=None)
                                except (ValueError, AttributeError):
                                    pass
                            elif isinstance(metric_timestamp, datetime):
                                provider_timestamp = metric_timestamp
                                # If timezone-aware, convert to UTC naive
                                if provider_timestamp.tzinfo is not None:
                                    provider_timestamp = provider_timestamp.astimezone().replace(tzinfo=None)
                        
                        # Aggregate by resource type
                        if resource_type == 'compute' or 'cpu' in metric_name:
                            try:
                                cloud_compute_sum += float(metric_value)
                            except (ValueError, TypeError):
                                pass
                        elif resource_type == 'storage' or 'storage' in metric_name or 'bucket' in metric_name or 'size' in metric_name:
                            try:
                                cloud_storage_sum += float(metric_value)
                            except (ValueError, TypeError):
                                pass
                        elif resource_type == 'network' or 'network' in metric_name or 'bandwidth' in metric_name or 'egress' in metric_name:
                            try:
                                cloud_bandwidth_sum += float(metric_value)
                            except (ValueError, TypeError):
                                pass
                    
                    # Map provider name to resource_type
                    provider_map = {
                        'aws': 'AWS',
                        'gcp': 'GCP',
                        'azure': 'Azure'
                    }
                    resource_type_name = provider_map.get(provider.lower(), provider.upper())
                    
                    # Store cloud provider row (server metrics + cloud metrics)
                    if cpu_usage is not None and memory_usage is not None:
                        success = store_unified_metrics(
                            cpu_usage=float(cpu_usage),
                            memory_usage=float(memory_usage),
                            network_traffic=network_traffic,
                            resource_type=resource_type_name,
                            cloud_compute=cloud_compute_sum if cloud_compute_sum > 0 else None,
                            cloud_storage=cloud_storage_sum if cloud_storage_sum > 0 else None,
                            cloud_bandwidth=cloud_bandwidth_sum if cloud_bandwidth_sum > 0 else None,
                            timestamp=provider_timestamp
                        )
                        if success:
                            storage_success = True
                
                except Exception as e:
                    print(f"Error aggregating and storing {provider} metrics: {e}")
                    # Continue with other providers even if one fails
        
        return storage_success
    
    except Exception as e:
        print(f"Unexpected error in store_all_metrics_unified: {e}")
        return False


def get_metrics_from_db(
    timestamp: Optional[str] = None,
    resource_type: Optional[str] = None,
    metric_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Retrieve metrics from the cloud_server_metrics table with optional filtering.
    
    Args:
        timestamp: Optional ISO format timestamp string. Filters rows where timestamp >= timestamp
        resource_type: Optional resource type filter (Server, AWS, GCP, Azure)
        metric_type: Optional metric type filter (CPU, Memory, Network, Cloud Compute, Cloud Storage, Cloud Bandwidth)
        limit: Maximum number of records to return (default: 50)
        offset: Number of records to skip for pagination (default: 0)
    
    Returns:
        Dictionary containing:
            - metrics: List of metric dictionaries
            - total: Total number of records matching filters
            - limit: Current limit
            - offset: Current offset
    """
    conn = None
    cursor = None
    
    try:
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            return {
                "metrics": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "error": "Failed to establish database connection"
            }
        
        cursor = conn.cursor()
        
        # Build WHERE clause dynamically
        where_clauses = []
        params = []
        
        # Handle timestamp filter
        if timestamp:
            try:
                # Parse ISO format timestamp
                if isinstance(timestamp, str):
                    # Try to parse ISO format
                    timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp_dt = timestamp
                
                where_clauses.append("timestamp >= %s")
                params.append(timestamp_dt)
            except (ValueError, AttributeError) as e:
                print(f"Error parsing timestamp: {e}")
                # If timestamp parsing fails, skip the filter
        
        # Handle resource_type filter
        if resource_type:
            where_clauses.append("resource_type = %s")
            params.append(resource_type)
        
        # Handle metric_type filter
        metric_type_map = {
            "CPU": "cpu_usage",
            "Memory": "memory_usage",
            "Network": "network_traffic",
            "Cloud Compute": "cloud_compute",
            "Cloud Storage": "cloud_storage",
            "Cloud Bandwidth": "cloud_bandwidth"
        }
        
        if metric_type:
            column_name = metric_type_map.get(metric_type)
            if column_name:
                where_clauses.append(f"{column_name} IS NOT NULL")
            # If metric_type doesn't match, ignore it
        
        # Build WHERE clause string
        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)
        
        # Build query for total count
        count_query = f"SELECT COUNT(*) FROM cloud_server_metrics {where_clause}"
        
        # Execute count query
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Build query for data retrieval
        query = f"""
            SELECT 
                id, cpu_usage, memory_usage, network_traffic,
                cloud_compute, cloud_storage, cloud_bandwidth,
                timestamp, resource_type
            FROM cloud_server_metrics
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
        """
        
        # Add limit and offset to params
        params_with_pagination = params + [limit, offset]
        
        # Execute data query
        cursor.execute(query, params_with_pagination)
        
        # Fetch all results
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        metrics = []
        for row in rows:
            metric = {
                "id": row[0],
                "cpu_usage": float(row[1]) if row[1] is not None else None,
                "memory_usage": float(row[2]) if row[2] is not None else None,
                "network_traffic": float(row[3]) if row[3] is not None else None,
                "cloud_compute": float(row[4]) if row[4] is not None else None,
                "cloud_storage": float(row[5]) if row[5] is not None else None,
                "cloud_bandwidth": float(row[6]) if row[6] is not None else None,
                "timestamp": row[7].isoformat() if isinstance(row[7], datetime) else str(row[7]) if row[7] else None,
                "resource_type": row[8]
            }
            metrics.append(metric)
        
        return {
            "metrics": metrics,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    
    except Error as e:
        print(f"Error retrieving metrics from database: {e}")
        return {
            "metrics": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "error": f"Database error: {str(e)}"
        }
    except Exception as e:
        print(f"Unexpected error retrieving metrics: {e}")
        return {
            "metrics": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "error": f"Unexpected error: {str(e)}"
        }
    finally:
        # Ensure cursor and connection are closed
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def store_cloud_cost_data(
    resource_name: str,
    resource_usage: float,
    cost: float,
    timestamp: Optional[datetime] = None
) -> bool:
    """
    Store real-time cloud cost data in PostgreSQL.
    
    This function stores cloud resource cost data (resource name, usage amount,
    calculated cost, and timestamp) in the cloud_resource_costs table.
    
    Args:
        resource_name: Name of the cloud resource (e.g., 'AWS EC2', 'S3 Bucket')
        resource_usage: Amount of resource used (e.g., CPU hours, GB of storage)
        cost: Associated cost for that usage
        timestamp: Timestamp when data was recorded (defaults to current time if None)
    
    Returns:
        True if storage successful, False otherwise
    """
    if not PSYCOPG2_AVAILABLE:
        print("Error: psycopg2 library is not available")
        return False
    
    # Validate inputs
    if not resource_name or not isinstance(resource_name, str):
        print("Error: resource_name must be a non-empty string")
        return False
    
    if not isinstance(resource_usage, (int, float)) or resource_usage < 0:
        print(f"Error: resource_usage must be a non-negative number, got: {resource_usage}")
        return False
    
    if not isinstance(cost, (int, float)) or cost < 0:
        print(f"Error: cost must be a non-negative number, got: {cost}")
        return False
    
    # Use current time if timestamp not provided
    if timestamp is None:
        timestamp = datetime.utcnow()
    elif not isinstance(timestamp, datetime):
        print(f"Error: timestamp must be a datetime object, got: {type(timestamp)}")
        return False
    
    # Ensure timestamp is UTC naive (consistent with existing code)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone().replace(tzinfo=None)
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            print("Error: Failed to connect to database")
            return False
        
        cursor = conn.cursor()
        
        # Insert cloud cost data
        insert_query = """
            INSERT INTO cloud_resource_costs (resource_name, resource_usage, cost, timestamp)
            VALUES (%s, %s, %s, %s);
        """
        
        cursor.execute(
            insert_query,
            (resource_name, float(resource_usage), float(cost), timestamp)
        )
        
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error storing cloud cost data: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error storing cloud cost data: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def get_historical_cost_data(
    resource_name: str,
    days: int = 7
) -> List[float]:
    """
    Retrieve historical cost data for a specific resource from the database.
    
    This function queries the cloud_resource_costs table to retrieve cost values
    for a specific resource over a specified time period. The data is used for
    calculating historical averages and detecting cost spikes.
    
    Args:
        resource_name: Name of the cloud resource (e.g., 'AWS EC2', 'S3 Storage')
        days: Number of days to look back (default: 7)
    
    Returns:
        List of cost values (floats) from the specified time period, ordered by timestamp DESC.
        Returns empty list if no data found or on error.
    
    Raises:
        RuntimeError: If psycopg2 is not available
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError(
            "psycopg2 library is not available. Please install it using: pip install psycopg2-binary"
        )
    
    # Validate inputs
    if not resource_name or not isinstance(resource_name, str):
        print("Error: resource_name must be a non-empty string")
        return []
    
    if not isinstance(days, int) or days < 1:
        print(f"Error: days must be a positive integer, got: {days}")
        return []
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            print("Error: Failed to connect to database")
            return []
        
        cursor = conn.cursor()
        
        # Query historical cost data using parameterized query
        # Calculate cutoff timestamp in Python to avoid INTERVAL parameterization issues
        from datetime import timedelta
        cutoff_timestamp = datetime.utcnow() - timedelta(days=days)
        
        query = """
            SELECT cost FROM cloud_resource_costs
            WHERE resource_name = %s 
              AND timestamp >= %s
            ORDER BY timestamp DESC;
        """
        
        cursor.execute(query, (resource_name, cutoff_timestamp))
        results = cursor.fetchall()
        
        # Extract cost values from results
        cost_values = [float(row[0]) for row in results if row[0] is not None]
        
        cursor.close()
        
        return cost_values
    
    except Error as e:
        print(f"Error retrieving historical cost data: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error retrieving historical cost data: {e}")
        return []
    finally:
        if conn:
            conn.close()


def store_alert_in_db(
    metric_name: str,
    value: float,
    timestamp: datetime,
    resource_type: str,
    severity: str = "medium",
    action_taken: str = "Alert Triggered"
) -> bool:
    """
    Store an alert in the PostgreSQL database.
    
    This function stores alert information when an anomaly is detected
    (e.g., when a metric exceeds a threshold). The alert contains the
    metric name, value that exceeded the threshold, timestamp, resource type,
    severity level, and action taken.
    
    Args:
        metric_name: Name of the metric that triggered the alert (e.g., 'CPU Usage', 'Cloud Cost')
        value: Metric value that exceeded the threshold
        timestamp: Timestamp when the alert was triggered (datetime object)
        resource_type: Resource type identifier (Server, AWS, GCP, Azure)
        severity: Severity level of the alert (default: "medium")
        action_taken: Action that was taken after the alert (default: "Alert Triggered")
    
    Returns:
        True if storage successful, False otherwise
    
    Raises:
        RuntimeError: If psycopg2 is not available
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError(
            "psycopg2 library is not available. Please install it using: pip install psycopg2-binary"
        )
    
    conn = None
    try:
        # Establish database connection
        conn = connect_to_db()
        if conn is None:
            print("Failed to establish database connection")
            return False
        
        # Ensure schema exists
        if not create_alerts_schema(conn):
            print("Failed to create/verify alerts schema")
            return False
        
        # Ensure self_healing_log schema exists
        if not create_self_healing_log_schema(conn):
            print("Failed to create/verify self_healing_log schema")
            # Don't fail alert storage if self_healing_log schema missing
        
        cursor = conn.cursor()
        
        # Ensure timestamp is UTC and timezone-naive
        if isinstance(timestamp, datetime):
            alert_timestamp = timestamp
            if alert_timestamp.tzinfo is not None:
                alert_timestamp = alert_timestamp.astimezone().replace(tzinfo=None)
        elif isinstance(timestamp, str):
            try:
                alert_timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if alert_timestamp.tzinfo is not None:
                    alert_timestamp = alert_timestamp.astimezone().replace(tzinfo=None)
            except (ValueError, AttributeError):
                alert_timestamp = datetime.utcnow()
        else:
            alert_timestamp = datetime.utcnow()
        
        # Validate inputs
        if not metric_name or not isinstance(metric_name, str):
            print("Error: metric_name must be a non-empty string")
            cursor.close()
            conn.close()
            return False
        
        if not isinstance(value, (int, float)):
            print(f"Error: value must be numeric, got {type(value)}")
            cursor.close()
            conn.close()
            return False
        
        if not resource_type or not isinstance(resource_type, str):
            print("Error: resource_type must be a non-empty string")
            cursor.close()
            conn.close()
            return False
        
        # Validate severity
        if not severity or not isinstance(severity, str):
            severity = "medium"
        
        # Validate action_taken
        if not action_taken or not isinstance(action_taken, str):
            action_taken = "Alert Triggered"
        
        # Insert query
        insert_query = """
            INSERT INTO alerts (metric_name, value, timestamp, resource_type, severity, action_taken)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        cursor.execute(
            insert_query,
            (metric_name, float(value), alert_timestamp, resource_type, severity, action_taken)
        )
        
        conn.commit()
        cursor.close()
        
        return True
    
    except Error as e:
        print(f"Error storing alert in database: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error storing alert: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()


def get_database_connection(connection_string: str) -> DatabaseConnection:
    """
    Create and return a database connection.
    
    Args:
        connection_string: PostgreSQL connection string
    
    Returns:
        DatabaseConnection instance
    
    TODO: Implement connection factory with connection pooling
    """
    # TODO: Implement connection pooling if needed
    return DatabaseConnection(connection_string)
