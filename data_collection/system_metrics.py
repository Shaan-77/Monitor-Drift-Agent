"""
Logic for collecting system metrics.

This module provides functionality to collect system metrics including
CPU usage, memory usage, disk I/O, and network statistics.
"""

from typing import Dict, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import random

# Try to import psutil, handle gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    def collect(self) -> Dict:
        """Collect metrics and return as dictionary."""
        pass


class SystemMetricsCollector(MetricCollector):
    """Collector for system-level metrics."""
    
    def collect_cpu_metrics(self) -> Dict:
        """
        Collect CPU usage metrics.
        
        Returns:
            Dictionary containing CPU metrics (usage percentage, load average, etc.)
        
        TODO: Implement actual CPU metric collection using psutil or similar
        """
        # TODO: Implement CPU metric collection
        return {
            "cpu_percent": 0.0,
            "cpu_count": 0,
            "load_average": [0.0, 0.0, 0.0],
            "timestamp": datetime.now().isoformat()
        }
    
    def collect_memory_metrics(self) -> Dict:
        """
        Collect memory usage metrics.
        
        Returns:
            Dictionary containing memory metrics (total, used, available, etc.)
        
        TODO: Implement actual memory metric collection
        """
        # TODO: Implement memory metric collection
        return {
            "total": 0,
            "used": 0,
            "available": 0,
            "percent": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
    def collect_disk_metrics(self) -> Dict:
        """
        Collect disk I/O metrics.
        
        Returns:
            Dictionary containing disk metrics (usage, I/O stats, etc.)
        
        TODO: Implement actual disk metric collection
        """
        # TODO: Implement disk metric collection
        return {
            "total": 0,
            "used": 0,
            "free": 0,
            "percent": 0.0,
            "read_bytes": 0,
            "write_bytes": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def collect_network_metrics(self) -> Dict:
        """
        Collect network statistics.
        
        Returns:
            Dictionary containing network metrics (bytes sent/received, etc.)
        
        TODO: Implement actual network metric collection
        """
        # TODO: Implement network metric collection
        return {
            "bytes_sent": 0,
            "bytes_recv": 0,
            "packets_sent": 0,
            "packets_recv": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def collect(self) -> Dict:
        """
        Collect all system metrics.
        
        Returns:
            Dictionary containing all collected system metrics
        
        TODO: Implement comprehensive metric collection
        """
        # TODO: Implement collection of all metrics
        return {
            "cpu": self.collect_cpu_metrics(),
            "memory": self.collect_memory_metrics(),
            "disk": self.collect_disk_metrics(),
            "network": self.collect_network_metrics(),
            "timestamp": datetime.now().isoformat()
        }


def get_cpu_usage() -> Dict:
    """
    Retrieve the current CPU usage percentage.
    
    Returns:
        Dictionary containing:
            - cpu_percent: float (CPU usage percentage)
            - cpu_count: int (number of CPU cores)
            - timestamp: datetime (exact capture time)
    
    Raises:
        RuntimeError: If psutil is not available or CPU metrics cannot be collected
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("psutil library is not available. Please install it using: pip install psutil")
    
    try:
        # Get CPU usage percentage with 1 second interval for accuracy
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise RuntimeError(f"Failed to collect CPU metrics: {str(e)}")


def get_memory_usage() -> Dict:
    """
    Retrieve the current memory usage (percentage and bytes).
    
    Returns:
        Dictionary containing:
            - total: int (total memory in bytes)
            - used: int (used memory in bytes)
            - available: int (available memory in bytes)
            - percent: float (memory usage percentage)
            - timestamp: datetime (exact capture time)
    
    Raises:
        RuntimeError: If psutil is not available or memory metrics cannot be collected
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("psutil library is not available. Please install it using: pip install psutil")
    
    try:
        memory = psutil.virtual_memory()
        
        return {
            "total": memory.total,
            "used": memory.used,
            "available": memory.available,
            "percent": memory.percent,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise RuntimeError(f"Failed to collect memory metrics: {str(e)}")


def get_network_traffic() -> Dict:
    """
    Retrieve the current network traffic (in bytes, sent and received).
    
    Returns:
        Dictionary containing:
            - bytes_sent: int (total bytes sent)
            - bytes_recv: int (total bytes received)
            - packets_sent: int (total packets sent)
            - packets_recv: int (total packets received)
            - timestamp: datetime (exact capture time)
    
    Raises:
        RuntimeError: If psutil is not available or network metrics cannot be collected
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("psutil library is not available. Please install it using: pip install psutil")
    
    try:
        net_io = psutil.net_io_counters()
        
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise RuntimeError(f"Failed to collect network metrics: {str(e)}")


def get_cpu_usage_percent() -> float:
    """
    Retrieve the current CPU usage as a percentage.
    
    Returns:
        float: CPU usage percentage (0.0 to 100.0)
    
    Raises:
        RuntimeError: If psutil is not available or CPU metrics cannot be collected
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("psutil library is not available. Please install it using: pip install psutil")
    
    try:
        # Get CPU usage percentage with 1 second interval for accuracy
        cpu_percent = psutil.cpu_percent(interval=1)
        return float(cpu_percent)
    except Exception as e:
        raise RuntimeError(f"Failed to collect CPU usage: {str(e)}")


def get_memory_usage_percent() -> float:
    """
    Retrieve the current memory usage as a percentage.
    
    Returns:
        float: Memory usage percentage (0.0 to 100.0)
    
    Raises:
        RuntimeError: If psutil is not available or memory metrics cannot be collected
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("psutil library is not available. Please install it using: pip install psutil")
    
    try:
        memory = psutil.virtual_memory()
        return float(memory.percent)
    except Exception as e:
        raise RuntimeError(f"Failed to collect memory usage: {str(e)}")


def get_server_metrics() -> Dict:
    """
    Collect server load metrics (CPU and memory) with timestamp.
    
    Returns:
        Dictionary containing:
            - cpu_usage: float (CPU usage percentage)
            - memory_usage: float (Memory usage percentage)
            - timestamp: datetime (exact capture time)
    
    Raises:
        RuntimeError: If psutil is not available or metrics cannot be collected
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("psutil library is not available. Please install it using: pip install psutil")
    
    try:
        # Collect CPU and memory usage percentages
        cpu_usage = get_cpu_usage_percent()
        memory_usage = get_memory_usage_percent()
        
        # Capture timestamp when metrics are collected (UTC)
        timestamp = datetime.utcnow()
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'timestamp': timestamp
        }
    except RuntimeError:
        # Re-raise RuntimeError from helper functions
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to collect server metrics: {str(e)}")


def collect_and_store_metrics() -> bool:
    """
    Collect system metrics and store them in the database.
    
    This function collects CPU, memory, and network metrics using
    the individual collection functions and stores them in PostgreSQL.
    
    Returns:
        True if collection and storage successful, False otherwise
    """
    try:
        # Import here to avoid circular dependencies
        from data_collection.database import store_metrics
        from utils.logger import get_logger
        
        logger = get_logger(__name__)
        
        # Collect all metrics
        logger.info("Collecting system metrics...")
        cpu_data = get_cpu_usage()
        memory_data = get_memory_usage()
        network_data = get_network_traffic()
        
        # Store metrics in database
        logger.info("Storing metrics in database...")
        success = store_metrics(cpu_data, memory_data, network_data)
        
        if success:
            logger.info("Metrics collected and stored successfully")
        else:
            logger.error("Failed to store metrics in database")
        
        return success
    
    except RuntimeError as e:
        # Handle errors from metric collection (e.g., psutil not available)
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Failed to collect metrics: {str(e)}")
        except ImportError:
            print(f"Failed to collect metrics: {str(e)}")
        return False
    
    except Exception as e:
        # Handle any other unexpected errors
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Unexpected error during metric collection: {str(e)}", exc_info=True)
        except ImportError:
            print(f"Unexpected error during metric collection: {str(e)}")
        return False


def collect_all_metrics() -> Dict:
    """
    Collect both server and cloud metrics into a unified structure.
    
    Coordinates collection of:
    - Server metrics (CPU usage, memory usage)
    - Cloud metrics (AWS, GCP, Azure)
    
    Returns:
        Dictionary containing:
            - timestamp: str (ISO format timestamp when collection started)
            - server_metrics: Dict (CPU and memory usage with timestamp)
            - cloud_metrics: Dict (AWS, GCP, Azure metrics by provider)
    
    Errors are handled gracefully - if a cloud provider is not configured
    or fails, it will be omitted from results rather than causing failure.
    """
    from utils.logger import get_logger
    logger = get_logger(__name__)
    
    # Capture collection start timestamp (UTC)
    collection_timestamp = datetime.utcnow()
    
    # Initialize result structure
    result = {
        'timestamp': collection_timestamp.isoformat(),
        'server_metrics': {},
        'cloud_metrics': {}
    }
    
    # Collect server metrics
    try:
        result['server_metrics'] = get_server_metrics()
        logger.info("Server metrics collected successfully")
    except RuntimeError as e:
        logger.error(f"Failed to collect server metrics: {str(e)}")
        result['server_metrics'] = {'error': str(e)}
    except Exception as e:
        logger.error(f"Unexpected error collecting server metrics: {str(e)}", exc_info=True)
        result['server_metrics'] = {'error': str(e)}
    
    # Collect cloud metrics - handle each provider independently
    try:
        from data_collection.cloud_metrics import get_aws_metrics, get_gcp_metrics, get_azure_metrics
    except ImportError as e:
        logger.warning(f"Cloud metrics module not available: {str(e)}")
        result['cloud_metrics'] = {'error': 'Cloud metrics module not available'}
        return result
    
    # Collect AWS metrics
    try:
        aws_metrics = get_aws_metrics()
        result['cloud_metrics']['aws'] = aws_metrics
        logger.info(f"AWS metrics collected: {len(aws_metrics)} metrics")
    except RuntimeError as e:
        logger.warning(f"AWS metrics collection failed (likely not configured): {str(e)}")
        result['cloud_metrics']['aws'] = []
    except Exception as e:
        logger.warning(f"Unexpected error collecting AWS metrics: {str(e)}")
        result['cloud_metrics']['aws'] = []
    
    # Collect GCP metrics
    try:
        gcp_metrics = get_gcp_metrics()
        result['cloud_metrics']['gcp'] = gcp_metrics
        logger.info(f"GCP metrics collected: {len(gcp_metrics)} metrics")
    except RuntimeError as e:
        logger.warning(f"GCP metrics collection failed (likely not configured): {str(e)}")
        result['cloud_metrics']['gcp'] = []
    except Exception as e:
        logger.warning(f"Unexpected error collecting GCP metrics: {str(e)}")
        result['cloud_metrics']['gcp'] = []
    
    # Collect Azure metrics
    try:
        azure_metrics = get_azure_metrics()
        result['cloud_metrics']['azure'] = azure_metrics
        logger.info(f"Azure metrics collected: {len(azure_metrics)} metrics")
    except RuntimeError as e:
        logger.warning(f"Azure metrics collection failed (likely not configured): {str(e)}")
        result['cloud_metrics']['azure'] = []
    except Exception as e:
        logger.warning(f"Unexpected error collecting Azure metrics: {str(e)}")
        result['cloud_metrics']['azure'] = []
    
    return result


def collect_system_metrics() -> Dict:
    """
    Convenience function to collect all system metrics.
    
    Returns:
        Dictionary containing all system metrics
    
    TODO: Implement metric collection using SystemMetricsCollector
    """
    collector = SystemMetricsCollector()
    return collector.collect()


def generate_simulated_cpu_usage(
    resource_id: str,
    base_value: float,
    variation: float,
    timestamp: datetime
) -> Dict:
    """
    Generate simulated CPU usage data.
    
    Args:
        resource_id: Identifier for the resource
        base_value: Base CPU usage percentage (0-100)
        variation: Maximum variation from base value
        timestamp: Timestamp for the metric
    
    Returns:
        Dictionary matching get_cpu_usage() format
    """
    # Use resource_id hash for deterministic variation
    random.seed(hash(resource_id + str(timestamp)) % 1000)
    cpu_variation = random.uniform(-variation, variation)
    cpu_percent = max(0.0, min(100.0, base_value + cpu_variation))
    
    return {
        "cpu_percent": cpu_percent,
        "cpu_count": 4 + (hash(resource_id) % 8),  # 4-12 cores
        "timestamp": timestamp
    }


def generate_simulated_memory_usage(
    resource_id: str,
    base_percent: float,
    variation: float,
    timestamp: datetime
) -> Dict:
    """
    Generate simulated memory usage data.
    
    Args:
        resource_id: Identifier for the resource
        base_percent: Base memory usage percentage (0-100)
        variation: Maximum variation from base percentage
        timestamp: Timestamp for the metric
    
    Returns:
        Dictionary matching get_memory_usage() format
    """
    # Use resource_id hash for deterministic variation
    random.seed(hash(resource_id + str(timestamp)) % 1000)
    memory_variation = random.uniform(-variation, variation)
    memory_percent = max(0.0, min(100.0, base_percent + memory_variation))
    
    total_memory = 16 * 1024 * 1024 * 1024  # 16GB base
    used_memory = int(total_memory * (memory_percent / 100.0))
    
    return {
        "total": total_memory,
        "used": used_memory,
        "available": total_memory - used_memory,
        "percent": memory_percent,
        "timestamp": timestamp
    }


def generate_simulated_network_traffic(
    resource_id: str,
    base_bytes: int,
    variation: float,
    timestamp: datetime
) -> Dict:
    """
    Generate simulated network traffic data.
    
    Args:
        resource_id: Identifier for the resource
        base_bytes: Base network traffic in bytes
        variation: Maximum variation as fraction (e.g., 0.1 for 10%)
        timestamp: Timestamp for the metric
    
    Returns:
        Dictionary matching get_network_traffic() format
    """
    # Use resource_id hash for deterministic variation
    random.seed(hash(resource_id + str(timestamp)) % 1000)
    bytes_variation = random.uniform(-variation, variation)
    bytes_sent = max(0, int(base_bytes * (1 + bytes_variation)))
    bytes_recv = max(0, int(base_bytes * 2 * (1 + bytes_variation)))  # Typically receive more
    
    return {
        "bytes_sent": bytes_sent,
        "bytes_recv": bytes_recv,
        "packets_sent": bytes_sent // 1500,  # Approximate packet count
        "packets_recv": bytes_recv // 1500,
        "timestamp": timestamp
    }
