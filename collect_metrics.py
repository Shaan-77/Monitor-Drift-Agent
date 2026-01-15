"""
Continuous metric collection script.

This script uses APScheduler to continuously collect system metrics
at regular intervals and store them in the PostgreSQL database.
"""

import signal
import sys
import time
import logging
from datetime import datetime

# Try to import APScheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.executors.pool import ThreadPoolExecutor
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None

# Import metric collection functions
try:
    from data_collection.system_metrics import collect_and_store_metrics, collect_all_metrics
except ImportError:
    def collect_and_store_metrics():
        raise RuntimeError("Metric collection module not available")
    def collect_all_metrics():
        raise RuntimeError("Metric collection module not available")

# Import settings and logging
try:
    from config.settings import get_settings
    from config.logging_config import setup_logging
except ImportError:
    def get_settings():
        return None
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def monitor_cost_spikes_and_trigger_alerts():
    """
    Monitor cloud costs and automatically detect spikes, triggering alerts.
    
    This function addresses the pain point of unforeseen costs by:
    1. Collecting real-time cloud costs
    2. Comparing against historical averages to detect spikes
    3. Monitoring sustained spikes over consecutive days
    4. Checking against absolute cost thresholds
    5. Automatically triggering alerts when anomalies are detected
    
    This allows engineers to investigate and take corrective actions
    before costs spiral out of control.
    
    Returns:
        Dictionary with monitoring results including alert counts
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Import required modules
        from data_collection.cloud_metrics import get_cloud_costs
        from anomaly_detection.threshold_detection import (
            compare_cost_to_historical,
            monitor_sustained_spikes,
            check_thresholds
        )
        from config.settings import get_settings
        
        settings = get_settings()
        if settings is None:
            logger.warning("Settings not available, skipping cost spike monitoring")
            return {'monitored': 0, 'alerts_triggered': 0, 'error': 'Settings not available'}
        
        # Check if cost spike monitoring is enabled
        cost_monitoring_enabled = getattr(settings, 'cost_spike_monitoring_enabled', True)
        if not cost_monitoring_enabled:
            logger.debug("Cost spike monitoring disabled in settings")
            return {'monitored': 0, 'alerts_triggered': 0, 'skipped': 'Monitoring disabled'}
        
        if not settings.enable_cloud_metrics:
            logger.debug("Cloud metrics disabled, skipping cost spike monitoring")
            return {'monitored': 0, 'alerts_triggered': 0, 'skipped': 'Cloud metrics disabled'}
        
        logger.info("Starting cost spike monitoring and alert detection...")
        
        # Collect real-time cloud costs
        try:
            cloud_cost_data = get_cloud_costs(store_in_db=True)
            cost_items = cloud_cost_data.get('cloud_cost', [])
            total_cost = cloud_cost_data.get('total_cost', 0.0)
            
            if not cost_items:
                logger.debug("No cloud cost data collected, skipping spike detection")
                return {'monitored': 0, 'alerts_triggered': 0, 'skipped': 'No cost data'}
            
            logger.info(f"Collected {len(cost_items)} cloud cost items. Total cost: ${total_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to collect cloud costs: {e}", exc_info=True)
            return {'monitored': 0, 'alerts_triggered': 0, 'error': str(e)}
        
        # Monitor each resource for cost spikes
        alerts_triggered = 0
        resources_monitored = 0
        
        for cost_item in cost_items:
            resource_name = cost_item.get('resource_name', 'Unknown')
            current_cost = cost_item.get('cost', 0.0)
            timestamp = cost_item.get('timestamp')
            
            if not timestamp:
                timestamp = datetime.utcnow()
            elif isinstance(timestamp, str):
                from datetime import datetime as dt
                try:
                    timestamp = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if timestamp.tzinfo:
                        timestamp = timestamp.astimezone().replace(tzinfo=None)
                except (ValueError, AttributeError):
                    timestamp = datetime.utcnow()
            
            resources_monitored += 1
            
            try:
                # 1. Check against absolute threshold (e.g., $500/day)
                threshold_alerts = check_thresholds(
                    cpu_usage=0.0,  # Not checking CPU here
                    cloud_cost=current_cost,
                    resource_type="Cloud"
                )
                if threshold_alerts:
                    alerts_triggered += len(threshold_alerts)
                    logger.warning(
                        f"Threshold alert triggered for {resource_name}: "
                        f"Cost ${current_cost:.2f} exceeds threshold ${settings.cloud_cost_threshold:.2f}"
                    )
                
                # 2. Compare to historical average (detect spikes)
                try:
                    historical_comparison = compare_cost_to_historical(
                        current_cost=current_cost,
                        resource_name=resource_name
                    )
                    
                    if historical_comparison.get('spike_detected', False):
                        alerts_triggered += 1
                        spike_percent = historical_comparison.get('spike_percentage', 0.0)
                        historical_avg = historical_comparison.get('historical_average', 0.0)
                        logger.warning(
                            f"Cost spike detected for {resource_name}: "
                            f"Current ${current_cost:.2f} is {spike_percent:.1f}% above "
                            f"historical average ${historical_avg:.2f}"
                        )
                except Exception as e:
                    # Historical comparison may fail if no historical data exists
                    logger.debug(f"Could not compare {resource_name} to historical: {e}")
                
                # 3. Monitor sustained spikes (consecutive days)
                try:
                    sustained_spike = monitor_sustained_spikes(
                        resource_name=resource_name,
                        current_cost=current_cost
                    )
                    
                    if sustained_spike.get('spike_detected', False):
                        consecutive_days = sustained_spike.get('consecutive_days', 0)
                        if consecutive_days >= settings.sustained_cost_spike_days:
                            alerts_triggered += 1
                            logger.warning(
                                f"Sustained cost spike for {resource_name}: "
                                f"Cost ${current_cost:.2f} exceeded threshold ${settings.cloud_cost_threshold:.2f} "
                                f"for {consecutive_days} consecutive days"
                            )
                except Exception as e:
                    logger.debug(f"Could not monitor sustained spikes for {resource_name}: {e}")
                
            except Exception as e:
                logger.error(
                    f"Error monitoring cost spikes for {resource_name}: {e}",
                    exc_info=True
                )
                continue
        
        result = {
            'monitored': resources_monitored,
            'alerts_triggered': alerts_triggered,
            'total_cost': total_cost
        }
        
        if alerts_triggered > 0:
            logger.warning(
                f"Cost spike monitoring completed: {alerts_triggered} alert(s) triggered "
                f"for {resources_monitored} resource(s)"
            )
        else:
            logger.info(
                f"Cost spike monitoring completed: No anomalies detected "
                f"for {resources_monitored} resource(s)"
            )
        
        return result
        
    except ImportError as e:
        logger.warning(f"Required modules not available for cost spike monitoring: {e}")
        return {'monitored': 0, 'alerts_triggered': 0, 'error': f'Import error: {e}'}
    except Exception as e:
        logger.error(f"Error during cost spike monitoring: {e}", exc_info=True)
        return {'monitored': 0, 'alerts_triggered': 0, 'error': str(e)}


def monitor_system_metrics_and_trigger_alerts(server_metrics: dict = None):
    """
    Monitor system metrics (CPU, memory) and automatically trigger alerts.
    
    This function monitors CPU and memory usage and triggers alerts when
    thresholds are exceeded, ensuring system resource spikes are detected.
    
    Args:
        server_metrics: Optional dictionary with server metrics.
                       If None, will collect fresh metrics.
    
    Returns:
        Dictionary with monitoring results including alert counts
    """
    logger = logging.getLogger(__name__)
    
    try:
        from anomaly_detection.threshold_detection import check_thresholds
        from config.settings import get_settings
        
        settings = get_settings()
        if settings is None:
            logger.warning("Settings not available, skipping system metrics monitoring")
            return {'monitored': 0, 'alerts_triggered': 0, 'error': 'Settings not available'}
        
        # Check if system metrics monitoring is enabled
        system_monitoring_enabled = getattr(settings, 'system_metrics_monitoring_enabled', True)
        if not system_monitoring_enabled:
            logger.debug("System metrics monitoring disabled in settings")
            return {'monitored': 0, 'alerts_triggered': 0, 'skipped': 'Monitoring disabled'}
        
        logger.info("Starting system metrics monitoring and alert detection...")
        
        # Get server metrics if not provided
        if server_metrics is None:
            try:
                from data_collection.system_metrics import get_server_metrics
                server_metrics = get_server_metrics()
            except Exception as e:
                logger.error(f"Failed to collect server metrics: {e}", exc_info=True)
                return {'monitored': 0, 'alerts_triggered': 0, 'error': str(e)}
        
        if not server_metrics or 'error' in server_metrics:
            logger.debug("No server metrics available, skipping monitoring")
            return {'monitored': 0, 'alerts_triggered': 0, 'skipped': 'No metrics'}
        
        alerts_triggered = 0
        
        # Extract CPU usage
        cpu_usage = 0.0
        if 'cpu_usage' in server_metrics:
            cpu_data = server_metrics['cpu_usage']
            if isinstance(cpu_data, dict):
                cpu_usage = cpu_data.get('cpu_percent', 0.0)
            elif isinstance(cpu_data, (int, float)):
                cpu_usage = float(cpu_data)
        elif 'cpu_percent' in server_metrics:
            cpu_usage = float(server_metrics['cpu_percent'])
        
        # Extract memory usage
        memory_usage = 0.0
        if 'memory_usage' in server_metrics:
            memory_data = server_metrics['memory_usage']
            if isinstance(memory_data, dict):
                memory_usage = memory_data.get('percent', 0.0)
            elif isinstance(memory_data, (int, float)):
                memory_usage = float(memory_data)
        elif 'memory_percent' in server_metrics:
            memory_usage = float(server_metrics['memory_percent'])
        
        # Monitor CPU usage
        if cpu_usage > 0:
            try:
                cpu_alerts = check_thresholds(
                    cpu_usage=cpu_usage,
                    cloud_cost=0.0,  # Not checking cloud cost here
                    resource_type="Server"
                )
                if cpu_alerts:
                    alerts_triggered += len(cpu_alerts)
                    logger.warning(
                        f"CPU usage alert triggered: {cpu_usage:.1f}% exceeds threshold "
                        f"{settings.cpu_usage_threshold:.1f}%"
                    )
            except Exception as e:
                logger.error(f"Error checking CPU thresholds: {e}", exc_info=True)
        
        # Monitor memory usage (using check_thresholds with memory as CPU equivalent)
        # Note: check_thresholds currently only checks CPU, but we can extend it
        # For now, we'll check memory manually
        if memory_usage > 0:
            try:
                memory_threshold = getattr(settings, 'default_memory_threshold', 80.0)
                if memory_usage > memory_threshold:
                    # Trigger alert for memory
                    try:
                        from anomaly_detection.alert_trigger import trigger_alert
                        alert_triggered = trigger_alert(
                            "Memory Usage",
                            memory_usage,
                            datetime.utcnow(),
                            "Server"
                        )
                        if alert_triggered:
                            alerts_triggered += 1
                            logger.warning(
                                f"Memory usage alert triggered: {memory_usage:.1f}% exceeds threshold "
                                f"{memory_threshold:.1f}%"
                            )
                    except Exception as e:
                        logger.error(f"Error triggering memory alert: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error checking memory thresholds: {e}", exc_info=True)
        
        result = {
            'monitored': 2 if (cpu_usage > 0 or memory_usage > 0) else 0,
            'alerts_triggered': alerts_triggered,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        }
        
        if alerts_triggered > 0:
            logger.warning(
                f"System metrics monitoring completed: {alerts_triggered} alert(s) triggered"
            )
        else:
            logger.info(
                f"System metrics monitoring completed: No anomalies detected "
                f"(CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)"
            )
        
        return result
        
    except ImportError as e:
        logger.warning(f"Required modules not available for system metrics monitoring: {e}")
        return {'monitored': 0, 'alerts_triggered': 0, 'error': f'Import error: {e}'}
    except Exception as e:
        logger.error(f"Error during system metrics monitoring: {e}", exc_info=True)
        return {'monitored': 0, 'alerts_triggered': 0, 'error': str(e)}


def run_metric_collection():
    """
    Job function to collect and store metrics.
    
    This function is called by the scheduler at regular intervals.
    It collects system metrics and stores them in the database.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Starting metric collection at {datetime.now()}")
        success = collect_and_store_metrics()
        if success:
            logger.info("Metric collection completed successfully")
        else:
            logger.warning("Metric collection completed but storage may have failed")
        
        # Automatically monitor system metrics and trigger alerts
        # This ensures system resource spikes are detected
        try:
            from data_collection.system_metrics import get_server_metrics
            server_metrics = get_server_metrics()
            monitor_system_metrics_and_trigger_alerts(server_metrics)
        except Exception as e:
            logger.error(f"Error during system metrics monitoring: {e}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Error during metric collection: {e}", exc_info=True)


def run_all_metrics_collection():
    """
    Job function to collect all metrics (server and cloud).
    
    This function is called by the scheduler at regular intervals.
    It collects both server and cloud metrics into a unified structure
    and stores them in the database.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Starting all metrics collection (server + cloud) at {datetime.now()}")
        all_metrics = collect_all_metrics()
        
        # Log summary of collected metrics
        server_ok = 'error' not in all_metrics.get('server_metrics', {})
        cloud_providers = len([k for k, v in all_metrics.get('cloud_metrics', {}).items() if v and not isinstance(v, list) or len(v) > 0])
        
        logger.info(
            f"All metrics collection completed. Server metrics: {'OK' if server_ok else 'Failed'}, "
            f"Cloud providers with metrics: {cloud_providers}"
        )
        
        # Log metric counts per provider
        for provider, metrics in all_metrics.get('cloud_metrics', {}).items():
            if isinstance(metrics, list):
                logger.info(f"  {provider.upper()}: {len(metrics)} metrics collected")
            elif metrics:
                logger.debug(f"  {provider.upper()}: {metrics}")
        
        # Store metrics in database
        try:
            from data_collection.database import store_all_metrics
            storage_success = store_all_metrics(all_metrics)
            if storage_success:
                logger.info("All metrics stored in database successfully")
            else:
                logger.warning("Failed to store all metrics in database")
        except ImportError as e:
            logger.warning(f"Database storage module not available: {str(e)}")
        except Exception as e:
            logger.error(f"Error storing metrics: {e}", exc_info=True)
        
        # Automatically monitor for cost spikes and trigger alerts
        # This addresses the pain point of unforeseen costs
        try:
            monitor_cost_spikes_and_trigger_alerts()
        except Exception as e:
            logger.error(f"Error during cost spike monitoring: {e}", exc_info=True)
        
        # Automatically monitor system metrics and trigger alerts
        # This ensures system resource spikes are detected
        try:
            server_metrics = all_metrics.get('server_metrics', {})
            monitor_system_metrics_and_trigger_alerts(server_metrics)
        except Exception as e:
            logger.error(f"Error during system metrics monitoring: {e}", exc_info=True)
        
        return all_metrics
        
    except Exception as e:
        logger.error(f"Error during all metrics collection: {e}", exc_info=True)
        return None


def main():
    """
    Main function to start the scheduler.
    
    Sets up APScheduler to run metric collection at regular intervals
    based on configuration settings.
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if APScheduler is available
    if not APSCHEDULER_AVAILABLE:
        logger.error(
            "APScheduler is not available. Please install it using: pip install apscheduler"
        )
        sys.exit(1)
    
    # Get settings
    settings = get_settings()
    if settings is None:
        logger.warning("Settings not available, using default interval of 60 seconds")
        interval = 60
        enable_cloud_metrics = False
    else:
        interval = settings.metric_collection_interval
        enable_cloud_metrics = settings.enable_cloud_metrics
    
    logger.info(f"Initializing metric collection scheduler with interval: {interval} seconds")
    if enable_cloud_metrics:
        logger.info("Cloud metrics collection is enabled")
    else:
        logger.info("Cloud metrics collection is disabled (set ENABLE_CLOUD_METRICS=true to enable)")
    
    # Create scheduler with thread pool executor
    executors = {
        'default': ThreadPoolExecutor(1)
    }
    scheduler = BackgroundScheduler(executors=executors)
    
    # Add job to scheduler - use all metrics collection if cloud metrics enabled
    if enable_cloud_metrics:
        scheduler.add_job(
            run_all_metrics_collection,
            'interval',
            seconds=interval,
            id='all_metrics_collection_job',
            name='All Metrics Collection (Server + Cloud)',
            replace_existing=True
        )
        initial_collection_func = run_all_metrics_collection
    else:
        scheduler.add_job(
            run_metric_collection,
            'interval',
            seconds=interval,
            id='metric_collection_job',
            name='System Metrics Collection',
            replace_existing=True
        )
        initial_collection_func = run_metric_collection
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down scheduler...")
        if scheduler.running:
            scheduler.shutdown(wait=True)
        logger.info("Scheduler shut down successfully")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start scheduler
    try:
        scheduler.start()
        logger.info(f"Scheduler started. Collecting metrics every {interval} seconds.")
        logger.info("Press Ctrl+C to stop the scheduler.")
        
        # Run initial collection immediately
        logger.info("Running initial metric collection...")
        initial_collection_func()
        
        # Keep process alive
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Received keyboard interrupt. Shutting down...")
            signal_handler(signal.SIGINT, None)
    
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}", exc_info=True)
        if scheduler.running:
            scheduler.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
