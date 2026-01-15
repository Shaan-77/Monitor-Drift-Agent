"""
Integration tests for cost spike monitoring and automatic alerting.

Tests that verify cost spikes are automatically detected and alerts
are triggered during the collection cycle.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import modules to test
try:
    from collect_metrics import (
        monitor_cost_spikes_and_trigger_alerts,
        monitor_system_metrics_and_trigger_alerts
    )
    from data_collection.cloud_metrics import get_cloud_costs
    from anomaly_detection.threshold_detection import (
        compare_cost_to_historical,
        monitor_sustained_spikes,
        check_thresholds
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    monitor_cost_spikes_and_trigger_alerts = None
    monitor_system_metrics_and_trigger_alerts = None


class TestCostSpikeMonitoring(unittest.TestCase):
    """Test cases for cost spike monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    @unittest.skipUnless(MONITORING_AVAILABLE, "Monitoring modules not available")
    @patch('data_collection.cloud_metrics.get_cloud_costs')
    @patch('anomaly_detection.threshold_detection.compare_cost_to_historical')
    @patch('anomaly_detection.threshold_detection.monitor_sustained_spikes')
    @patch('anomaly_detection.threshold_detection.check_thresholds')
    @patch('config.settings.get_settings')
    def test_cost_spike_monitoring_detects_spikes(self, mock_get_settings, mock_check_thresholds,
                                                   mock_monitor_sustained, mock_compare_historical,
                                                   mock_get_cloud_costs):
        """
        Test that cost spike monitoring detects spikes and triggers alerts.
        """
        # Mock settings
        mock_settings = Mock()
        mock_settings.enable_cloud_metrics = True
        mock_settings.cost_spike_monitoring_enabled = True
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.sustained_cost_spike_days = 3
        mock_get_settings.return_value = mock_settings
        
        # Mock cloud cost data with a spike
        mock_get_cloud_costs.return_value = {
            'cloud_cost': [
                {
                    'resource_name': 'AWS EC2',
                    'cost': 600.0,  # Above threshold
                    'resource_usage': 100.0,
                    'timestamp': datetime.utcnow()
                },
                {
                    'resource_name': 'AWS S3',
                    'cost': 200.0,  # Normal
                    'resource_usage': 50.0,
                    'timestamp': datetime.utcnow()
                }
            ],
            'total_cost': 800.0,
            'timestamp': datetime.utcnow()
        }
        
        # Mock threshold check - should trigger alert for EC2
        mock_check_thresholds.return_value = [
            {
                'metric_name': 'Cloud Cost',
                'value': 600.0,
                'threshold': 500.0
            }
        ]
        
        # Mock historical comparison - no spike for S3
        def mock_compare(current_cost, resource_name, **kwargs):
            if resource_name == 'AWS EC2':
                return {
                    'spike_detected': True,
                    'spike_percentage': 25.0,
                    'historical_average': 480.0,
                    'alert_triggered': True
                }
            return {
                'spike_detected': False,
                'spike_percentage': 0.0,
                'historical_average': 200.0,
                'alert_triggered': False
            }
        
        mock_compare_historical.side_effect = mock_compare
        
        # Mock sustained spike monitoring
        mock_monitor_sustained.return_value = {
            'spike_detected': False,
            'consecutive_days': 0
        }
        
        # Run monitoring
        result = monitor_cost_spikes_and_trigger_alerts()
        
        # Verify monitoring ran
        self.assertIsInstance(result, dict)
        self.assertIn('monitored', result)
        self.assertIn('alerts_triggered', result)
        
        # Verify cloud costs were collected
        mock_get_cloud_costs.assert_called_once()
        
        # Verify threshold check was called
        self.assertTrue(mock_check_thresholds.called)
        
        # Verify historical comparison was called
        self.assertTrue(mock_compare_historical.called)
    
    @unittest.skipUnless(MONITORING_AVAILABLE, "Monitoring modules not available")
    @patch('config.settings.get_settings')
    def test_cost_spike_monitoring_respects_settings(self, mock_get_settings):
        """
        Test that cost spike monitoring respects enable/disable settings.
        """
        # Test with monitoring disabled
        mock_settings = Mock()
        mock_settings.enable_cloud_metrics = True
        mock_settings.cost_spike_monitoring_enabled = False
        mock_get_settings.return_value = mock_settings
        
        result = monitor_cost_spikes_and_trigger_alerts()
        
        self.assertIsInstance(result, dict)
        self.assertIn('skipped', result)
        self.assertEqual(result.get('skipped'), 'Monitoring disabled')
        
        # Test with cloud metrics disabled
        mock_settings.cost_spike_monitoring_enabled = True
        mock_settings.enable_cloud_metrics = False
        
        result = monitor_cost_spikes_and_trigger_alerts()
        
        self.assertIsInstance(result, dict)
        self.assertIn('skipped', result)
        self.assertEqual(result.get('skipped'), 'Cloud metrics disabled')
    
    @unittest.skipUnless(MONITORING_AVAILABLE, "Monitoring modules not available")
    @patch('anomaly_detection.threshold_detection.check_thresholds')
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    @patch('config.settings.get_settings')
    def test_system_metrics_monitoring_triggers_alerts(self, mock_get_settings,
                                                       mock_trigger_alert, mock_check_thresholds):
        """
        Test that system metrics monitoring triggers alerts for high CPU/memory.
        """
        # Mock settings
        mock_settings = Mock()
        mock_settings.system_metrics_monitoring_enabled = True
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.default_memory_threshold = 80.0
        mock_get_settings.return_value = mock_settings
        
        # Mock threshold check for CPU
        mock_check_thresholds.return_value = [
            {
                'metric_name': 'CPU Usage',
                'value': 85.0,
                'threshold': 80.0
            }
        ]
        
        # Mock trigger_alert for memory
        mock_trigger_alert.return_value = True
        
        # Test with high CPU and memory
        server_metrics = {
            'cpu_usage': {'cpu_percent': 85.0},
            'memory_usage': {'percent': 90.0}
        }
        
        result = monitor_system_metrics_and_trigger_alerts(server_metrics)
        
        # Verify monitoring ran
        self.assertIsInstance(result, dict)
        self.assertIn('monitored', result)
        self.assertIn('alerts_triggered', result)
        
        # Verify threshold check was called for CPU
        self.assertTrue(mock_check_thresholds.called)
        
        # Verify alert was triggered for memory
        self.assertTrue(mock_trigger_alert.called)


if __name__ == '__main__':
    unittest.main()
