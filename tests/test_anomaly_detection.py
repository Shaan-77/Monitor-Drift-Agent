"""
Unit tests for anomaly detection modules.

Tests for threshold_detection, machine_learning, and alert_trigger modules.
"""

import unittest
from unittest.mock import Mock, patch, ANY
from datetime import datetime
import time

# Import modules to test
try:
    from anomaly_detection.threshold_detection import (
        check_thresholds,
        compare_cost_to_historical,
        ThresholdDetector,
        ThresholdRule
    )
    from anomaly_detection.alert_trigger import trigger_alert, trigger_cost_alert, AlertTrigger
    from config.settings import get_settings
    try:
        from data_collection.database import get_historical_cost_data
        COST_SPIKE_DETECTION_AVAILABLE = True
    except ImportError:
        get_historical_cost_data = None
        COST_SPIKE_DETECTION_AVAILABLE = False
    THRESHOLD_DETECTION_AVAILABLE = True
except ImportError:
    THRESHOLD_DETECTION_AVAILABLE = False
    COST_SPIKE_DETECTION_AVAILABLE = False
    check_thresholds = None
    compare_cost_to_historical = None
    trigger_alert = None
    trigger_cost_alert = None
    get_historical_cost_data = None

# TODO: Import ML modules when available
# from anomaly_detection.machine_learning import IsolationForestDetector, AutoencoderDetector


class TestThresholdDetector(unittest.TestCase):
    """Test cases for ThresholdDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # TODO: Initialize ThresholdDetector with test rules
        pass
    
    def test_threshold_rule_evaluation(self):
        """Test threshold rule evaluation."""
        # TODO: Implement test for threshold rule evaluation
        # rule = ThresholdRule("cpu_percent", 80.0, "gt", "high")
        # self.assertTrue(rule.evaluate(90.0))
        # self.assertFalse(rule.evaluate(70.0))
        pass
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        # TODO: Implement test for anomaly detection
        pass
    
    def test_add_remove_rules(self):
        """Test adding and removing rules."""
        # TODO: Implement test for rule management
        pass


class TestMLAnomalyDetector(unittest.TestCase):
    """Test cases for ML-based anomaly detectors."""
    
    def setUp(self):
        """Set up test fixtures."""
        # TODO: Initialize ML detectors
        pass
    
    def test_isolation_forest_training(self):
        """Test Isolation Forest model training."""
        # TODO: Implement test for model training
        pass
    
    def test_isolation_forest_detection(self):
        """Test Isolation Forest anomaly detection."""
        # TODO: Implement test for anomaly detection
        pass
    
    def test_autoencoder_training(self):
        """Test Autoencoder model training."""
        # TODO: Implement test for model training
        pass
    
    def test_autoencoder_detection(self):
        """Test Autoencoder anomaly detection."""
        # TODO: Implement test for anomaly detection
        pass


class TestAlertTrigger(unittest.TestCase):
    """Test cases for AlertTrigger."""
    
    def setUp(self):
        """Set up test fixtures."""
        # TODO: Initialize AlertTrigger with mock alert system
        pass
    
    def test_trigger_alert(self):
        """Test alert triggering."""
        # TODO: Implement test for alert triggering
        pass
    
    def test_determine_severity(self):
        """Test severity determination."""
        # TODO: Implement test for severity determination
        pass
    
    def test_format_alert_message(self):
        """Test alert message formatting."""
        # TODO: Implement test for message formatting
        pass


@unittest.skipUnless(THRESHOLD_DETECTION_AVAILABLE, "Threshold detection module not available")
class TestCheckThresholds(unittest.TestCase):
    """Test cases for check_thresholds() function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear threshold exceedance tracking before each test
        if THRESHOLD_DETECTION_AVAILABLE:
            from anomaly_detection.threshold_detection import _threshold_exceedance_times
            _threshold_exceedance_times.clear()
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_cpu_threshold_exceedance_immediate(self, mock_get_settings, mock_trigger_alert):
        """
        Test CPU usage exceeding threshold but not meeting duration requirement.
        
        Verifies that alert is not triggered immediately when threshold is exceeded
        but duration requirement is not met.
        """
        # Setup mock settings
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5  # 5 minutes
        mock_get_settings.return_value = mock_settings
        
        # Mock trigger_alert to return True
        mock_trigger_alert.return_value = True
        
        # Call with CPU usage above threshold
        result = check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        
        # Alert should not be triggered yet (duration not met)
        mock_trigger_alert.assert_not_called()
        self.assertEqual(len(result), 0)
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    @patch('anomaly_detection.threshold_detection.time')
    @patch('anomaly_detection.threshold_detection.datetime')
    def test_cpu_threshold_exceedance_with_duration(self, mock_datetime, mock_time, mock_get_settings, mock_trigger_alert):
        """
        Test CPU usage exceeding threshold for required duration.
        
        Verifies that alert is triggered when CPU usage exceeds threshold
        for the required duration (5 minutes).
        """
        # Setup mock settings
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5  # 5 minutes
        mock_get_settings.return_value = mock_settings
        
        # Mock time.time() to simulate time passing (Unix timestamps)
        # 6 minutes = 360 seconds
        start_timestamp = 1704110400.0  # Unix timestamp for 2024-01-01 12:00:00
        end_timestamp = start_timestamp + (6 * 60)  # 6 minutes later
        
        mock_time.time.side_effect = [start_timestamp, end_timestamp]
        
        # Mock datetime.utcnow() for alert timestamps
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 6, 0)  # 6 minutes later
        mock_datetime.utcnow.side_effect = [start_time, end_time]
        
        # Mock trigger_alert to return True
        mock_trigger_alert.return_value = True
        
        # First call - start tracking
        result1 = check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        self.assertEqual(len(result1), 0)
        mock_trigger_alert.assert_not_called()
        
        # Second call - duration met, alert triggered
        result2 = check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        mock_trigger_alert.assert_called_once_with("CPU Usage", 85.0, end_time, "Server")
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0]["metric_name"], "CPU Usage")
        self.assertEqual(result2[0]["value"], 85.0)
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    @patch('anomaly_detection.threshold_detection.time')
    @patch('anomaly_detection.threshold_detection.datetime')
    def test_cloud_cost_threshold_exceedance(self, mock_datetime, mock_time, mock_get_settings, mock_trigger_alert):
        """
        Test cloud cost exceeding threshold with duration requirement.
        
        Verifies that alert is triggered when cloud cost exceeds threshold
        for the required duration (5 minutes).
        """
        # Setup mock settings
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5  # 5 minutes
        mock_get_settings.return_value = mock_settings
        
        # Mock time.time() to simulate time passing (Unix timestamps)
        # 6 minutes = 360 seconds
        start_timestamp = 1704110400.0  # Unix timestamp for 2024-01-01 12:00:00
        end_timestamp = start_timestamp + (6 * 60)  # 6 minutes later
        
        mock_time.time.side_effect = [start_timestamp, end_timestamp]
        
        # Mock datetime.utcnow() for alert timestamps
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 6, 0)  # 6 minutes later
        mock_datetime.utcnow.side_effect = [start_time, end_time]
        
        # Mock trigger_alert to return True
        mock_trigger_alert.return_value = True
        
        # First call - start tracking (no alert yet)
        result1 = check_thresholds(cpu_usage=50.0, cloud_cost=600.0)
        self.assertEqual(len(result1), 0)
        mock_trigger_alert.assert_not_called()
        
        # Second call - duration met, alert triggered
        result2 = check_thresholds(cpu_usage=50.0, cloud_cost=600.0)
        mock_trigger_alert.assert_called_once_with("Cloud Cost", 600.0, end_time, "Server")
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0]["metric_name"], "Cloud Cost")
        self.assertEqual(result2[0]["value"], 600.0)
    
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_cpu_below_threshold_resets_tracking(self, mock_get_settings):
        """
        Test that CPU usage below threshold resets duration tracking.
        
        Verifies that when CPU usage drops below threshold, the exceedance
        tracking is reset.
        """
        # Setup mock settings
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5
        mock_get_settings.return_value = mock_settings
        
        from anomaly_detection.threshold_detection import _threshold_exceedance_times
        
        # First call - CPU above threshold, start tracking
        check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        self.assertIn(("cpu_usage", "Server"), _threshold_exceedance_times)
        
        # Second call - CPU below threshold, reset tracking
        check_thresholds(cpu_usage=75.0, cloud_cost=0.0)
        self.assertNotIn(("cpu_usage", "Server"), _threshold_exceedance_times)
    
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_invalid_cpu_usage_value(self, mock_get_settings):
        """
        Test validation of CPU usage value.
        
        Verifies that ValueError is raised for invalid CPU usage values.
        """
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        # Test invalid values
        with self.assertRaises(ValueError):
            check_thresholds(cpu_usage=-10.0, cloud_cost=0.0)
        
        with self.assertRaises(ValueError):
            check_thresholds(cpu_usage=150.0, cloud_cost=0.0)
    
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_invalid_cloud_cost_value(self, mock_get_settings):
        """
        Test validation of cloud cost value.
        
        Verifies that ValueError is raised for invalid cloud cost values.
        """
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        with self.assertRaises(ValueError):
            check_thresholds(cpu_usage=50.0, cloud_cost=-100.0)
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    @patch('anomaly_detection.threshold_detection.time')
    @patch('anomaly_detection.threshold_detection.datetime')
    def test_cpu_usage_threshold(self, mock_datetime, mock_time, mock_get_settings, mock_trigger_alert):
        """
        Test that CPU usage exceeding 80% triggers an alert after 5 minutes.
        
        Verifies that when CPU usage is 85% (exceeding the 80% threshold),
        an alert is triggered after the required duration (5 minutes) is met.
        """
        # Setup mock settings
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5  # 5 minutes
        mock_get_settings.return_value = mock_settings
        
        # Mock time.time() to simulate time passing (Unix timestamps)
        # 6 minutes = 360 seconds
        start_timestamp = 1704110400.0  # Unix timestamp for 2024-01-01 12:00:00
        end_timestamp = start_timestamp + (6 * 60)  # 6 minutes later
        
        mock_time.time.side_effect = [start_timestamp, end_timestamp]
        
        # Mock datetime.utcnow() for alert timestamps
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 6, 0)  # 6 minutes later
        mock_datetime.utcnow.side_effect = [start_time, end_time]
        
        # Mock trigger_alert to return True
        mock_trigger_alert.return_value = True
        
        # First call - CPU usage 85% exceeds threshold, start tracking (no alert yet)
        result1 = check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        
        # Verify alert is not triggered yet (duration not met)
        mock_trigger_alert.assert_not_called()
        self.assertEqual(len(result1), 0)
        
        # Second call - duration met (6 minutes > 5 minutes), alert should be triggered
        result2 = check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        
        # Verify alert was triggered with correct parameters
        mock_trigger_alert.assert_called_once_with("CPU Usage", 85.0, end_time, "Server")
        
        # Verify return value contains alert information
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0]["metric_name"], "CPU Usage")
        self.assertEqual(result2[0]["value"], 85.0)
        self.assertEqual(result2[0]["resource_type"], "Server")
        self.assertEqual(result2[0]["threshold"], 80.0)
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    @patch('anomaly_detection.threshold_detection.time')
    @patch('anomaly_detection.threshold_detection.datetime')
    def test_cloud_cost_threshold(self, mock_datetime, mock_time, mock_get_settings, mock_trigger_alert):
        """
        Test that cloud cost exceeding $500 triggers an alert after 5 minutes.
        
        Verifies that when cloud cost is $600 (exceeding the $500 threshold),
        an alert is triggered after the required duration (5 minutes) is met.
        """
        # Setup mock settings
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5  # 5 minutes
        mock_get_settings.return_value = mock_settings
        
        # Mock time.time() to simulate time passing (Unix timestamps)
        # 6 minutes = 360 seconds
        start_timestamp = 1704110400.0  # Unix timestamp for 2024-01-01 12:00:00
        end_timestamp = start_timestamp + (6 * 60)  # 6 minutes later
        
        mock_time.time.side_effect = [start_timestamp, end_timestamp]
        
        # Mock datetime.utcnow() for alert timestamps
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 6, 0)  # 6 minutes later
        mock_datetime.utcnow.side_effect = [start_time, end_time]
        
        # Mock trigger_alert to return True
        mock_trigger_alert.return_value = True
        
        # First call - cloud cost $600 exceeds threshold, start tracking (no alert yet)
        result1 = check_thresholds(cpu_usage=0.0, cloud_cost=600.0)
        
        # Verify alert is not triggered yet (duration not met)
        mock_trigger_alert.assert_not_called()
        self.assertEqual(len(result1), 0)
        
        # Second call - duration met (6 minutes > 5 minutes), alert should be triggered
        result2 = check_thresholds(cpu_usage=0.0, cloud_cost=600.0)
        
        # Verify alert was triggered with correct parameters
        mock_trigger_alert.assert_called_once_with("Cloud Cost", 600.0, end_time, "Server")
        
        # Verify return value contains alert information
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2[0]["metric_name"], "Cloud Cost")
        self.assertEqual(result2[0]["value"], 600.0)
        self.assertEqual(result2[0]["resource_type"], "Server")
        self.assertEqual(result2[0]["threshold"], 500.0)


@unittest.skipUnless(THRESHOLD_DETECTION_AVAILABLE, "Alert trigger module not available")
class TestTriggerAlertFunction(unittest.TestCase):
    """Test cases for trigger_alert() function."""
    
    @patch('data_collection.database.store_alert_in_db')
    def test_trigger_alert_success(self, mock_store_alert):
        """
        Test successful alert triggering and storage.
        
        Verifies that trigger_alert() calls store_alert_in_db() with
        correct parameters and returns True on success.
        """
        # Mock database function to return True
        mock_store_alert.return_value = True
        
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        result = trigger_alert("CPU Usage", 85.5, test_timestamp, "Server")
        
        # Verify database function was called with correct parameters
        # Note: trigger_alert now includes severity and action_taken parameters
        mock_store_alert.assert_called_once()
        call_args = mock_store_alert.call_args[0]
        self.assertEqual(call_args[0], "CPU Usage")
        self.assertEqual(call_args[1], 85.5)
        self.assertEqual(call_args[2], test_timestamp)
        self.assertEqual(call_args[3], "Server")
        # Verify severity and action_taken are included
        self.assertIn(call_args[4], ["high", "medium", "low", "critical"])  # severity
        self.assertIsInstance(call_args[5], str)  # action_taken
        
        # Verify return value
        self.assertTrue(result)
    
    @patch('data_collection.database.store_alert_in_db')
    def test_trigger_alert_database_failure(self, mock_store_alert):
        """
        Test alert triggering when database storage fails.
        
        Verifies that trigger_alert() returns False when database
        storage fails.
        """
        # Mock database function to return False
        mock_store_alert.return_value = False
        
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        result = trigger_alert("CPU Usage", 85.5, test_timestamp, "Server")
        
        # Verify database function was called
        mock_store_alert.assert_called_once()
        
        # Verify return value is False (database storage failed)
        self.assertFalse(result)
    
    def test_trigger_alert_invalid_metric_name(self):
        """
        Test validation of metric_name parameter.
        
        Verifies that trigger_alert() returns False for invalid metric_name.
        """
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Test empty string
        result = trigger_alert("", 85.5, test_timestamp, "Server")
        self.assertFalse(result)
        
        # Test None
        result = trigger_alert(None, 85.5, test_timestamp, "Server")
        self.assertFalse(result)
    
    def test_trigger_alert_invalid_value(self):
        """
        Test validation of value parameter.
        
        Verifies that trigger_alert() returns False for invalid value.
        """
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Test non-numeric value
        result = trigger_alert("CPU Usage", "invalid", test_timestamp, "Server")
        self.assertFalse(result)
    
    def test_trigger_alert_invalid_timestamp(self):
        """
        Test validation of timestamp parameter.
        
        Verifies that trigger_alert() returns False for invalid timestamp.
        """
        # Test non-datetime timestamp
        result = trigger_alert("CPU Usage", 85.5, "invalid", "Server")
        self.assertFalse(result)
    
    def test_trigger_alert_invalid_resource_type(self):
        """
        Test validation of resource_type parameter.
        
        Verifies that trigger_alert() returns False for invalid resource_type.
        """
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Test empty string
        result = trigger_alert("CPU Usage", 85.5, test_timestamp, "")
        self.assertFalse(result)
        
        # Test None
        result = trigger_alert("CPU Usage", 85.5, test_timestamp, None)
        self.assertFalse(result)
    
    @patch('data_collection.database.store_alert_in_db')
    def test_trigger_alert_contains_correct_details(self, mock_store_alert):
        """
        Test that alert contains all necessary details.
        
        Verifies that trigger_alert() passes all required fields
        (metric_name, value, timestamp, resource_type) to store_alert_in_db().
        """
        mock_store_alert.return_value = True
        
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metric_name = "CPU Usage"
        value = 85.5
        resource_type = "Server"
        
        result = trigger_alert(metric_name, value, test_timestamp, resource_type)
        
        # Verify all parameters are passed correctly
        call_args = mock_store_alert.call_args[0]
        self.assertEqual(call_args[0], metric_name)
        self.assertEqual(call_args[1], value)
        self.assertEqual(call_args[2], test_timestamp)
        self.assertEqual(call_args[3], resource_type)
        
        self.assertTrue(result)


@unittest.skipUnless(THRESHOLD_DETECTION_AVAILABLE, "Threshold detection module not available")
class TestThresholdToAlertFlow(unittest.TestCase):
    """Integration tests for threshold detection to alert storage flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        from anomaly_detection.threshold_detection import _threshold_exceedance_times
        _threshold_exceedance_times.clear()
    
    @patch('data_collection.database.store_alert_in_db')
    @patch('anomaly_detection.threshold_detection.get_settings')
    @patch('anomaly_detection.threshold_detection.time')
    @patch('anomaly_detection.threshold_detection.datetime')
    def test_complete_flow_cpu_alert(self, mock_datetime, mock_time, mock_get_settings, mock_store_alert):
        """
        Test complete flow: CPU threshold exceedance -> duration tracking -> alert storage.
        
        Verifies the complete flow from threshold detection through alert storage.
        """
        # Setup mocks
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5
        mock_get_settings.return_value = mock_settings
        
        # Mock time.time() to simulate time passing (Unix timestamps)
        # 6 minutes = 360 seconds
        start_timestamp = 1704110400.0  # Unix timestamp for 2024-01-01 12:00:00
        end_timestamp = start_timestamp + (6 * 60)  # 6 minutes later
        
        mock_time.time.side_effect = [start_timestamp, end_timestamp]
        
        # Mock datetime.utcnow() for alert timestamps
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 6, 0)
        mock_datetime.utcnow.side_effect = [start_time, end_time]
        
        mock_store_alert.return_value = True
        
        # First call - start tracking
        check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        mock_store_alert.assert_not_called()
        
        # Second call - duration met, alert should be stored
        result = check_thresholds(cpu_usage=85.0, cloud_cost=0.0)
        
        # Verify alert was stored (with severity and action_taken parameters)
        mock_store_alert.assert_called_once()
        call_args = mock_store_alert.call_args[0]
        self.assertEqual(call_args[0], "CPU Usage")
        self.assertEqual(call_args[1], 85.0)
        self.assertEqual(call_args[2], end_time)
        self.assertEqual(call_args[3], "Server")
        # Verify severity and action_taken are included
        self.assertIn(call_args[4], ["high", "medium", "low", "critical"])  # severity
        self.assertIsInstance(call_args[5], str)  # action_taken
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metric_name"], "CPU Usage")
        self.assertEqual(result[0]["value"], 85.0)
        self.assertEqual(result[0]["resource_type"], "Server")
    
    @patch('data_collection.database.store_alert_in_db')
    @patch('anomaly_detection.threshold_detection.get_settings')
    @patch('anomaly_detection.threshold_detection.time')
    @patch('anomaly_detection.threshold_detection.datetime')
    def test_complete_flow_cloud_cost_alert(self, mock_datetime, mock_time, mock_get_settings, mock_store_alert):
        """
        Test complete flow: Cloud cost threshold exceedance -> duration tracking -> alert storage.
        
        Verifies the complete flow for cloud cost alerts with duration requirement.
        """
        # Setup mocks
        mock_settings = Mock()
        mock_settings.cpu_usage_threshold = 80.0
        mock_settings.cloud_cost_threshold = 500.0
        mock_settings.cpu_threshold_duration = 5
        mock_get_settings.return_value = mock_settings
        
        # Mock time.time() to simulate time passing (Unix timestamps)
        # 6 minutes = 360 seconds
        start_timestamp = 1704110400.0  # Unix timestamp for 2024-01-01 12:00:00
        end_timestamp = start_timestamp + (6 * 60)  # 6 minutes later
        
        mock_time.time.side_effect = [start_timestamp, end_timestamp]
        
        # Mock datetime.utcnow() for alert timestamps
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 6, 0)
        mock_datetime.utcnow.side_effect = [start_time, end_time]
        
        mock_store_alert.return_value = True
        
        # First call - start tracking (no alert yet)
        check_thresholds(cpu_usage=50.0, cloud_cost=600.0)
        mock_store_alert.assert_not_called()
        
        # Second call - duration met, alert should be stored
        result = check_thresholds(cpu_usage=50.0, cloud_cost=600.0)
        
        # Verify alert was stored (with severity and action_taken parameters)
        mock_store_alert.assert_called_once()
        call_args = mock_store_alert.call_args[0]
        self.assertEqual(call_args[0], "Cloud Cost")
        self.assertEqual(call_args[1], 600.0)
        self.assertEqual(call_args[2], end_time)
        self.assertEqual(call_args[3], "Server")
        # Verify severity and action_taken are included
        self.assertIn(call_args[4], ["high", "medium", "low", "critical"])  # severity
        self.assertIsInstance(call_args[5], str)  # action_taken
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metric_name"], "Cloud Cost")
        self.assertEqual(result[0]["value"], 600.0)
        self.assertEqual(result[0]["resource_type"], "Server")


@unittest.skipUnless(COST_SPIKE_DETECTION_AVAILABLE, "Cost spike detection not available")
class TestCompareCostToHistorical(unittest.TestCase):
    """Test cases for compare_cost_to_historical() function."""
    
    @patch('data_collection.database.get_historical_cost_data')
    @patch('anomaly_detection.alert_trigger.trigger_cost_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_cost_spike_detected_and_alert_triggered(self, mock_get_settings, mock_trigger_cost_alert, mock_get_historical):
        """
        Test that cost spike is detected and alert is triggered.
        
        Verifies that when current cost exceeds historical average by more than
        threshold percentage, an alert is triggered.
        """
        # Setup: historical average = 100, current = 130, threshold = 20%
        # Spike = 30% > 20% threshold, so alert should be triggered
        mock_get_historical.return_value = [90.0, 100.0, 110.0]  # Average = 100.0
        mock_get_settings.return_value = Mock(
            cost_spike_threshold_percent=20.0,
            historical_cost_lookback_days=7
        )
        mock_trigger_cost_alert.return_value = True
        
        result = compare_cost_to_historical(130.0, "AWS EC2")
        
        # Verify spike detected
        self.assertTrue(result['spike_detected'])
        self.assertEqual(result['current_cost'], 130.0)
        self.assertEqual(result['historical_average'], 100.0)
        self.assertGreater(result['spike_percentage'], 20.0)
        
        # Verify alert was triggered
        mock_trigger_cost_alert.assert_called_once()
        self.assertTrue(result['alert_triggered'])
    
    @patch('data_collection.database.get_historical_cost_data')
    @patch('anomaly_detection.alert_trigger.trigger_cost_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_no_alert_for_normal_cost(self, mock_get_settings, mock_trigger_cost_alert, mock_get_historical):
        """
        Test that no alert is triggered when cost is within threshold.
        
        Verifies that when current cost does not exceed historical average
        by threshold percentage, no alert is triggered.
        """
        # Setup: historical average = 100, current = 115, threshold = 20%
        # Spike = 15% < 20% threshold, so no alert
        mock_get_historical.return_value = [90.0, 100.0, 110.0]  # Average = 100.0
        mock_get_settings.return_value = Mock(
            cost_spike_threshold_percent=20.0,
            historical_cost_lookback_days=7
        )
        
        result = compare_cost_to_historical(115.0, "AWS EC2")
        
        # Verify no spike detected
        self.assertFalse(result['spike_detected'])
        self.assertLess(result['spike_percentage'], 20.0)
        
        # Verify alert was not triggered
        mock_trigger_cost_alert.assert_not_called()
        self.assertFalse(result['alert_triggered'])
    
    @patch('data_collection.database.get_historical_cost_data')
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_no_historical_data_returns_early(self, mock_get_settings, mock_get_historical):
        """
        Test that function returns early when no historical data exists.
        
        Verifies that when there's no historical data, the function returns
        without triggering an alert.
        """
        mock_get_historical.return_value = []  # No historical data
        mock_get_settings.return_value = Mock(
            cost_spike_threshold_percent=20.0,
            historical_cost_lookback_days=7
        )
        
        result = compare_cost_to_historical(130.0, "AWS EC2")
        
        # Verify no spike detected (no data to compare)
        self.assertFalse(result['spike_detected'])
        self.assertEqual(result['historical_average'], 0.0)
    
    @patch('data_collection.database.get_historical_cost_data')
    @patch('anomaly_detection.alert_trigger.trigger_cost_alert')
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_division_by_zero_handling(self, mock_get_settings, mock_trigger_cost_alert, mock_get_historical):
        """
        Test handling of zero historical average.
        
        Verifies that when historical average is 0, any positive cost
        is considered a spike.
        """
        mock_get_historical.return_value = [0.0, 0.0, 0.0]  # Average = 0.0
        mock_get_settings.return_value = Mock(
            cost_spike_threshold_percent=20.0,
            historical_cost_lookback_days=7
        )
        mock_trigger_cost_alert.return_value = True
        
        result = compare_cost_to_historical(10.0, "AWS EC2")
        
        # Verify spike detected when historical average is 0
        self.assertTrue(result['spike_detected'])
        self.assertEqual(result['historical_average'], 0.0)
    
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_invalid_input_validation(self, mock_get_settings):
        """
        Test input validation for compare_cost_to_historical().
        
        Verifies that ValueError is raised for invalid inputs.
        """
        mock_get_settings.return_value = Mock(
            cost_spike_threshold_percent=20.0,
            historical_cost_lookback_days=7
        )
        
        # Test negative cost
        with self.assertRaises(ValueError):
            compare_cost_to_historical(-10.0, "AWS EC2")
        
        # Test empty resource_name
        with self.assertRaises(ValueError):
            compare_cost_to_historical(100.0, "")


@unittest.skipUnless(COST_SPIKE_DETECTION_AVAILABLE, "Cost alert triggering not available")
class TestTriggerCostAlert(unittest.TestCase):
    """Test cases for trigger_cost_alert() function."""
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    def test_trigger_cost_alert_success(self, mock_trigger_alert):
        """
        Test successful cost alert triggering.
        
        Verifies that trigger_cost_alert() calls trigger_alert() with
        correct parameters and returns True on success.
        """
        mock_trigger_alert.return_value = True
        
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        result = trigger_cost_alert(600.0, "AWS EC2", test_timestamp, "Historical Average Exceeded")
        
        # Verify trigger_alert was called with correct parameters
        mock_trigger_alert.assert_called_once()
        call_args = mock_trigger_alert.call_args[0]
        self.assertIn("Cloud Cost", call_args[0])  # metric_name contains "Cloud Cost"
        self.assertEqual(call_args[1], 600.0)  # value
        self.assertEqual(call_args[2], test_timestamp)  # timestamp
        self.assertEqual(call_args[3], "AWS")  # resource_type (extracted from resource_name)
        
        # Verify return value
        self.assertTrue(result)
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    def test_trigger_cost_alert_provider_extraction(self, mock_trigger_alert):
        """
        Test that provider is correctly extracted from resource_name.
        
        Verifies that trigger_cost_alert() correctly extracts provider
        (AWS, GCP, Azure) from resource_name for resource_type.
        """
        mock_trigger_alert.return_value = True
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Test AWS
        trigger_cost_alert(600.0, "AWS EC2", test_timestamp)
        call_args = mock_trigger_alert.call_args[0]
        self.assertEqual(call_args[3], "AWS")
        
        # Reset mock for next test
        mock_trigger_alert.reset_mock()
        
        # Test GCP
        trigger_cost_alert(600.0, "GCP Compute Engine", test_timestamp)
        call_args = mock_trigger_alert.call_args[0]
        self.assertEqual(call_args[3], "GCP")
        
        # Reset mock for next test
        mock_trigger_alert.reset_mock()
        
        # Test Azure
        trigger_cost_alert(600.0, "Azure VM", test_timestamp)
        call_args = mock_trigger_alert.call_args[0]
        self.assertEqual(call_args[3], "Azure")
    
    def test_trigger_cost_alert_invalid_inputs(self):
        """
        Test input validation for trigger_cost_alert().
        
        Verifies that function returns False for invalid inputs.
        """
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Test negative cost
        result = trigger_cost_alert(-10.0, "AWS EC2", test_timestamp)
        self.assertFalse(result)
        
        # Test empty resource_name
        result = trigger_cost_alert(100.0, "", test_timestamp)
        self.assertFalse(result)
        
        # Test invalid timestamp
        result = trigger_cost_alert(100.0, "AWS EC2", "invalid")
        self.assertFalse(result)
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    def test_trigger_cost_alert_alert_reason_in_metric_name(self, mock_trigger_alert):
        """
        Test that alert_reason is included in metric_name.
        
        Verifies that the alert_reason parameter is included in the
        metric_name passed to trigger_alert().
        """
        mock_trigger_alert.return_value = True
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        trigger_cost_alert(600.0, "AWS EC2", test_timestamp, "Historical Average Exceeded")
        
        call_args = mock_trigger_alert.call_args[0]
        self.assertIn("Historical Average Exceeded", call_args[0])


@unittest.skipUnless(COST_SPIKE_DETECTION_AVAILABLE, "Cost spike detection not available")
class TestCostSpikeDetectionFlow(unittest.TestCase):
    """Integration tests for cost spike detection flow."""
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    @patch('data_collection.database.get_historical_cost_data')
    @patch('anomaly_detection.threshold_detection.get_settings')
    def test_complete_flow_cost_spike_to_alert(self, mock_get_settings, mock_get_historical, mock_trigger_alert):
        """
        Test complete flow: historical comparison -> spike detection -> alert storage.
        
        Verifies the complete flow from historical cost retrieval through
        spike detection to alert storage.
        """
        # Setup mocks
        mock_get_historical.return_value = [90.0, 100.0, 110.0]  # Average = 100.0
        mock_get_settings.return_value = Mock(
            cost_spike_threshold_percent=20.0,
            historical_cost_lookback_days=7
        )
        mock_trigger_alert.return_value = True
        
        # Call compare_cost_to_historical with spike (130 > 100 * 1.2)
        result = compare_cost_to_historical(130.0, "AWS EC2")
        
        # Verify spike detected
        self.assertTrue(result['spike_detected'])
        self.assertEqual(result['current_cost'], 130.0)
        self.assertEqual(result['historical_average'], 100.0)
        
        # Verify alert was triggered (via trigger_cost_alert -> trigger_alert)
        mock_trigger_alert.assert_called()
        self.assertTrue(result['alert_triggered'])


if __name__ == '__main__':
    unittest.main()
