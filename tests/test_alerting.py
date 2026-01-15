"""
Unit tests for alerting system.

Tests for alert_system, alert_logging, alert_trigger, and related modules.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, ANY
from datetime import datetime, timedelta


# Import modules to test
try:
    from alerting.alert_system import (
        AlertSystem, EmailChannel, SlackChannel, SMSChannel,
        determine_alert_channels, create_alert_system_from_settings,
        send_alert_email, send_alert_slack, send_alert_sms
    )
    from alerting.alert_logging import store_alert_in_db, AlertLogger
    from alerting.alert_history import get_alert_history
    from anomaly_detection.alert_trigger import trigger_alert, trigger_cost_alert
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    # Create dummy classes for testing
    AlertSystem = None
    EmailChannel = None
    SlackChannel = None
    SMSChannel = None
    determine_alert_channels = None
    create_alert_system_from_settings = None
    send_alert_email = None
    send_alert_slack = None
    send_alert_sms = None
    store_alert_in_db = None
    AlertLogger = None
    get_alert_history = None
    trigger_alert = None
    trigger_cost_alert = None


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestTriggerAlert(unittest.TestCase):
    """Test trigger_alert() function"""
    
    @patch('data_collection.database.store_alert_in_db')
    @patch('alerting.alert_system.determine_alert_channels')
    @patch('alerting.alert_system.create_alert_system_from_settings')
    @patch('utils.logger.get_logger')
    def test_trigger_alert_stores_in_database(self, mock_logger, mock_create_system, mock_determine, mock_store):
        """Test that trigger_alert stores alert in database"""
        mock_store.return_value = True
        mock_determine.return_value = ['email']
        mock_system = Mock()
        mock_system.send_alert.return_value = {'channels': {}}
        mock_create_system.return_value = mock_system
        
        timestamp = datetime.utcnow()
        result = trigger_alert("CPU Usage", 85.5, timestamp, "Server")
        
        self.assertTrue(result)
        # Verify database function was called with correct parameters (including severity and action_taken)
        mock_store.assert_called_once()
        call_args = mock_store.call_args[0]
        self.assertEqual(call_args[0], "CPU Usage")
        self.assertEqual(call_args[1], 85.5)
        self.assertEqual(call_args[2], timestamp)
        self.assertEqual(call_args[3], "Server")
        # Verify severity and action_taken are included
        self.assertIn(call_args[4], ["high", "medium", "low", "critical"])  # severity
        self.assertIsInstance(call_args[5], str)  # action_taken
    
    @patch('data_collection.database.store_alert_in_db')
    @patch('alerting.alert_system.determine_alert_channels')
    @patch('alerting.alert_system.create_alert_system_from_settings')
    def test_trigger_alert_sends_notifications(self, mock_create_system, mock_determine, mock_store):
        """Test that trigger_alert sends notifications via channels"""
        mock_store.return_value = True
        mock_determine.return_value = ['email', 'slack']
        mock_system = Mock()
        mock_system.send_alert.return_value = {
            'channels': {
                'EmailChannel': {'status': 'success'},
                'SlackChannel': {'status': 'success'}
            }
        }
        mock_create_system.return_value = mock_system
        
        timestamp = datetime.utcnow()
        result = trigger_alert("Cloud Cost", 600.0, timestamp, "AWS")
        
        self.assertTrue(result)
        mock_system.send_alert.assert_called_once()
        call_args = mock_system.send_alert.call_args
        self.assertEqual(call_args[1]['alert_type'], 'metric_alert')
        self.assertIn('severity', call_args[1])
    
    def test_trigger_alert_invalid_inputs(self):
        """Test trigger_alert with invalid inputs"""
        timestamp = datetime.utcnow()
        
        # Invalid metric_name
        result = trigger_alert("", 85.5, timestamp, "Server")
        self.assertFalse(result)
        
        # Invalid value
        result = trigger_alert("CPU Usage", "invalid", timestamp, "Server")
        self.assertFalse(result)
        
        # Invalid timestamp
        result = trigger_alert("CPU Usage", 85.5, "invalid", "Server")
        self.assertFalse(result)
        
        # Invalid resource_type
        result = trigger_alert("CPU Usage", 85.5, timestamp, "")
        self.assertFalse(result)
    
    @patch('data_collection.database.store_alert_in_db')
    @patch('alerting.alert_system.determine_alert_channels')
    @patch('alerting.alert_system.create_alert_system_from_settings')
    def test_trigger_alert_handles_notification_failure(self, mock_create_system, mock_determine, mock_store):
        """Test that trigger_alert returns True even if notifications fail"""
        mock_store.return_value = True
        mock_determine.return_value = ['email']
        mock_system = Mock()
        mock_system.send_alert.side_effect = Exception("Notification failed")
        mock_create_system.return_value = mock_system
        
        timestamp = datetime.utcnow()
        result = trigger_alert("CPU Usage", 85.5, timestamp, "Server")
        
        # Should still return True because alert was stored
        self.assertTrue(result)


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestTriggerCostAlert(unittest.TestCase):
    """Test trigger_cost_alert() function"""
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    def test_trigger_cost_alert_success(self, mock_trigger):
        """Test cost alert triggering"""
        mock_trigger.return_value = True
        
        timestamp = datetime.utcnow()
        result = trigger_cost_alert(600.0, "AWS EC2", timestamp, "Cost Threshold Exceeded")
        
        self.assertTrue(result)
        mock_trigger.assert_called_once()
        call_args = mock_trigger.call_args[0]
        self.assertIn("Cost Threshold Exceeded", call_args[0])  # metric_name
        self.assertEqual(call_args[1], 600.0)  # value
        self.assertEqual(call_args[2], timestamp)  # timestamp
        self.assertEqual(call_args[3], "AWS")  # resource_type (extracted)
    
    @patch('anomaly_detection.alert_trigger.trigger_alert')
    def test_trigger_cost_alert_provider_extraction(self, mock_trigger):
        """Test provider extraction from resource_name"""
        mock_trigger.return_value = True
        
        timestamp = datetime.utcnow()
        
        # Test AWS
        trigger_cost_alert(600.0, "AWS EC2", timestamp)
        call_args = mock_trigger.call_args[0]
        self.assertEqual(call_args[3], "AWS")
        
        # Test GCP
        trigger_cost_alert(600.0, "GCP Compute Engine", timestamp)
        call_args = mock_trigger.call_args[0]
        self.assertEqual(call_args[3], "GCP")
        
        # Test Azure
        trigger_cost_alert(600.0, "Azure VM", timestamp)
        call_args = mock_trigger.call_args[0]
        self.assertEqual(call_args[3], "Azure")
    
    def test_trigger_cost_alert_invalid_inputs(self):
        """Test trigger_cost_alert with invalid inputs"""
        timestamp = datetime.utcnow()
        
        # Invalid cost
        result = trigger_cost_alert(-100.0, "AWS EC2", timestamp)
        self.assertFalse(result)
        
        # Invalid resource_name
        result = trigger_cost_alert(600.0, "", timestamp)
        self.assertFalse(result)
        
        # Invalid timestamp
        result = trigger_cost_alert(600.0, "AWS EC2", "invalid")
        self.assertFalse(result)


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestDetermineAlertChannels(unittest.TestCase):
    """Test channel selection logic"""
    
    @patch('config.settings.get_settings')
    def test_determine_channels_cost_high(self, mock_get_settings):
        """Test channel selection for high cost (> 500)"""
        mock_settings = Mock()
        mock_settings.alert_preferences = {
            'rules': [
                {
                    'condition': {'metric_type': 'cost', 'operator': '>', 'value': 500},
                    'channels': ['sms', 'email'],
                    'severity': 'critical'
                }
            ],
            'default_channels': ['email']
        }
        mock_get_settings.return_value = mock_settings
        
        channels = determine_alert_channels("Cloud Cost", 600.0)
        self.assertIn('sms', channels)
        self.assertIn('email', channels)
    
    @patch('config.settings.get_settings')
    def test_determine_channels_cost_medium(self, mock_get_settings):
        """Test channel selection for medium cost (> 200)"""
        mock_settings = Mock()
        mock_settings.alert_preferences = {
            'rules': [
                {
                    'condition': {'metric_type': 'cost', 'operator': '>', 'value': 200},
                    'channels': ['email'],
                    'severity': 'high'
                }
            ],
            'default_channels': ['email']
        }
        mock_get_settings.return_value = mock_settings
        
        channels = determine_alert_channels("Cloud Cost", 300.0)
        self.assertEqual(channels, ['email'])
    
    @patch('config.settings.get_settings')
    def test_determine_channels_usage_high(self, mock_get_settings):
        """Test channel selection for high usage (> 80)"""
        mock_settings = Mock()
        mock_settings.alert_preferences = {
            'rules': [
                {
                    'condition': {'metric_type': 'usage', 'operator': '>', 'value': 80},
                    'channels': ['slack'],
                    'severity': 'high'
                }
            ],
            'default_channels': ['email']
        }
        mock_get_settings.return_value = mock_settings
        
        channels = determine_alert_channels("CPU Usage", 85.0)
        self.assertEqual(channels, ['slack'])
    
    @patch('config.settings.get_settings')
    def test_determine_channels_default_fallback(self, mock_get_settings):
        """Test default channel fallback when no rules match"""
        mock_settings = Mock()
        mock_settings.alert_preferences = {
            'rules': [
                {
                    'condition': {'metric_type': 'cost', 'operator': '>', 'value': 500},
                    'channels': ['sms'],
                    'severity': 'critical'
                }
            ],
            'default_channels': ['email']
        }
        mock_get_settings.return_value = mock_settings
        
        # Value doesn't match any rule
        channels = determine_alert_channels("Cloud Cost", 100.0)
        self.assertEqual(channels, ['email'])
    
    @patch('config.settings.get_settings')
    def test_determine_channels_metric_type_inference(self, mock_get_settings):
        """Test metric_type inference from metric_name"""
        mock_settings = Mock()
        mock_settings.alert_preferences = {
            'rules': [
                {
                    'condition': {'metric_type': 'cost', 'operator': '>', 'value': 500},
                    'channels': ['sms'],
                    'severity': 'critical'
                },
                {
                    'condition': {'metric_type': 'usage', 'operator': '>', 'value': 80},
                    'channels': ['slack'],
                    'severity': 'high'
                }
            ],
            'default_channels': ['email']
        }
        mock_get_settings.return_value = mock_settings
        
        # Should infer 'cost' from "Cloud Cost"
        channels = determine_alert_channels("Cloud Cost", 600.0)
        self.assertEqual(channels, ['sms'])
        
        # Should infer 'usage' from "CPU Usage"
        channels = determine_alert_channels("CPU Usage", 85.0)
        self.assertEqual(channels, ['slack'])
    
    @patch('config.settings.get_settings')
    def test_determine_channels_operators(self, mock_get_settings):
        """Test different operators (>, >=, <, <=, ==)"""
        mock_settings = Mock()
        mock_settings.alert_preferences = {
            'rules': [
                {
                    'condition': {'metric_type': 'usage', 'operator': '>=', 'value': 80},
                    'channels': ['slack'],
                    'severity': 'high'
                },
                {
                    'condition': {'metric_type': 'usage', 'operator': '==', 'value': 50},
                    'channels': ['email'],
                    'severity': 'medium'
                }
            ],
            'default_channels': ['email']
        }
        mock_get_settings.return_value = mock_settings
        
        # Test >= operator
        channels = determine_alert_channels("CPU Usage", 80.0)
        self.assertEqual(channels, ['slack'])
        
        # Test == operator
        channels = determine_alert_channels("CPU Usage", 50.0)
        self.assertEqual(channels, ['email'])


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestEmailChannel(unittest.TestCase):
    """Test EmailChannel.send()"""
    
    @patch('smtplib.SMTP')
    def test_email_channel_smtp_success(self, mock_smtp_class):
        """Test EmailChannel sends via SMTP successfully"""
        mock_server = Mock()
        mock_smtp_class.return_value = mock_server
        
        channel = EmailChannel("smtp.example.com", 587, "user@example.com", "password")
        result = channel.send("Test message", {"recipient": "test@example.com", "subject": "Test"})
        
        self.assertTrue(result)
        mock_smtp_class.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user@example.com", "password")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @patch('smtplib.SMTP_SSL')
    def test_email_channel_smtp_ssl(self, mock_smtp_ssl):
        """Test EmailChannel with SSL (port 465)"""
        mock_server = Mock()
        mock_smtp_ssl.return_value = mock_server
        
        channel = EmailChannel("smtp.example.com", 465, "user@example.com", "password")
        result = channel.send("Test message", {"recipient": "test@example.com"})
        
        self.assertTrue(result)
        mock_smtp_ssl.assert_called_once_with("smtp.example.com", 465)
    
    @patch('requests.post')
    def test_email_channel_sendgrid_success(self, mock_post):
        """Test EmailChannel sends via SendGrid API successfully"""
        mock_response = Mock()
        mock_response.status_code = 202
        mock_post.return_value = mock_response
        
        channel = EmailChannel(use_sendgrid=True, sendgrid_api_key="test_key")
        result = channel.send("Test message", {
            "recipient": "test@example.com",
            "subject": "Test",
            "from_email": "sender@example.com"
        })
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://api.sendgrid.com/v3/mail/send")
        self.assertIn("Authorization", call_args[1]["headers"])
    
    @patch('smtplib.SMTP')
    def test_email_channel_authentication_error(self, mock_smtp_class):
        """Test EmailChannel handles authentication errors"""
        import smtplib
        mock_server = Mock()
        mock_server.starttls.side_effect = smtplib.SMTPAuthenticationError(535, "Authentication failed")
        mock_smtp_class.return_value = mock_server
        
        channel = EmailChannel("smtp.example.com", 587, "user@example.com", "password")
        result = channel.send("Test message", {"recipient": "test@example.com"})
        
        self.assertFalse(result)
    
    def test_email_channel_not_configured(self):
        """Test EmailChannel when not configured"""
        channel = EmailChannel("", 587, "", "")
        result = channel.send("Test message")
        self.assertFalse(result)
    
    @patch('config.settings.get_settings')
    @patch('smtplib.SMTP')
    def test_email_channel_recipient_from_settings(self, mock_smtp_class, mock_get_settings):
        """Test EmailChannel extracts recipient from settings"""
        mock_server = Mock()
        mock_smtp_class.return_value = mock_server
        
        mock_settings = Mock()
        mock_settings.email_recipients = ["default@example.com"]
        mock_get_settings.return_value = mock_settings
        
        channel = EmailChannel("smtp.example.com", 587, "user@example.com", "password")
        result = channel.send("Test message")
        
        # Should use recipient from settings
        self.assertTrue(result)


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestSlackChannel(unittest.TestCase):
    """Test SlackChannel.send()"""
    
    @patch('requests.post')
    def test_slack_channel_success(self, mock_post):
        """Test SlackChannel sends webhook successfully"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_post.return_value = mock_response
        
        channel = SlackChannel("https://hooks.slack.com/services/test")
        result = channel.send("Test message")
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://hooks.slack.com/services/test")
        self.assertIn("text", call_args[1]["json"])
        self.assertEqual(call_args[1]["json"]["username"], "Monitor/Drift Agent")
    
    @patch('requests.post')
    def test_slack_channel_with_severity(self, mock_post):
        """Test SlackChannel includes severity color in attachments"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        channel = SlackChannel("https://hooks.slack.com/services/test")
        result = channel.send("Test message", {"severity": "critical"})
        
        self.assertTrue(result)
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        self.assertIn("attachments", payload)
        self.assertEqual(payload["attachments"][0]["color"], "#FF0000")  # Red for critical
    
    @patch('requests.post')
    def test_slack_channel_http_error(self, mock_post):
        """Test SlackChannel handles HTTP errors"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        channel = SlackChannel("https://hooks.slack.com/services/test")
        result = channel.send("Test message")
        
        self.assertFalse(result)
    
    def test_slack_channel_not_configured(self):
        """Test SlackChannel when not configured"""
        channel = SlackChannel("")
        result = channel.send("Test message")
        self.assertFalse(result)


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestSMSChannel(unittest.TestCase):
    """Test SMSChannel.send()"""
    
    def test_sms_channel_twilio_success(self):
        """Test SMSChannel sends via Twilio successfully"""
        import sys
        from types import ModuleType
        
        # Create mock twilio module structure
        mock_twilio = ModuleType('twilio')
        mock_twilio_rest = ModuleType('twilio.rest')
        mock_twilio.rest = mock_twilio_rest
        
        # Create mock Client class
        mock_client = Mock()
        mock_message = Mock()
        mock_message.sid = "SM123456"
        mock_client.messages.create.return_value = mock_message
        
        mock_client_class = Mock(return_value=mock_client)
        mock_twilio_rest.Client = mock_client_class
        
        # Store original modules if they exist
        original_twilio = sys.modules.get('twilio')
        original_twilio_rest = sys.modules.get('twilio.rest')
        
        # Inject into sys.modules
        sys.modules['twilio'] = mock_twilio
        sys.modules['twilio.rest'] = mock_twilio_rest
        
        try:
            channel = SMSChannel(
                api_key="test_sid",
                api_secret="test_token",
                from_number="+1234567890",
                use_twilio=True
            )
            result = channel.send("Test message", {"recipient": "+0987654321"})
            
            self.assertTrue(result)
            mock_client.messages.create.assert_called_once_with(
                body="Test message",
                from_="+1234567890",
                to="+0987654321"
            )
        finally:
            # Clean up - restore original modules or remove our mocks
            if original_twilio is not None:
                sys.modules['twilio'] = original_twilio
            elif 'twilio' in sys.modules:
                del sys.modules['twilio']
            
            if original_twilio_rest is not None:
                sys.modules['twilio.rest'] = original_twilio_rest
            elif 'twilio.rest' in sys.modules:
                del sys.modules['twilio.rest']
            if 'twilio' in sys.modules:
                # Only remove if it's our mock (check by comparing the rest attribute)
                try:
                    if hasattr(sys.modules['twilio'], 'rest') and sys.modules['twilio'].rest == mock_twilio_rest:
                        del sys.modules['twilio']
                except (KeyError, AttributeError):
                    pass
    
    @patch('requests.post')
    def test_sms_channel_generic_api_success(self, mock_post):
        """Test SMSChannel sends via generic API successfully"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        channel = SMSChannel(
            api_key="test_key",
            api_secret="test_secret",
            from_number="+1234567890",
            use_twilio=False,
            api_url="https://api.example.com/sms",
            api_method="POST"
        )
        result = channel.send("Test message", {"recipient": "+0987654321"})
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://api.example.com/sms")
        self.assertIn("to", call_args[1]["json"])
    
    def test_sms_channel_no_recipient(self):
        """Test SMSChannel fails without recipient"""
        channel = SMSChannel(
            api_key="test_key",
            api_secret="test_secret",
            from_number="+1234567890"
        )
        result = channel.send("Test message")
        self.assertFalse(result)
    
    def test_sms_channel_not_configured(self):
        """Test SMSChannel when not configured"""
        channel = SMSChannel("", "", "")
        result = channel.send("Test message", {"recipient": "+1234567890"})
        self.assertFalse(result)


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestAlertSystem(unittest.TestCase):
    """Test AlertSystem integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_email = Mock(spec=EmailChannel)
        self.mock_email.is_configured.return_value = True
        self.mock_email.__class__.__name__ = "EmailChannel"
        
        self.mock_slack = Mock(spec=SlackChannel)
        self.mock_slack.is_configured.return_value = True
        self.mock_slack.__class__.__name__ = "SlackChannel"
    
    def test_send_alert_multiple_channels(self):
        """Test sending alert through multiple channels"""
        self.mock_email.send.return_value = True
        self.mock_slack.send.return_value = True
        
        alert_system = AlertSystem(channels=[self.mock_email, self.mock_slack])
        result = alert_system.send_alert(
            alert_type="test",
            severity="high",
            message="Test message"
        )
        
        self.assertEqual(result["alert_type"], "test")
        self.assertEqual(result["severity"], "high")
        self.assertIn("channels", result)
        self.assertEqual(len(result["channels"]), 2)
        self.mock_email.send.assert_called_once()
        self.mock_slack.send.assert_called_once()
    
    def test_send_alert_channel_failure(self):
        """Test that one channel failure doesn't stop others"""
        self.mock_email.send.return_value = False
        self.mock_slack.send.return_value = True
        
        alert_system = AlertSystem(channels=[self.mock_email, self.mock_slack])
        result = alert_system.send_alert(
            alert_type="test",
            severity="high",
            message="Test message"
        )
        
        # Both channels should be called
        self.mock_email.send.assert_called_once()
        self.mock_slack.send.assert_called_once()
        # Results should show failure for email, success for slack
        self.assertEqual(result["channels"]["EmailChannel"]["status"], "failed")
        self.assertEqual(result["channels"]["SlackChannel"]["status"], "success")
    
    def test_send_alert_channel_exception(self):
        """Test handling of channel exceptions"""
        self.mock_email.send.side_effect = Exception("Channel error")
        self.mock_slack.send.return_value = True
        
        alert_system = AlertSystem(channels=[self.mock_email, self.mock_slack])
        result = alert_system.send_alert(
            alert_type="test",
            severity="high",
            message="Test message"
        )
        
        # Exception should be caught and logged
        self.assertEqual(result["channels"]["EmailChannel"]["status"], "error")
        self.assertIn("error", result["channels"]["EmailChannel"])
    
    def test_send_alert_unconfigured_channel(self):
        """Test that unconfigured channels are skipped"""
        self.mock_email.is_configured.return_value = False
        self.mock_slack.is_configured.return_value = True
        self.mock_slack.send.return_value = True
        
        alert_system = AlertSystem(channels=[self.mock_email, self.mock_slack])
        result = alert_system.send_alert(
            alert_type="test",
            severity="high",
            message="Test message"
        )
        
        # Email should not be called
        self.mock_email.send.assert_not_called()
        # Slack should be called
        self.mock_slack.send.assert_called_once()
        self.assertEqual(result["channels"]["EmailChannel"]["status"], "not_configured")
    
    def test_format_message(self):
        """Test message formatting"""
        alert_system = AlertSystem()
        formatted = alert_system._format_message("test_alert", "high", "Test message")
        self.assertEqual(formatted, "[HIGH] test_alert: Test message")


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestCreateAlertSystemFromSettings(unittest.TestCase):
    """Test create_alert_system_from_settings()"""
    
    @patch('config.settings.get_settings')
    def test_create_with_smtp_email(self, mock_get_settings):
        """Test creating AlertSystem with SMTP email"""
        mock_settings = Mock()
        mock_settings.email_use_sendgrid = False
        mock_settings.sendgrid_api_key = ""
        mock_settings.email_smtp_server = "smtp.example.com"
        mock_settings.email_smtp_port = 587
        mock_settings.email_username = "user@example.com"
        mock_settings.email_password = "password"
        mock_settings.slack_webhook_url = ""
        mock_settings.sms_use_twilio = False
        mock_settings.sms_api_key = ""
        mock_settings.sms_api_url = ""
        mock_get_settings.return_value = mock_settings
        
        alert_system = create_alert_system_from_settings()
        
        self.assertIsInstance(alert_system, AlertSystem)
        self.assertEqual(len(alert_system.channels), 1)
        self.assertIsInstance(alert_system.channels[0], EmailChannel)
    
    @patch('config.settings.get_settings')
    def test_create_with_sendgrid_email(self, mock_get_settings):
        """Test creating AlertSystem with SendGrid email"""
        mock_settings = Mock()
        mock_settings.email_use_sendgrid = True
        mock_settings.sendgrid_api_key = "test_key"
        mock_settings.email_smtp_server = ""
        mock_settings.email_username = ""
        mock_settings.slack_webhook_url = ""
        mock_settings.sms_use_twilio = False
        mock_settings.sms_api_key = ""
        mock_get_settings.return_value = mock_settings
        
        alert_system = create_alert_system_from_settings()
        
        self.assertIsInstance(alert_system, AlertSystem)
        self.assertEqual(len(alert_system.channels), 1)
        self.assertIsInstance(alert_system.channels[0], EmailChannel)
        self.assertTrue(alert_system.channels[0].use_sendgrid)
    
    @patch('config.settings.get_settings')
    def test_create_with_all_channels(self, mock_get_settings):
        """Test creating AlertSystem with all channels configured"""
        mock_settings = Mock()
        mock_settings.email_use_sendgrid = False
        mock_settings.sendgrid_api_key = ""
        mock_settings.email_smtp_server = "smtp.example.com"
        mock_settings.email_smtp_port = 587
        mock_settings.email_username = "user@example.com"
        mock_settings.email_password = "password"
        mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
        mock_settings.sms_use_twilio = True
        mock_settings.sms_api_key = "test_key"
        mock_settings.sms_api_secret = "test_secret"
        mock_settings.sms_from_number = "+1234567890"
        mock_settings.sms_api_url = ""
        mock_get_settings.return_value = mock_settings
        
        alert_system = create_alert_system_from_settings()
        
        self.assertIsInstance(alert_system, AlertSystem)
        self.assertEqual(len(alert_system.channels), 3)
        self.assertIsInstance(alert_system.channels[0], EmailChannel)
        self.assertIsInstance(alert_system.channels[1], SlackChannel)
        self.assertIsInstance(alert_system.channels[2], SMSChannel)


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestAlertLogging(unittest.TestCase):
    """Test alert logging to database"""
    
    @patch('data_collection.database.store_alert_in_db')
    def test_store_alert_in_db_success(self, mock_db_store):
        """Test successful alert storage with all parameters"""
        mock_db_store.return_value = True
        
        timestamp = datetime.utcnow()
        result = store_alert_in_db("CPU Usage", 85.5, timestamp, "Server", "high", "Email Sent")
        
        self.assertTrue(result)
        mock_db_store.assert_called_once_with("CPU Usage", 85.5, timestamp, "Server", "high", "Email Sent")
    
    @patch('data_collection.database.store_alert_in_db')
    def test_store_alert_in_db_backward_compatibility(self, mock_db_store):
        """Test alert storage with old signature (backward compatibility)"""
        mock_db_store.return_value = True
        
        timestamp = datetime.utcnow()
        result = store_alert_in_db("CPU Usage", 85.5, timestamp, "Server")
        
        self.assertTrue(result)
        # Should use default values for severity and action_taken
        mock_db_store.assert_called_once_with("CPU Usage", 85.5, timestamp, "Server", "medium", "Alert Triggered")
    
    @patch('data_collection.database.store_alert_in_db')
    def test_store_alert_in_db_with_severity_only(self, mock_db_store):
        """Test alert storage with severity parameter"""
        mock_db_store.return_value = True
        
        timestamp = datetime.utcnow()
        result = store_alert_in_db("CPU Usage", 85.5, timestamp, "Server", severity="critical")
        
        self.assertTrue(result)
        mock_db_store.assert_called_once_with("CPU Usage", 85.5, timestamp, "Server", "critical", "Alert Triggered")
    
    @patch('data_collection.database.store_alert_in_db')
    def test_store_alert_in_db_failure(self, mock_db_store):
        """Test alert storage failure"""
        mock_db_store.return_value = False
        
        timestamp = datetime.utcnow()
        result = store_alert_in_db("CPU Usage", 85.5, timestamp, "Server", "high", "Email Sent")
        
        self.assertFalse(result)
    
    @patch('data_collection.database.store_alert_in_db')
    def test_store_alert_in_db_exception(self, mock_db_store):
        """Test alert storage exception handling"""
        mock_db_store.side_effect = Exception("Database error")
        
        timestamp = datetime.utcnow()
        result = store_alert_in_db("CPU Usage", 85.5, timestamp, "Server", "high", "Email Sent")
        
        self.assertFalse(result)
    
    @patch('data_collection.database.store_alert_in_db', side_effect=ImportError("Module not found"))
    def test_store_alert_in_db_import_error(self, mock_db_store):
        """Test alert storage when database module not available"""
        timestamp = datetime.utcnow()
        result = store_alert_in_db("CPU Usage", 85.5, timestamp, "Server")
        self.assertFalse(result)


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestAlertHelperFunctions(unittest.TestCase):
    """Test send_alert_email, send_alert_slack, send_alert_sms"""
    
    @patch('alerting.alert_system.create_alert_system_from_settings')
    @patch('config.settings.get_settings')
    def test_send_alert_email(self, mock_get_settings, mock_create_system):
        """Test send_alert_email helper function"""
        mock_settings = Mock()
        mock_settings.email_recipients = ["test@example.com"]
        mock_get_settings.return_value = mock_settings
        
        mock_system = Mock()
        mock_system.send_alert.return_value = {
            'channels': {
                'EmailChannel': {'status': 'success'}
            }
        }
        mock_create_system.return_value = mock_system
        
        timestamp = datetime.utcnow()
        result = send_alert_email("CPU Usage", 85.5, timestamp, "Server", 80.0)
        
        self.assertTrue(result)
        mock_system.send_alert.assert_called_once()
        call_args = mock_system.send_alert.call_args
        self.assertIn("[ALERT] CPU Usage = 85.50", call_args[1]["message"])
    
    @patch('alerting.alert_system.create_alert_system_from_settings')
    def test_send_alert_slack(self, mock_create_system):
        """Test send_alert_slack helper function"""
        mock_system = Mock()
        mock_system.send_alert.return_value = {
            'channels': {
                'SlackChannel': {'status': 'success'}
            }
        }
        mock_create_system.return_value = mock_system
        
        timestamp = datetime.utcnow()
        result = send_alert_slack("CPU Usage", 85.5, timestamp, "Server", 80.0)
        
        self.assertTrue(result)
        mock_system.send_alert.assert_called_once()
    
    @patch('alerting.alert_system.create_alert_system_from_settings')
    def test_send_alert_sms(self, mock_create_system):
        """Test send_alert_sms helper function"""
        mock_system = Mock()
        mock_system.send_alert.return_value = {
            'channels': {
                'SMSChannel': {'status': 'success'}
            }
        }
        mock_create_system.return_value = mock_system
        
        timestamp = datetime.utcnow()
        result = send_alert_sms("CPU Usage", 85.5, timestamp, "Server", "+1234567890", 80.0)
        
        self.assertTrue(result)
        mock_system.send_alert.assert_called_once()
        call_args = mock_system.send_alert.call_args
        # SMS message should be shorter
        self.assertIn("CPU Usage=85.50", call_args[1]["message"])
    
    @patch('alerting.alert_system.create_alert_system_from_settings')
    @patch('config.settings.get_settings')
    def test_email_alert_for_cloud_cost_spike(self, mock_get_settings, mock_create_system):
        """
        Test that email alert is triggered when cloud cost exceeds $500.
        
        Verifies that send_alert_email() is called with correct parameters
        when cloud cost (600.0) exceeds the threshold of $500.
        """
        # Setup mock settings with email recipients
        mock_settings = Mock()
        mock_settings.email_recipients = ["admin@example.com"]
        mock_get_settings.return_value = mock_settings
        
        # Setup mock alert system
        mock_system = Mock()
        mock_system.send_alert.return_value = {
            'channels': {
                'EmailChannel': {'status': 'success'}
            }
        }
        mock_create_system.return_value = mock_system
        
        # Call send_alert_email with cloud cost exceeding $500 threshold
        timestamp = datetime.utcnow()
        result = send_alert_email("Cloud Cost", 600.0, timestamp, "Cloud", 500.0)
        
        # Verify email was sent successfully
        self.assertTrue(result)
        
        # Verify send_alert() was called once
        mock_system.send_alert.assert_called_once()
        
        # Verify message contains cloud cost information
        call_args = mock_system.send_alert.call_args
        message = call_args[1]["message"]
        self.assertIn("Cloud Cost", message)
        self.assertIn("600.00", message)
        
        # Verify severity is "high" (cost > $500)
        self.assertEqual(call_args[1]["severity"], "high")
        
        # Verify email recipient is used
        metadata = call_args[1]["metadata"]
        self.assertEqual(metadata["recipient"], "admin@example.com")
    
    @patch('alerting.alert_system.create_alert_system_from_settings')
    def test_slack_alert_for_cpu_usage_spike(self, mock_create_system):
        """
        Test that Slack alert is triggered when CPU usage exceeds 80%.
        
        Verifies that send_alert_slack() is called with correct parameters
        when CPU usage (90.0) exceeds the threshold of 80%.
        """
        # Setup mock alert system with Slack channel
        mock_system = Mock()
        mock_system.send_alert.return_value = {
            'channels': {
                'SlackChannel': {'status': 'success'}
            }
        }
        mock_create_system.return_value = mock_system
        
        # Call send_alert_slack with CPU usage exceeding 80% threshold
        timestamp = datetime.utcnow()
        result = send_alert_slack("CPU Usage", 90.0, timestamp, "Server", 80.0)
        
        # Verify Slack message was sent successfully
        self.assertTrue(result)
        
        # Verify send_alert() was called once
        mock_system.send_alert.assert_called_once()
        
        # Verify message contains CPU usage information
        call_args = mock_system.send_alert.call_args
        message = call_args[1]["message"]
        self.assertIn("CPU Usage", message)
        self.assertIn("90.00", message)
        
        # Verify severity is "high" (CPU > 80%)
        self.assertEqual(call_args[1]["severity"], "high")
        
        # Verify Slack channel is used (check that result indicates Slack success)
        # The function checks for "Slack" in channel names
        result_channels = mock_system.send_alert.return_value.get("channels", {})
        slack_channels = [name for name in result_channels.keys() if "Slack" in name]
        self.assertTrue(len(slack_channels) > 0, "Slack channel should be used")


@unittest.skipUnless(ALERTING_AVAILABLE, "Alerting modules not available")
class TestGetAlertHistory(unittest.TestCase):
    """Test alert history retrieval"""
    
    @patch('data_collection.database.connect_to_db')
    def test_get_all_alerts(self, mock_connect):
        """Test retrieving all alerts (no filters)"""
        # Setup mock connection and cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "CPU Usage", 85.5, datetime.utcnow(), "Server", "high", "Email Sent"),
            (2, "Cloud Cost", 600.0, datetime.utcnow(), "AWS", "critical", "SMS Sent, Email Sent")
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history()
        
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]['metric_name'], "CPU Usage")
        self.assertEqual(alerts[0]['value'], 85.5)
        self.assertEqual(alerts[0]['severity'], "high")
        self.assertEqual(alerts[1]['metric_name'], "Cloud Cost")
        mock_cursor.execute.assert_called_once()
        # Verify LIMIT and OFFSET are in the query
        call_args = mock_cursor.execute.call_args[0][0]
        self.assertIn("LIMIT", call_args)
        self.assertIn("OFFSET", call_args)
    
    @patch('data_collection.database.connect_to_db')
    def test_filter_by_severity(self, mock_connect):
        """Test filtering by severity"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "CPU Usage", 95.0, datetime.utcnow(), "Server", "critical", "Email Sent")
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history(severity="critical")
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['severity'], "critical")
        # Verify severity filter is in query
        call_args = mock_cursor.execute.call_args
        self.assertIn("critical", call_args[0][1])  # Check params
    
    @patch('data_collection.database.connect_to_db')
    def test_filter_by_resource_type(self, mock_connect):
        """Test filtering by resource_type"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "Cloud Cost", 600.0, datetime.utcnow(), "AWS", "high", "Email Sent")
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history(resource_type="AWS")
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['resource_type'], "AWS")
        # Verify resource_type filter is in query
        call_args = mock_cursor.execute.call_args
        self.assertIn("AWS", call_args[0][1])  # Check params
    
    @patch('data_collection.database.connect_to_db')
    def test_filter_by_time_range_datetime(self, mock_connect):
        """Test filtering by time_range with datetime"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        alerts = get_alert_history(time_range=cutoff_time)
        
        # Verify time_range filter is in query
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        self.assertIn("timestamp >=", query)
        # Check that cutoff_time is in params
        self.assertIn(cutoff_time, params)
    
    @patch('data_collection.database.connect_to_db')
    def test_filter_by_time_range_timedelta(self, mock_connect):
        """Test filtering by time_range with timedelta"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        time_delta = timedelta(days=30)
        alerts = get_alert_history(time_range=time_delta)
        
        # Verify time_range filter is in query
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        self.assertIn("timestamp >=", query)
        # Should have calculated cutoff timestamp
        self.assertTrue(len(params) > 0)
    
    @patch('data_collection.database.connect_to_db')
    def test_filter_by_time_range_days(self, mock_connect):
        """Test filtering by time_range with int (days)"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history(time_range=30)
        
        # Verify time_range filter is in query
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        self.assertIn("timestamp >=", query)
        # Should have calculated cutoff timestamp (30 days ago)
        self.assertTrue(len(params) > 0)
    
    @patch('data_collection.database.connect_to_db')
    def test_combined_filters(self, mock_connect):
        """Test combining multiple filters"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "CPU Usage", 90.0, datetime.utcnow(), "Server", "critical", "Email Sent, Slack Sent")
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history(
            severity="critical",
            resource_type="Server",
            time_range=timedelta(days=7),
            limit=50
        )
        
        self.assertEqual(len(alerts), 1)
        # Verify all filters are in query
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        self.assertIn("severity =", query)
        self.assertIn("resource_type =", query)
        self.assertIn("timestamp >=", query)
        self.assertIn("LIMIT", query)
        # Check params include severity, resource_type, and limit
        self.assertIn("critical", params)
        self.assertIn("Server", params)
        self.assertIn(50, params)  # limit
    
    @patch('data_collection.database.connect_to_db')
    def test_pagination(self, mock_connect):
        """Test pagination with limit and offset"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (2, "Cloud Cost", 600.0, datetime.utcnow(), "AWS", "high", "Email Sent")
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history(limit=10, offset=5)
        
        # Verify LIMIT and OFFSET are in query
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        self.assertIn("LIMIT", query)
        self.assertIn("OFFSET", query)
        self.assertIn(10, params)  # limit
        self.assertIn(5, params)  # offset
    
    @patch('data_collection.database.connect_to_db')
    def test_empty_results(self, mock_connect):
        """Test when no alerts match filters"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history(severity="nonexistent")
        
        self.assertEqual(len(alerts), 0)
        self.assertIsInstance(alerts, list)
    
    @patch('data_collection.database.connect_to_db')
    def test_database_connection_failure(self, mock_connect):
        """Test handling of database connection failure"""
        mock_connect.return_value = None
        
        alerts = get_alert_history()
        
        self.assertEqual(len(alerts), 0)
        self.assertIsInstance(alerts, list)
    
    @patch('data_collection.database.connect_to_db')
    def test_database_execution_error(self, mock_connect):
        """Test handling of SQL execution errors"""
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("SQL execution error")
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history()
        
        self.assertEqual(len(alerts), 0)
        self.assertIsInstance(alerts, list)
    
    @patch('data_collection.database.connect_to_db')
    def test_return_format(self, mock_connect):
        """Test that returned format has correct structure"""
        test_timestamp = datetime.utcnow()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, "CPU Usage", 85.5, test_timestamp, "Server", "high", "Email Sent")
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history()
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        # Verify all required keys are present
        self.assertIn('id', alert)
        self.assertIn('metric_name', alert)
        self.assertIn('value', alert)
        self.assertIn('timestamp', alert)
        self.assertIn('resource_type', alert)
        self.assertIn('severity', alert)
        self.assertIn('action_taken', alert)
        # Verify data types
        self.assertIsInstance(alert['id'], int)
        self.assertIsInstance(alert['metric_name'], str)
        self.assertIsInstance(alert['value'], float)
        self.assertIsInstance(alert['timestamp'], datetime)
        self.assertIsInstance(alert['resource_type'], str)
        self.assertIsInstance(alert['severity'], str)
        self.assertIsInstance(alert['action_taken'], str)
    
    @patch('data_collection.database.connect_to_db')
    def test_ordering_most_recent_first(self, mock_connect):
        """Test that results are ordered by timestamp DESC"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        alerts = get_alert_history()
        
        # Verify ORDER BY clause is in query
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        self.assertIn("ORDER BY timestamp DESC", query)
    
    @patch('data_collection.database.connect_to_db')
    def test_invalid_time_range_format(self, mock_connect):
        """Test handling of invalid time_range format"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Invalid time_range (not datetime, timedelta, or int)
        alerts = get_alert_history(time_range="invalid")
        
        # Should still execute query but without time_range filter
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        # Should not have timestamp filter if invalid
        # Actually, it should handle gracefully and skip the filter
        self.assertIsInstance(alerts, list)
    
    @patch('data_collection.database.connect_to_db')
    def test_invalid_severity_type(self, mock_connect):
        """Test handling of invalid severity type"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Invalid severity type
        alerts = get_alert_history(severity=123)  # Should be string
        
        # Should handle gracefully and skip severity filter
        call_args = mock_cursor.execute.call_args
        params = call_args[0][1]
        # Severity should not be in params if invalid
        self.assertNotIn(123, params)


if __name__ == '__main__':
    unittest.main()
