"""
Alerting system for sending notifications.

This module provides functionality to send alerts via various channels
including email, Slack, SMS, and other notification systems.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    def send(self, message: str, metadata: Optional[Dict] = None) -> bool:
        """Send a notification."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the channel is properly configured."""
        pass


class EmailChannel(NotificationChannel):
    """Email notification channel supporting SMTP and SendGrid."""
    
    def __init__(
        self,
        smtp_server: str = "",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        use_sendgrid: bool = False,
        sendgrid_api_key: str = ""
    ):
        """
        Initialize email channel.
        
        Args:
            smtp_server: SMTP server address (for SMTP mode)
            smtp_port: SMTP server port (for SMTP mode)
            username: SMTP username (for SMTP mode)
            password: SMTP password (for SMTP mode)
            use_sendgrid: If True, use SendGrid API instead of SMTP
            sendgrid_api_key: SendGrid API key (for SendGrid mode)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_sendgrid = use_sendgrid
        self.sendgrid_api_key = sendgrid_api_key
        
        # Configured if either SMTP or SendGrid is properly set up
        if use_sendgrid:
            self.configured = bool(sendgrid_api_key)
        else:
            self.configured = bool(smtp_server and username and password)
    
    def send(self, message: str, metadata: Optional[Dict] = None) -> bool:
        """
        Send email notification using SMTP or SendGrid.
        
        Args:
            message: Alert message content
            metadata: Optional metadata containing recipient, subject, etc.
        
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.is_configured():
            return False
        
        # Get recipient from metadata or use default
        recipient = metadata.get("recipient") if metadata else None
        if not recipient:
            # Try to get from settings
            try:
                from config.settings import get_settings
                settings = get_settings()
                recipients = settings.email_recipients
                if recipients and len(recipients) > 0:
                    recipient = recipients[0]  # Use first recipient as default
            except (ImportError, AttributeError):
                pass
        
        if not recipient:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("No email recipient specified")
            except ImportError:
                print("Error: No email recipient specified")
            return False
        
        subject = metadata.get("subject", "Monitor/Drift Agent Alert") if metadata else "Monitor/Drift Agent Alert"
        
        # Use SendGrid if configured
        if self.use_sendgrid and self.sendgrid_api_key:
            return self._send_via_sendgrid(message, recipient, subject, metadata)
        else:
            return self._send_via_smtp(message, recipient, subject)
    
    def _send_via_smtp(self, message: str, recipient: str, subject: str) -> bool:
        """Send email via SMTP."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server and send
            try:
                # Try TLS first (port 587)
                if self.smtp_port == 587:
                    server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                    server.starttls()
                    server.login(self.username, self.password)
                    server.send_message(msg)
                    server.quit()
                # Try SSL (port 465)
                elif self.smtp_port == 465:
                    server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
                    server.login(self.username, self.password)
                    server.send_message(msg)
                    server.quit()
                # Default: try TLS
                else:
                    server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                    server.starttls()
                    server.login(self.username, self.password)
                    server.send_message(msg)
                    server.quit()
                
                return True
            except smtplib.SMTPAuthenticationError as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"SMTP authentication failed: {e}")
                except ImportError:
                    print(f"Error: SMTP authentication failed: {e}")
                return False
            except smtplib.SMTPException as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"SMTP error: {e}")
                except ImportError:
                    print(f"Error: SMTP error: {e}")
                return False
            except Exception as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"Error sending email via SMTP: {e}", exc_info=True)
                except ImportError:
                    print(f"Error sending email via SMTP: {e}")
                return False
        except ImportError:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("smtplib not available for email sending")
            except ImportError:
                print("Error: smtplib not available for email sending")
            return False
    
    def _send_via_sendgrid(self, message: str, recipient: str, subject: str, metadata: Optional[Dict] = None) -> bool:
        """Send email via SendGrid API."""
        try:
            import requests
            
            # SendGrid API endpoint
            url = "https://api.sendgrid.com/v3/mail/send"
            
            # Get from email from metadata or use username
            from_email = metadata.get("from_email", self.username) if metadata else self.username
            
            # Prepare payload according to SendGrid API spec
            payload = {
                "personalizations": [
                    {
                        "to": [{"email": recipient}],
                        "subject": subject
                    }
                ],
                "from": {"email": from_email},
                "content": [
                    {
                        "type": "text/plain",
                        "value": message
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.sendgrid_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 202:
                return True
            else:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"SendGrid API error: {response.status_code} - {response.text}")
                except ImportError:
                    print(f"Error: SendGrid API error: {response.status_code} - {response.text}")
                return False
        except ImportError:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("requests library not available for SendGrid")
            except ImportError:
                print("Error: requests library not available for SendGrid")
            return False
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error sending email via SendGrid: {e}", exc_info=True)
            except ImportError:
                print(f"Error sending email via SendGrid: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if email channel is configured."""
        return self.configured


class SlackChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str):
        """
        Initialize Slack channel.
        
        Args:
            webhook_url: Slack webhook URL
        
        TODO: Initialize Slack client
        """
        self.webhook_url = webhook_url
        self.configured = bool(webhook_url)
    
    def send(self, message: str, metadata: Optional[Dict] = None) -> bool:
        """
        Send Slack notification via webhook.
        
        Args:
            message: Alert message content
            metadata: Optional metadata for formatting (may include severity)
        
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.is_configured():
            return False
        
        try:
            import requests
            
            # Determine color based on severity
            severity = metadata.get("severity", "medium") if metadata else "medium"
            color_map = {
                "critical": "#FF0000",  # Red
                "high": "#FF6600",     # Orange
                "medium": "#FFAA00",   # Yellow
                "low": "#00AAFF"       # Blue
            }
            color = color_map.get(severity.lower(), "#FFAA00")
            
            # Build payload
            payload = {
                "text": message,
                "username": "Monitor/Drift Agent",
                "icon_emoji": ":warning:"
            }
            
            # Add attachments with color coding if severity is available
            if metadata and "severity" in metadata:
                payload["attachments"] = [
                    {
                        "color": color,
                        "text": message,
                        "footer": "Monitor/Drift Agent",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            
            # Send POST request to Slack webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True
            else:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"Slack webhook error: {response.status_code} - {response.text}")
                except ImportError:
                    print(f"Error: Slack webhook error: {response.status_code} - {response.text}")
                return False
        except ImportError:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("requests library not available for Slack")
            except ImportError:
                print("Error: requests library not available for Slack")
            return False
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error sending Slack notification: {e}", exc_info=True)
            except ImportError:
                print(f"Error sending Slack notification: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if Slack channel is configured."""
        return self.configured


class SMSChannel(NotificationChannel):
    """SMS notification channel supporting Twilio and generic API."""
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        from_number: str = "",
        use_twilio: bool = True,
        api_url: str = "",
        api_method: str = "POST"
    ):
        """
        Initialize SMS channel.
        
        Args:
            api_key: SMS service API key (Twilio Account SID or generic API key)
            api_secret: SMS service API secret (Twilio Auth Token or generic API secret)
            from_number: Sender phone number
            use_twilio: If True, use Twilio API; otherwise use generic HTTP API
            api_url: Generic SMS API endpoint URL (for non-Twilio mode)
            api_method: HTTP method for generic API (default: POST)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.from_number = from_number
        self.use_twilio = use_twilio
        self.api_url = api_url
        self.api_method = api_method.upper()
        
        # Configured if either Twilio or generic API is properly set up
        if use_twilio:
            self.configured = bool(api_key and api_secret and from_number)
        else:
            self.configured = bool(api_url and api_key)
    
    def send(self, message: str, metadata: Optional[Dict] = None) -> bool:
        """
        Send SMS notification via Twilio or generic API.
        
        Args:
            message: Alert message content
            metadata: Optional metadata containing recipient phone number
        
        Returns:
            True if SMS sent successfully, False otherwise
        """
        if not self.is_configured():
            return False
        
        recipient = metadata.get("recipient") if metadata else None
        if not recipient:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("No SMS recipient phone number specified")
            except ImportError:
                print("Error: No SMS recipient phone number specified")
            return False
        
        # Use Twilio if configured
        if self.use_twilio:
            return self._send_via_twilio(message, recipient)
        else:
            return self._send_via_generic_api(message, recipient, metadata)
    
    def _send_via_twilio(self, message: str, recipient: str) -> bool:
        """Send SMS via Twilio API."""
        try:
            from twilio.rest import Client
            
            client = Client(self.api_key, self.api_secret)
            twilio_message = client.messages.create(
                body=message,
                from_=self.from_number,
                to=recipient
            )
            
            return twilio_message.sid is not None
        except ImportError:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("twilio library not available for SMS sending")
            except ImportError:
                print("Error: twilio library not available for SMS sending")
            return False
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error sending SMS via Twilio: {e}", exc_info=True)
            except ImportError:
                print(f"Error sending SMS via Twilio: {e}")
            return False
    
    def _send_via_generic_api(self, message: str, recipient: str, metadata: Optional[Dict] = None) -> bool:
        """Send SMS via generic HTTP API."""
        try:
            import requests
            
            # Prepare request based on method
            if self.api_method == "POST":
                # Default payload structure (can be customized via metadata)
                payload = metadata.get("api_payload", {}) if metadata else {}
                if not payload:
                    # Default structure
                    payload = {
                        "to": recipient,
                        "message": message,
                        "from": self.from_number
                    }
                
                # Prepare headers
                headers = metadata.get("api_headers", {}) if metadata else {}
                if not headers and self.api_key:
                    # Default: use API key in Authorization header
                    headers["Authorization"] = f"Bearer {self.api_key}"
                    if self.api_secret:
                        headers["X-API-Secret"] = self.api_secret
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=10
                )
            elif self.api_method == "GET":
                # GET request with query parameters
                params = metadata.get("api_params", {}) if metadata else {}
                if not params:
                    params = {
                        "to": recipient,
                        "message": message,
                        "from": self.from_number
                    }
                if self.api_key:
                    params["api_key"] = self.api_key
                
                headers = metadata.get("api_headers", {}) if metadata else {}
                response = requests.get(
                    self.api_url,
                    params=params,
                    headers=headers,
                    timeout=10
                )
            else:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"Unsupported HTTP method: {self.api_method}")
                except ImportError:
                    print(f"Error: Unsupported HTTP method: {self.api_method}")
                return False
            
            # Check response (typically 200 or 201 for success)
            if response.status_code in [200, 201]:
                return True
            else:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"Generic SMS API error: {response.status_code} - {response.text}")
                except ImportError:
                    print(f"Error: Generic SMS API error: {response.status_code} - {response.text}")
                return False
        except ImportError:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("requests library not available for generic SMS API")
            except ImportError:
                print("Error: requests library not available for generic SMS API")
            return False
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error sending SMS via generic API: {e}", exc_info=True)
            except ImportError:
                print(f"Error sending SMS via generic API: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if SMS channel is configured."""
        return self.configured


class AlertSystem:
    """Main alerting system that manages multiple notification channels."""
    
    def __init__(self, channels: List[NotificationChannel] = None):
        """
        Initialize alert system.
        
        Args:
            channels: List of notification channels to use
        """
        self.channels = channels or []
    
    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.channels.append(channel)
    
    def remove_channel(self, channel: NotificationChannel):
        """Remove a notification channel."""
        if channel in self.channels:
            self.channels.remove(channel)
    
    def send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Send an alert through all configured channels.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity level
            message: Alert message
            metadata: Optional additional metadata
        
        Returns:
            Dictionary containing send results for each channel
        
        TODO: Implement multi-channel alert sending
        """
        results = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "channels": {}
        }
        
        # Format message with severity and type
        formatted_message = self._format_message(alert_type, severity, message)
        
        for channel in self.channels:
            if channel.is_configured():
                channel_name = channel.__class__.__name__
                try:
                    success = channel.send(formatted_message, metadata)
                    results["channels"][channel_name] = {
                        "status": "success" if success else "failed",
                        "sent_at": datetime.now().isoformat()
                    }
                except Exception as e:
                    results["channels"][channel_name] = {
                        "status": "error",
                        "error": str(e),
                        "sent_at": datetime.now().isoformat()
                    }
            else:
                channel_name = channel.__class__.__name__
                results["channels"][channel_name] = {
                    "status": "not_configured"
                }
        
        return results
    
    def _format_message(self, alert_type: str, severity: str, message: str) -> str:
        """
        Format alert message with type and severity.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
        
        Returns:
            Formatted message string
        
        TODO: Implement message formatting
        """
        return f"[{severity.upper()}] {alert_type}: {message}"


def trigger_alert(
    metric_name: str,
    value: float,
    timestamp: datetime,
    resource_type: str,
    send_notifications: bool = True,
    alert_system: Optional[AlertSystem] = None
) -> bool:
    """
    Trigger an alert and store it in the database.
    
    This function integrates with the existing trigger_alert() from
    anomaly_detection.alert_trigger to store alerts in the database,
    and optionally sends notifications via AlertSystem channels.
    
    Args:
        metric_name: Name of the metric that triggered the alert
        value: Metric value that exceeded the threshold
        timestamp: Timestamp when the alert was triggered
        resource_type: Resource type identifier (Server, AWS, GCP, Azure, Cloud)
        send_notifications: If True, send notifications via AlertSystem (default: True)
        alert_system: Optional AlertSystem instance. If None, notifications are skipped.
    
    Returns:
        True if alert stored successfully, False otherwise
    """
    try:
        # Import trigger_alert from anomaly_detection.alert_trigger
        from anomaly_detection.alert_trigger import trigger_alert as store_alert
        
        # Store alert in database
        alert_stored = store_alert(
            metric_name=metric_name,
            value=value,
            timestamp=timestamp,
            resource_type=resource_type
        )
        
        if not alert_stored:
            return False
        
        # Optionally send notifications via AlertSystem
        if send_notifications and alert_system:
            try:
                # Format alert message
                message = (
                    f"Policy violation detected: {metric_name} = {value:.2f} "
                    f"(Resource: {resource_type})"
                )
                
                # Determine severity based on violation amount (if available)
                # For now, use "high" as default
                severity = "high"
                
                # Send alert through all configured channels
                alert_system.send_alert(
                    alert_type="policy_violation",
                    severity=severity,
                    message=message,
                    metadata={
                        "metric_name": metric_name,
                        "value": value,
                        "resource_type": resource_type,
                        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
                    }
                )
            except Exception as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Failed to send alert notifications: {e}")
                except ImportError:
                    print(f"Warning: Failed to send alert notifications: {e}")
                # Don't fail if notifications fail - alert is already stored
        
        return True
    
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Alert trigger module not available for storing alerts")
        except ImportError:
            print("Error: Alert trigger module not available for storing alerts")
        return False
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error triggering alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error triggering alert: {e}")
        return False


def trigger_policy_alert(
    policy: 'ResourcePolicy',
    enforcement_result: Dict[str, Any],
    alert_system: Optional[AlertSystem] = None
) -> bool:
    """
    Trigger an alert for a policy violation.
    
    This is a specialized function for policy violations that constructs
    alert details from the policy and enforcement result.
    
    Args:
        policy: ResourcePolicy instance that was violated
        enforcement_result: Enforcement result dictionary from enforce_resource_policy()
        alert_system: Optional AlertSystem instance for sending notifications
    
    Returns:
        True if alert triggered successfully, False otherwise
    """
    try:
        from policy_management.policy_definition import ResourcePolicy
        
        # Validate inputs
        if not isinstance(policy, ResourcePolicy):
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"policy must be a ResourcePolicy instance, got {type(policy)}")
            except ImportError:
                print(f"Error: policy must be a ResourcePolicy instance, got {type(policy)}")
            return False
        
        if not isinstance(enforcement_result, dict):
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"enforcement_result must be a dictionary, got {type(enforcement_result)}")
            except ImportError:
                print(f"Error: enforcement_result must be a dictionary, got {type(enforcement_result)}")
            return False
        
        # Extract alert details
        metric_name = enforcement_result.get('resource_name', policy.resource_name)
        value = enforcement_result.get('current_value', 0.0)
        timestamp = enforcement_result.get('timestamp', datetime.utcnow())
        resource_type = enforcement_result.get('resource_type', 'Unknown')
        
        # Ensure timestamp is datetime object
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.utcnow()
        
        # Trigger alert
        return trigger_alert(
            metric_name=metric_name,
            value=float(value),
            timestamp=timestamp,
            resource_type=resource_type,
            send_notifications=True,
            alert_system=alert_system
        )
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error triggering policy alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error triggering policy alert: {e}")
        return False


def determine_alert_channels(
    metric_name: str,
    value: float,
    threshold_type: Optional[str] = None
) -> List[str]:
    """
    Determine which alert channels to use based on metric and value.
    
    This function matches the metric against alert preference rules to determine
    which channels should be used for notification.
    
    Args:
        metric_name: Name of the metric (e.g., "Cloud Cost", "CPU Usage")
        value: Current metric value
        threshold_type: Optional threshold type ("cost", "usage"). If None, inferred from metric_name
    
    Returns:
        List of channel names to use (e.g., ['email', 'sms'])
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
        prefs = settings.alert_preferences
    except (ImportError, AttributeError):
        # Fallback to default channels
        return ['email']
    
    # Infer metric_type from metric_name if not provided
    if not threshold_type:
        metric_lower = metric_name.lower()
        if 'cost' in metric_lower or 'price' in metric_lower:
            threshold_type = 'cost'
        elif 'usage' in metric_lower or 'cpu' in metric_lower or 'memory' in metric_lower:
            threshold_type = 'usage'
        else:
            # Unknown type, use default channels
            return prefs.get('default_channels', ['email'])
    
    # Match against rules
    rules = prefs.get('rules', [])
    for rule in rules:
        condition = rule.get('condition', {})
        rule_metric_type = condition.get('metric_type')
        operator = condition.get('operator', '>')
        threshold_value = condition.get('value')
        
        # Check if metric type matches
        if rule_metric_type != threshold_type:
            continue
        
        # Check if condition is met
        condition_met = False
        if operator == '>':
            condition_met = value > threshold_value
        elif operator == '>=':
            condition_met = value >= threshold_value
        elif operator == '<':
            condition_met = value < threshold_value
        elif operator == '<=':
            condition_met = value <= threshold_value
        elif operator == '==':
            condition_met = value == threshold_value
        
        if condition_met:
            # Return channels for this rule
            return rule.get('channels', prefs.get('default_channels', ['email']))
    
    # No rule matched, use default channels
    return prefs.get('default_channels', ['email'])


def create_alert_system_from_settings() -> AlertSystem:
    """
    Create an AlertSystem instance from settings configuration.
    
    This function reads settings and initializes appropriate channels
    (EmailChannel, SlackChannel, SMSChannel) based on configuration.
    
    Returns:
        AlertSystem instance with configured channels
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
    except ImportError:
        return AlertSystem(channels=[])
    
    channels = []
    
    # Initialize EmailChannel
    if settings.email_use_sendgrid and settings.sendgrid_api_key:
        email_channel = EmailChannel(
            use_sendgrid=True,
            sendgrid_api_key=settings.sendgrid_api_key
        )
        if email_channel.is_configured():
            channels.append(email_channel)
    elif settings.email_smtp_server and settings.email_username:
        email_channel = EmailChannel(
            smtp_server=settings.email_smtp_server,
            smtp_port=settings.email_smtp_port,
            username=settings.email_username,
            password=settings.email_password,
            use_sendgrid=False
        )
        if email_channel.is_configured():
            channels.append(email_channel)
    
    # Initialize SlackChannel
    if settings.slack_webhook_url:
        slack_channel = SlackChannel(webhook_url=settings.slack_webhook_url)
        if slack_channel.is_configured():
            channels.append(slack_channel)
    
    # Initialize SMSChannel
    if settings.sms_use_twilio and settings.sms_api_key and settings.sms_api_secret:
        sms_channel = SMSChannel(
            api_key=settings.sms_api_key,
            api_secret=settings.sms_api_secret,
            from_number=settings.sms_from_number,
            use_twilio=True
        )
        if sms_channel.is_configured():
            channels.append(sms_channel)
    elif settings.sms_api_url and settings.sms_api_key:
        sms_channel = SMSChannel(
            api_key=settings.sms_api_key,
            api_secret=settings.sms_api_secret,
            from_number=settings.sms_from_number,
            use_twilio=False,
            api_url=settings.sms_api_url,
            api_method=settings.sms_api_method
        )
        if sms_channel.is_configured():
            channels.append(sms_channel)
    
    return AlertSystem(channels=channels)


def send_alert_email(
    metric_name: str,
    value: float,
    timestamp: datetime,
    resource_type: str,
    threshold_value: Optional[float] = None
) -> bool:
    """
    Send alert via email channel.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        timestamp: Alert timestamp
        resource_type: Resource type
        threshold_value: Optional threshold value that was exceeded
    
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        alert_system = create_alert_system_from_settings()
        
        # Format message
        message = f"[ALERT] {metric_name} = {value:.2f}\n"
        message += f"Resource: {resource_type}\n"
        message += f"Timestamp: {timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)}\n"
        if threshold_value is not None:
            message += f"Threshold: {threshold_value:.2f}\n"
        
        # Get recipient from settings
        try:
            from config.settings import get_settings
            settings = get_settings()
            recipients = settings.email_recipients
        except (ImportError, AttributeError):
            recipients = []
        
        if not recipients:
            return False
        
        # Send to all recipients
        success = True
        for recipient in recipients:
            metadata = {
                "recipient": recipient,
                "subject": f"Alert: {metric_name}",
                "severity": "high"
            }
            result = alert_system.send_alert(
                alert_type="metric_alert",
                severity="high",
                message=message,
                metadata=metadata
            )
            # Check if at least one channel succeeded
            if not any(
                ch.get("status") == "success"
                for ch in result.get("channels", {}).values()
            ):
                success = False
        
        return success
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error sending email alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error sending email alert: {e}")
        return False


def send_alert_slack(
    metric_name: str,
    value: float,
    timestamp: datetime,
    resource_type: str,
    threshold_value: Optional[float] = None
) -> bool:
    """
    Send alert via Slack channel.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        timestamp: Alert timestamp
        resource_type: Resource type
        threshold_value: Optional threshold value that was exceeded
    
    Returns:
        True if Slack message sent successfully, False otherwise
    """
    try:
        alert_system = create_alert_system_from_settings()
        
        # Format message
        message = f"Alert: {metric_name} = {value:.2f}\n"
        message += f"Resource: {resource_type}\n"
        message += f"Timestamp: {timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)}\n"
        if threshold_value is not None:
            message += f"Threshold: {threshold_value:.2f}\n"
        
        metadata = {
            "severity": "high"
        }
        
        result = alert_system.send_alert(
            alert_type="metric_alert",
            severity="high",
            message=message,
            metadata=metadata
        )
        
        # Check if Slack channel succeeded
        return any(
            ch.get("status") == "success"
            for name, ch in result.get("channels", {}).items()
            if "Slack" in name
        )
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error sending Slack alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error sending Slack alert: {e}")
        return False


def send_alert_sms(
    metric_name: str,
    value: float,
    timestamp: datetime,
    resource_type: str,
    recipient: str,
    threshold_value: Optional[float] = None
) -> bool:
    """
    Send alert via SMS channel.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        timestamp: Alert timestamp
        resource_type: Resource type
        recipient: Phone number to send SMS to
        threshold_value: Optional threshold value that was exceeded
    
    Returns:
        True if SMS sent successfully, False otherwise
    """
    try:
        alert_system = create_alert_system_from_settings()
        
        # Format message (SMS has character limit, keep it short)
        message = f"Alert: {metric_name}={value:.2f} {resource_type}"
        
        metadata = {
            "recipient": recipient,
            "severity": "critical"
        }
        
        result = alert_system.send_alert(
            alert_type="metric_alert",
            severity="critical",
            message=message,
            metadata=metadata
        )
        
        # Check if SMS channel succeeded
        return any(
            ch.get("status") == "success"
            for name, ch in result.get("channels", {}).items()
            if "SMS" in name
        )
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error sending SMS alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error sending SMS alert: {e}")
        return False


def create_alert_system(channels: List[NotificationChannel] = None) -> AlertSystem:
    """
    Create an alert system instance.
    
    Args:
        channels: Optional list of notification channels
    
    Returns:
        AlertSystem instance
    
    TODO: Implement alert system factory
    """
    return AlertSystem(channels=channels)
