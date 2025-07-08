import smtplib
import requests
import json
import logging
# from email.mime.text import MimeText
# from email.mime.multipart import MimeMultipart
# from email.mime.image import MimeImage
from datetime import datetime
from pathlib import Path
import mimetypes
import base64
from typing import Dict, Any, Optional

class NotificationService:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def update_config(self, config):
        """Update notification configuration"""
        self.config = config
        self.logger.info("Notification configuration updated")

    def send_notification(self, detection_data: Dict[str, Any]):
        """Send notification for pigeon detection"""
        if not self.config.enabled:
            return

        try:
            # Send webhook notification
            if self.config.webhook.get('enabled', False):
                self._send_webhook_notification(detection_data)

            # Send email notification
            if self.config.email.get('enabled', False):
                self._send_email_notification(detection_data)

            # Send Pushover notification
            if self.config.pushover.get('enabled', False):
                self._send_pushover_notification(detection_data)

            # Send Telegram notification
            if self.config.telegram.get('enabled', False):
                self._send_telegram_notification(detection_data)

        except Exception as e:
            self.logger.error(f"Error sending notifications: {e}")

    def _send_webhook_notification(self, detection_data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            webhook_config = self.config.webhook
            url = webhook_config.get('url')

            if not url:
                self.logger.warning("Webhook URL not configured")
                return

            payload = {
                'event': 'pigeon_detected',
                'timestamp': detection_data['timestamp'],
                'confidence': detection_data['confidence'],
                'image_path': detection_data.get('image_path'),
                'detection_count': detection_data.get('detection_count', 0),
                'frame_count': detection_data.get('frame_count', 0)
            }

            timeout = webhook_config.get('timeout', 5)
            retry_attempts = webhook_config.get('retry_attempts', 3)

            for attempt in range(retry_attempts):
                try:
                    response = requests.post(url, json=payload, timeout=timeout)
                    response.raise_for_status()

                    self.logger.info(f"Webhook notification sent successfully (attempt {attempt + 1})")
                    break

                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                    if attempt == retry_attempts - 1:
                        raise

        except Exception as e:
            self.logger.error(f"Webhook notification failed: {e}")

    def _send_email_notification(self, detection_data: Dict[str, Any]):
        """Send email notification"""
        pass
        # try:
        #     email_config = self.config.email
        #
        #     # Check required configuration
        #     required_fields = ['smtp_server', 'smtp_port', 'sender_email', 'sender_password', 'recipient_email']
        #     for field in required_fields:
        #         if not email_config.get(field):
        #             self.logger.warning(f"Email configuration missing: {field}")
        #             return
        #
        #     # Create message
        #     msg = MimeMultipart()
        #     msg['From'] = email_config['sender_email']
        #     msg['To'] = email_config['recipient_email']
        #
        #     # Format subject
        #     subject_template = email_config.get('subject_template', 'Pigeon Detected! (Confidence: {confidence:.2%})')
        #     msg['Subject'] = subject_template.format(confidence=detection_data['confidence'])
        #
        #     # Create email body
        #     body = self._create_email_body(detection_data)
        #     msg.attach(MimeText(body, 'html'))
        #
        #     # Attach image if available
        #     if detection_data.get('image_path') and Path(detection_data['image_path']).exists():
        #         self._attach_image(msg, detection_data['image_path'])
        #
        #     # Send email
        #     server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        #     server.starttls()
        #     server.login(email_config['sender_email'], email_config['sender_password'])
        #     server.send_message(msg)
        #     server.quit()
        #
        #     self.logger.info("Email notification sent successfully")
        #
        # except Exception as e:
        #     self.logger.error(f"Email notification failed: {e}")

    def _create_email_body(self, detection_data: Dict[str, Any]) -> str:
        """Create HTML email body"""
        return f"""
        <html>
        <body>
        <h2>🐦 Pigeon Detection Alert</h2>
        <p><strong>Time:</strong> {detection_data['timestamp']}</p>
        <p><strong>Confidence:</strong> {detection_data['confidence']:.2%}</p>
        <p><strong>Detection Count:</strong> {detection_data.get('detection_count', 'N/A')}</p>
        <p><strong>Frame Count:</strong> {detection_data.get('frame_count', 'N/A')}</p>
        {f'<p><strong>Image:</strong> {detection_data["image_path"]}</p>' if detection_data.get('image_path') else ''}
        <p><em>Sent from Pigeon Detection System</em></p>
        </body>
        </html>
        """

    # def _attach_image(self, msg: MimeMultipart, image_path: str):
    #     """Attach image to email"""
    #     try:
    #         with open(image_path, 'rb') as f:
    #             img_data = f.read()
    #
    #         # Determine MIME type
    #         mime_type, _ = mimetypes.guess_type(image_path)
    #         if mime_type is None:
    #             mime_type = 'image/jpeg'
    #
    #         main_type, sub_type = mime_type.split('/', 1)
    #
    #         img = MimeImage(img_data, _subtype=sub_type)
    #         img.add_header('Content-Disposition', f'attachment; filename={Path(image_path).name}')
    #         msg.attach(img)
    #
    #     except Exception as e:
    #         self.logger.error(f"Failed to attach image: {e}")

    def _send_pushover_notification(self, detection_data: Dict[str, Any]):
        """Send Pushover notification"""
        try:
            pushover_config = self.config.pushover

            user_key = pushover_config.get('user_key')
            api_token = pushover_config.get('api_token')

            if not user_key or not api_token:
                self.logger.warning("Pushover configuration incomplete")
                return

            message = f"Pigeon detected with {detection_data['confidence']:.2%} confidence at {detection_data['timestamp']}"

            payload = {
                'token': api_token,
                'user': user_key,
                'message': message,
                'title': 'Pigeon Detection Alert',
                'priority': 1,
                'sound': 'pigeon'
            }

            # Send image if available
            files = {}
            if detection_data.get('image_path') and Path(detection_data['image_path']).exists():
                files['attachment'] = open(detection_data['image_path'], 'rb')

            response = requests.post(
                'https://api.pushover.net/1/messages.json',
                data=payload,
                files=files
            )

            if files:
                files['attachment'].close()

            response.raise_for_status()
            self.logger.info("Pushover notification sent successfully")

        except Exception as e:
            self.logger.error(f"Pushover notification failed: {e}")

    def _send_telegram_notification(self, detection_data: Dict[str, Any]):
        """Send Telegram notification"""
        try:
            telegram_config = self.config.telegram

            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')

            if not bot_token or not chat_id:
                self.logger.warning("Telegram configuration incomplete")
                return

            message = f"🐦 Pigeon detected with {detection_data['confidence']:.2%} confidence at {detection_data['timestamp']}"

            # Send message
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }

            response = requests.post(url, data=payload)
            response.raise_for_status()

            # Send image if available
            if detection_data.get('image_path') and Path(detection_data['image_path']).exists():
                self._send_telegram_image(bot_token, chat_id, detection_data['image_path'])

            self.logger.info("Telegram notification sent successfully")

        except Exception as e:
            self.logger.error(f"Telegram notification failed: {e}")

    def _send_telegram_image(self, bot_token: str, chat_id: str, image_path: str):
        """Send image via Telegram"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': chat_id}

                response = requests.post(url, files=files, data=data)
                response.raise_for_status()

        except Exception as e:
            self.logger.error(f"Failed to send Telegram image: {e}")

    def test_all_notifications(self):
        """Test all configured notification methods"""
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.95,
            'detection_count': 1,
            'frame_count': 1000,
            'image_path': None
        }

        self.logger.info("Testing all notification methods...")

        if self.config.webhook.get('enabled', False):
            self.logger.info("Testing webhook notification...")
            self._send_webhook_notification(test_data)

        if self.config.email.get('enabled', False):
            self.logger.info("Testing email notification...")
            self._send_email_notification(test_data)

        if self.config.pushover.get('enabled', False):
            self.logger.info("Testing Pushover notification...")
            self._send_pushover_notification(test_data)

        if self.config.telegram.get('enabled', False):
            self.logger.info("Testing Telegram notification...")
            self._send_telegram_notification(test_data)

        self.logger.info("Notification testing completed")

    def get_notification_status(self) -> Dict[str, Any]:
        """Get status of all notification methods"""
        status = {
            'webhook': {
                'enabled': self.config.webhook.get('enabled', False),
                'configured': bool(self.config.webhook.get('url'))
            },
            'email': {
                'enabled': self.config.email.get('enabled', False),
                'configured': all([
                    self.config.email.get('smtp_server'),
                    self.config.email.get('sender_email'),
                    self.config.email.get('recipient_email')
                ])
            },
            'pushover': {
                'enabled': self.config.pushover.get('enabled', False),
                'configured': bool(self.config.pushover.get('user_key') and self.config.pushover.get('api_token'))
            },
            'telegram': {
                'enabled': self.config.telegram.get('enabled', False),
                'configured': bool(self.config.telegram.get('bot_token') and self.config.telegram.get('chat_id'))
            }
        }

        return status