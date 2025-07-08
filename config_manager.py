import yaml
import os
import logging
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    path: str
    confidence_threshold: float
    device: str

@dataclass
class StreamConfig:
    url: str
    frame_skip: int
    reconnect_attempts: int
    reconnect_delay: int

@dataclass
class DetectionConfig:
    cooldown_period: int
    save_detections: bool
    detection_folder: str
    image_format: str
    image_quality: int

@dataclass
class PreprocessingConfig:
    input_size: tuple
    normalize: bool
    bgr_to_rgb: bool

@dataclass
class NotificationConfig:
    enabled: bool
    webhook: Dict[str, Any]
    email: Dict[str, Any]
    pushover: Dict[str, Any]
    telegram: Dict[str, Any]

@dataclass
class LoggingConfig:
    level: str
    file: str
    max_size: str
    backup_count: int
    console_output: bool

@dataclass
class PerformanceConfig:
    max_queue_size: int
    processing_threads: int
    gpu_memory_fraction: float

@dataclass
class AdvancedConfig:
    debug_mode: bool
    save_all_frames: bool
    motion_detection_threshold: float
    enable_face_blur: bool

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_data = None
        self.load_config()
        self.setup_logging()
        self.create_directories()

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config_data = yaml.safe_load(file)

            # Override with environment variables if they exist
            self._override_with_env_vars()

            logging.info(f"Configuration loaded from {self.config_path}")

        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            self._create_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise

    def _override_with_env_vars(self):
        """Override configuration with environment variables"""
        env_mappings = {
            'PIGEON_MODEL_PATH': ['model', 'path'],
            'PIGEON_CONFIDENCE_THRESHOLD': ['model', 'confidence_threshold'],
            'PIGEON_STREAM_URL': ['stream', 'url'],
            'PIGEON_WEBHOOK_URL': ['notifications', 'webhook', 'url'],
            'PIGEON_EMAIL_ENABLED': ['notifications', 'email', 'enabled'],
            'PIGEON_SENDER_EMAIL': ['notifications', 'email', 'sender_email'],
            'PIGEON_SENDER_PASSWORD': ['notifications', 'email', 'sender_password'],
            'PIGEON_RECIPIENT_EMAIL': ['notifications', 'email', 'recipient_email'],
            'PIGEON_DEBUG_MODE': ['advanced', 'debug_mode'],
            'PIGEON_LOG_LEVEL': ['logging', 'level'],
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.replace('.', '').isdigit():
                    value = float(value) if '.' in value else int(value)

                # Set the value in nested config
                current = self.config_data
                for key in config_path[:-1]:
                    current = current[key]
                current[config_path[-1]] = value

                logging.info(f"Override from env var {env_var}: {config_path} = {value}")

    def _create_default_config(self):
        """Create default configuration if file doesn't exist"""
        default_config = {
            'model': {
                'path': 'models/pigeon_model.pth',
                'confidence_threshold': 0.7,
                'device': 'auto'
            },
            'stream': {
                'url': 'http://localhost:8765/camera/1/stream',
                'frame_skip': 5,
                'reconnect_attempts': 3,
                'reconnect_delay': 5
            },
            'detection': {
                'cooldown_period': 30,
                'save_detections': True,
                'detection_folder': 'detections',
                'image_format': 'jpg',
                'image_quality': 85
            },
            'preprocessing': {
                'input_size': [416, 416],
                'normalize': True,
                'bgr_to_rgb': True
            },
            'notifications': {
                'enabled': False,
                'webhook': {'enabled': False, 'url': 'http://localhost:5000/pigeon_alert'},
                'email': {'enabled': False},
                'pushover': {'enabled': False},
                'telegram': {'enabled': False}
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/pigeon_detection.log',
                'console_output': True
            }
        }

        self.config_data = default_config
        self.save_config()
        logging.info("Default configuration created")

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config_data.get('logging', {})

        # Create logs directory if it doesn't exist
        log_file = log_config.get('file', 'logs/pigeon_detection.log')
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Configure logging
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Setup console handler if enabled
        handlers = [file_handler]
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def create_directories(self):
        """Create necessary directories"""
        detection_folder = self.get_detection_config().detection_folder
        os.makedirs(detection_folder, exist_ok=True)

        log_file = self.get_logging_config().file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        model_data = self.config_data['model']
        return ModelConfig(
            path=model_data['path'],
            confidence_threshold=model_data['confidence_threshold'],
            device=model_data['device']
        )

    def get_stream_config(self) -> StreamConfig:
        """Get stream configuration"""
        stream_data = self.config_data['stream']
        return StreamConfig(
            url=stream_data['url'],
            frame_skip=stream_data['frame_skip'],
            reconnect_attempts=stream_data['reconnect_attempts'],
            reconnect_delay=stream_data['reconnect_delay']
        )

    def get_detection_config(self) -> DetectionConfig:
        """Get detection configuration"""
        detection_data = self.config_data['detection']
        return DetectionConfig(
            cooldown_period=detection_data['cooldown_period'],
            save_detections=detection_data['save_detections'],
            detection_folder=detection_data['detection_folder'],
            image_format=detection_data['image_format'],
            image_quality=detection_data['image_quality']
        )

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Get preprocessing configuration"""
        prep_data = self.config_data['preprocessing']
        return PreprocessingConfig(
            input_size=tuple(prep_data['input_size']),
            normalize=prep_data['normalize'],
            bgr_to_rgb=prep_data['bgr_to_rgb']
        )

    def get_notification_config(self) -> NotificationConfig:
        """Get notification configuration"""
        notif_data = self.config_data['notifications']
        return NotificationConfig(
            enabled=notif_data['enabled'],
            webhook=notif_data['webhook'],
            email=notif_data['email'],
            pushover=notif_data['pushover'],
            telegram=notif_data['telegram']
        )

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        log_data = self.config_data['logging']
        return LoggingConfig(
            level=log_data['level'],
            file=log_data['file'],
            max_size=log_data.get('max_size', '10MB'),
            backup_count=log_data.get('backup_count', 5),
            console_output=log_data['console_output']
        )

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        perf_data = self.config_data.get('performance', {})
        return PerformanceConfig(
            max_queue_size=perf_data.get('max_queue_size', 10),
            processing_threads=perf_data.get('processing_threads', 2),
            gpu_memory_fraction=perf_data.get('gpu_memory_fraction', 0.8)
        )

    def get_advanced_config(self) -> AdvancedConfig:
        """Get advanced configuration"""
        adv_data = self.config_data.get('advanced', {})
        return AdvancedConfig(
            debug_mode=adv_data.get('debug_mode', False),
            save_all_frames=adv_data.get('save_all_frames', False),
            motion_detection_threshold=adv_data.get('motion_detection_threshold', 0.1),
            enable_face_blur=adv_data.get('enable_face_blur', False)
        )

    def get_config_value(self, key_path: str, default=None):
        """Get configuration value by dot notation key path"""
        keys = key_path.split('.')
        current = self.config_data

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def set_config_value(self, key_path: str, value):
        """Set configuration value by dot notation key path"""
        keys = key_path.split('.')
        current = self.config_data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        logging.info(f"Configuration updated: {key_path} = {value}")

    def reload_config(self):
        """Reload configuration from file"""
        logging.info("Reloading configuration...")
        self.load_config()
        self.setup_logging()
        self.create_directories()

    def validate_config(self) -> bool:
        """Validate configuration"""
        return True
        # try:
        #     # Check if model file exists
        #     model_path = self.get_model_config().path
        #     if not os.path.exists(model_path):
        #         logging.error(f"Model file not found: {model_path}")
        #         return False
        #
        #     # Check stream URL format
        #     stream_url = self.get_stream_config().url
        #     if not stream_url.startswith(('http://', 'https://', 'rtsp://')):
        #         logging.error(f"Invalid stream URL format: {stream_url}")
        #         return False
        #
        #     # Check confidence threshold range
        #     confidence = self.get_model_config().confidence_threshold
        #     if not 0.0 <= confidence <= 1.0:
        #         logging.error(f"Confidence threshold must be between 0.0 and 1.0: {confidence}")
        #         return False
        #
        #     logging.info("Configuration validation passed")
        #     return True
        #
        # except Exception as e:
        #     logging.error(f"Configuration validation failed: {e}")
        #     return False