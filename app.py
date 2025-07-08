import os

import cv2
import numpy as np
import torch
import threading
import time
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from config_manager import ConfigManager
# from notification_service import NotificationService

class RealTimePigeonDetector:
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config_manager = ConfigManager(config_path)

        # Validate configuration
        if not self.config_manager.validate_config():
            logging.error("Configuration validation failed. Exiting.")
            sys.exit(1)

        # Get configuration objects
        self.model_config = self.config_manager.get_model_config()
        self.stream_config = self.config_manager.get_stream_config()
        self.detection_config = self.config_manager.get_detection_config()
        self.preprocessing_config = self.config_manager.get_preprocessing_config()
        self.performance_config = self.config_manager.get_performance_config()
        self.advanced_config = self.config_manager.get_advanced_config()

        # Initialize state
        self.running = False
        self.last_detection_time = 0
        self.frame_count = 0
        self.detection_count = 0

        # Initialize notification service
        self.notification_service = None
        # NotificationService(
        #     self.config_manager.get_notification_config()
        # )

        # Load model
        self.load_model()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        logging.info("RealTimePigeonDetector initialized successfully")

    def load_model(self):
        """Load the pigeon detection model"""
        pass
        # try:
        #     # Determine device
        #     if self.model_config.device == "auto":
        #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     else:
        #         self.device = torch.device(self.model_config.device)
        #
        #     logging.info(f"Loading model from {self.model_config.path} on device {self.device}")
        #
        #     # Load model
        #     self.model = torch.load(self.model_config.path, map_location=self.device)
        #     self.model.eval()
        #
        #     # Set GPU memory fraction if using CUDA
        #     if self.device.type == 'cuda':
        #         torch.cuda.set_per_process_memory_fraction(
        #             self.performance_config.gpu_memory_fraction
        #         )
        #
        #     logging.info("Model loaded successfully")
        #
        # except Exception as e:
        #     logging.error(f"Failed to load model: {e}")
        #     raise

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        try:
            # Resize frame
            frame_resized = cv2.resize(frame, self.preprocessing_config.input_size)

            # Convert BGR to RGB if needed
            if self.preprocessing_config.bgr_to_rgb:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_resized

            # Normalize if needed
            if self.preprocessing_config.normalize:
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
            else:
                frame_normalized = frame_rgb.astype(np.float32)

            # Convert to tensor
            frame_tensor = torch.FloatTensor(frame_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

            return frame_tensor

        except Exception as e:
            logging.error(f"Frame preprocessing error: {e}")
            return None

    def detect_pigeon(self, frame):
        """Detect pigeon in frame"""
        return False, 0.0
        # try:
        #     # Preprocess frame
        #     frame_tensor = self.preprocess_frame(frame)
        #     if frame_tensor is None:
        #         return False, 0.0
        #
        #     # Run inference
        #     with torch.no_grad():
        #         predictions = self.model(frame_tensor)
        #
        #     # Process predictions
        #     confidence = self.process_predictions(predictions)
        #
        #     is_pigeon = confidence > self.model_config.confidence_threshold
        #
        #     if self.advanced_config.debug_mode:
        #         logging.debug(f"Detection confidence: {confidence:.4f}, threshold: {self.model_config.confidence_threshold}")
        #
        #     return is_pigeon, confidence
        #
        # except Exception as e:
        #     logging.error(f"Detection error: {e}")
        #     return False, 0.0

    def process_predictions(self, predictions):
        """Process model predictions"""
        try:
            # Adapt this based on your model's output format
            if hasattr(predictions, 'softmax'):
                probs = torch.softmax(predictions, dim=1)
                return probs[0][1].item()  # Assuming class 1 is pigeon
            elif isinstance(predictions, torch.Tensor):
                if predictions.dim() > 1:
                    return torch.max(predictions).item()
                else:
                    return predictions.item()
            else:
                return float(predictions)

        except Exception as e:
            logging.error(f"Prediction processing error: {e}")
            return 0.0

    def save_detection_image(self, frame, confidence):
        """Save detection image to disk"""
        try:
            if not self.detection_config.save_detections:
                return None

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"pigeon_detection_{timestamp}_{confidence:.3f}.{self.detection_config.image_format}"
            filepath = Path(self.detection_config.detection_folder) / filename

            # Save with specified quality
            if self.detection_config.image_format.lower() == 'jpg':
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, self.detection_config.image_quality])
            else:
                cv2.imwrite(str(filepath), frame)

            logging.info(f"Detection image saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logging.error(f"Error saving detection image: {e}")
            return None

    def handle_detection(self, confidence, frame):
        """Handle pigeon detection"""
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_detection_time < self.detection_config.cooldown_period:
            if self.advanced_config.debug_mode:
                logging.debug(f"Detection in cooldown period, skipping notification")
            return

        self.last_detection_time = current_time
        self.detection_count += 1

        # Save detection image
        image_path = self.save_detection_image(frame, confidence)

        # Send notifications
        detection_data = {
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'image_path': image_path,
            'detection_count': self.detection_count,
            'frame_count': self.frame_count
        }

        # self.notification_service.send_notification(detection_data)

        logging.info(f"Pigeon detected! Confidence: {confidence:.2%}, Total detections: {self.detection_count}")

    def connect_to_stream(self):
        """Connect to video stream with retry logic"""
        for attempt in range(self.stream_config.reconnect_attempts):
            try:
                logging.info(f"Connecting to stream: {self.stream_config.url} (attempt {attempt + 1})")

                cap = cv2.VideoCapture(self.stream_config.url)

                if cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = cap.read()
                    if ret:
                        logging.info("Successfully connected to stream")
                        return cap
                    else:
                        cap.release()
                        logging.warning("Stream connected but cannot read frames")
                else:
                    logging.warning("Failed to open stream")

            except Exception as e:
                logging.error(f"Stream connection error: {e}")

            if attempt < self.stream_config.reconnect_attempts - 1:
                logging.info(f"Retrying in {self.stream_config.reconnect_delay} seconds...")
                time.sleep(self.stream_config.reconnect_delay)

        logging.error("Failed to connect to stream after all attempts")
        return None

    def process_stream(self):
        """Process video stream for pigeon detection"""
        cap = self.connect_to_stream()

        if cap is None:
            logging.error("Could not connect to stream")
            return

        logging.info("Starting pigeon detection processing")

        try:
            while self.running:
                ret, frame = cap.read()

                if not ret:
                    logging.warning("Failed to read frame from stream")

                    # Try to reconnect
                    cap.release()
                    cap = self.connect_to_stream()

                    if cap is None:
                        logging.error("Stream reconnection failed")
                        break
                    else:
                        continue

                self.frame_count += 1

                # Skip frames for performance
                if self.frame_count % self.stream_config.frame_skip != 0:
                    continue

                # Save all frames if debug mode is enabled
                if self.advanced_config.save_all_frames:
                    debug_path = f"debug/frame_{self.frame_count:06d}.jpg"
                    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                    cv2.imwrite(debug_path, frame)

                # Detect pigeon
                is_pigeon, confidence = self.detect_pigeon(frame)

                if is_pigeon:
                    self.handle_detection(confidence, frame)

                # Log progress periodically
                if self.frame_count % 1000 == 0:
                    logging.info(f"Processed {self.frame_count} frames, {self.detection_count} detections")

        except Exception as e:
            logging.error(f"Stream processing error: {e}")
        finally:
            cap.release()
            logging.info("Stream processing stopped")

    def start(self):
        """Start the detection process"""
        if self.running:
            logging.warning("Detection is already running")
            return

        self.running = True
        self.detection_thread = threading.Thread(target=self.process_stream, daemon=True)
        self.detection_thread.start()

        logging.info("Pigeon detection started")

    def stop(self):
        """Stop the detection process"""
        if not self.running:
            return

        logging.info("Stopping pigeon detection...")
        self.running = False

        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=5)
            if self.detection_thread.is_alive():
                logging.warning("Detection thread did not stop gracefully")

        logging.info("Pigeon detection stopped")

    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def reload_config(self):
        """Reload configuration without restarting"""
        logging.info("Reloading configuration...")
        self.config_manager.reload_config()

        # Update configuration objects
        self.model_config = self.config_manager.get_model_config()
        self.stream_config = self.config_manager.get_stream_config()
        self.detection_config = self.config_manager.get_detection_config()
        self.preprocessing_config = self.config_manager.get_preprocessing_config()

        # Update notification service
        self.notification_service.update_config(
            self.config_manager.get_notification_config()
        )

        logging.info("Configuration reloaded successfully")

    def get_status(self):
        """Get current status information"""
        return {
            'running': self.running,
            'frames_processed': self.frame_count,
            'detections_count': self.detection_count,
            'last_detection': datetime.fromtimestamp(self.last_detection_time) if self.last_detection_time > 0 else None,
            'model_path': self.model_config.path,
            'stream_url': self.stream_config.url,
            'device': str(self.device)
        }

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Pigeon Detection")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    # Create detector
    detector = RealTimePigeonDetector(args.config)

    try:
        detector.start()

        if args.daemon:
            # Run as daemon
            while True:
                time.sleep(60)  # Check every minute
                if not detector.running:
                    break
        else:
            # Interactive mode
            print("Pigeon detection started. Press 'q' to quit, 'r' to reload config, 's' for status")
            while True:
                user_input = input().strip().lower()

                if user_input == 'q':
                    break
                elif user_input == 'r':
                    detector.reload_config()
                elif user_input == 's':
                    status = detector.get_status()
                    print(f"Status: {status}")
                else:
                    print("Commands: 'q' (quit), 'r' (reload config), 's' (status)")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop()