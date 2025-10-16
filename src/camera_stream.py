"""
Helmet Detection System - Camera Stream Processing

This module handles real-time camera feed processing for helmet detection.
Supports multiple camera sources and provides live detection capabilities.
"""

import cv2
import threading
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from queue import Queue, Empty
from pathlib import Path
import json

from .helmet_detector import HelmetDetector, DetectionStats
from .alarm_system import AlarmSystem, AlarmLogger

logger = logging.getLogger(__name__)


class CameraConfig:
    """Configuration for camera settings."""
    
    def __init__(self):
        self.camera_id = 0  # Default camera
        self.camera_url = None  # For IP cameras
        self.width = 640
        self.height = 480
        self.fps = 30
        self.buffer_size = 10
        
        # Detection settings
        self.detection_interval = 1  # Process every N frames
        self.save_violations = True
        self.save_path = "logs/violations"
        
        # Performance settings
        self.max_fps = 30
        self.skip_frames = False
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'camera_id': self.camera_id,
            'camera_url': self.camera_url,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'buffer_size': self.buffer_size,
            'detection_interval': self.detection_interval,
            'save_violations': self.save_violations,
            'save_path': self.save_path,
            'max_fps': self.max_fps,
            'skip_frames': self.skip_frames
        }
    
    def from_dict(self, config_dict: Dict):
        """Load config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class CameraStream:
    """
    Main camera stream processing class.
    Handles camera capture, detection, and alarm triggering.
    """
    
    def __init__(self, detector: HelmetDetector, alarm_system: AlarmSystem, 
                 config: Optional[CameraConfig] = None):
        """
        Initialize camera stream processor.
        
        Args:
            detector: Helmet detector instance
            alarm_system: Alarm system instance
            config: Camera configuration
        """
        self.detector = detector
        self.alarm_system = alarm_system
        self.config = config or CameraConfig()
        
        # Camera and processing
        self.camera = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=self.config.buffer_size)
        self.detection_queue = Queue(maxsize=5)
        
        # Threading
        self.capture_thread = None
        self.detection_thread = None
        self.display_thread = None
        
        # Statistics
        self.stats = DetectionStats()
        self.alarm_logger = AlarmLogger()
        
        # Callbacks
        self.frame_callbacks: List[Callable] = []
        self.detection_callbacks: List[Callable] = []
        
        # Load config
        self._load_config()
    
    def _load_config(self):
        """Load camera configuration from file."""
        config_file = Path("config/camera_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                self.config.from_dict(config_dict)
                logger.info("Camera configuration loaded from file")
            except Exception as e:
                logger.error(f"Failed to load camera config: {e}")
    
    def save_config(self):
        """Save camera configuration to file."""
        config_file = Path("config/camera_config.json")
        config_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info("Camera configuration saved")
        except Exception as e:
            logger.error(f"Failed to save camera config: {e}")
    
    def add_frame_callback(self, callback: Callable):
        """Add callback for processed frames."""
        self.frame_callbacks.append(callback)
    
    def add_detection_callback(self, callback: Callable):
        """Add callback for detection results."""
        self.detection_callbacks.append(callback)
    
    def _init_camera(self) -> bool:
        """Initialize camera capture."""
        try:
            if self.config.camera_url:
                # IP camera or video file
                self.camera = cv2.VideoCapture(self.config.camera_url)
            else:
                # Local camera
                self.camera = cv2.VideoCapture(self.config.camera_id)
            
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            logger.info(f"Camera initialized: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _capture_frames(self):
        """Capture frames from camera in separate thread."""
        frame_count = 0
        
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Skip frames if needed for performance
                if self.config.skip_frames and frame_count % 2 == 0:
                    continue
                
                # Add frame to queue
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    # Queue full, remove oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except Empty:
                        pass
                
                # Control FPS
                if self.config.max_fps > 0:
                    time.sleep(1.0 / self.config.max_fps)
                    
            except Exception as e:
                logger.error(f"Error in capture thread: {e}")
                time.sleep(0.1)
    
    def _process_detections(self):
        """Process frames for helmet detection in separate thread."""
        frame_count = 0
        
        while self.is_running:
            try:
                # Get frame from queue
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                frame_count += 1
                
                # Process every N frames
                if frame_count % self.config.detection_interval == 0:
                    # Run detection
                    detection_result = self.detector.detect_persons_and_helmets(frame)
                    
                    # Update statistics
                    self.stats.update(detection_result)
                    
                    # Trigger alarm if violations detected
                    if detection_result['has_violations']:
                        self.alarm_system.trigger_alarm(detection_result)
                        self.alarm_logger.log_alarm(detection_result)
                        
                        # Save violation frame if enabled
                        if self.config.save_violations:
                            self._save_violation_frame(frame, detection_result)
                    
                    # Add detection result to queue
                    try:
                        self.detection_queue.put_nowait({
                            'frame': frame,
                            'detection_result': detection_result,
                            'timestamp': time.time()
                        })
                    except:
                        # Queue full, remove oldest
                        try:
                            self.detection_queue.get_nowait()
                            self.detection_queue.put_nowait({
                                'frame': frame,
                                'detection_result': detection_result,
                                'timestamp': time.time()
                            })
                        except Empty:
                            pass
                    
                    # Call detection callbacks
                    for callback in self.detection_callbacks:
                        try:
                            callback(detection_result)
                        except Exception as e:
                            logger.error(f"Detection callback failed: {e}")
                
            except Exception as e:
                logger.error(f"Error in detection thread: {e}")
                time.sleep(0.1)
    
    def _save_violation_frame(self, frame: np.ndarray, detection_result: Dict):
        """Save frame with violations."""
        try:
            save_dir = Path(self.config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"violation_{timestamp}_{detection_result['violations']}violations.jpg"
            filepath = save_dir / filename
            
            # Draw detections on frame
            annotated_frame = self.detector.draw_detections(frame, 
                detection_result['helmet_detections'] + detection_result['no_helmet_detections'])
            
            cv2.imwrite(str(filepath), annotated_frame)
            logger.info(f"Violation frame saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save violation frame: {e}")
    
    def start(self) -> bool:
        """Start camera stream processing."""
        if self.is_running:
            logger.warning("Camera stream is already running")
            return True
        
        # Initialize camera
        if not self._init_camera():
            return False
        
        # Start processing
        self.is_running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.detection_thread = threading.Thread(target=self._process_detections, daemon=True)
        
        self.capture_thread.start()
        self.detection_thread.start()
        
        logger.info("Camera stream processing started")
        return True
    
    def stop(self):
        """Stop camera stream processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        logger.info("Camera stream processing stopped")
    
    def get_latest_detection(self) -> Optional[Dict]:
        """Get the latest detection result."""
        try:
            return self.detection_queue.get_nowait()
        except Empty:
            return None
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        stats = self.stats.get_stats()
        stats.update({
            'is_running': self.is_running,
            'frame_queue_size': self.frame_queue.qsize(),
            'detection_queue_size': self.detection_queue.qsize(),
            'camera_opened': self.camera.isOpened() if self.camera else False
        })
        return stats
    
    def test_camera(self) -> bool:
        """Test camera connection."""
        if not self._init_camera():
            return False
        
        ret, frame = self.camera.read()
        if ret:
            logger.info("Camera test successful")
            self.camera.release()
            return True
        else:
            logger.error("Camera test failed")
            self.camera.release()
            return False


class MultiCameraManager:
    """Manage multiple camera streams."""
    
    def __init__(self, detector: HelmetDetector, alarm_system: AlarmSystem):
        self.detector = detector
        self.alarm_system = alarm_system
        self.cameras: Dict[str, CameraStream] = {}
        self.configs: Dict[str, CameraConfig] = {}
    
    def add_camera(self, name: str, config: CameraConfig):
        """Add a camera stream."""
        camera_stream = CameraStream(self.detector, self.alarm_system, config)
        self.cameras[name] = camera_stream
        self.configs[name] = config
        logger.info(f"Added camera: {name}")
    
    def start_camera(self, name: str) -> bool:
        """Start a specific camera."""
        if name in self.cameras:
            return self.cameras[name].start()
        return False
    
    def stop_camera(self, name: str):
        """Stop a specific camera."""
        if name in self.cameras:
            self.cameras[name].stop()
    
    def start_all(self):
        """Start all cameras."""
        for name, camera in self.cameras.items():
            camera.start()
    
    def stop_all(self):
        """Stop all cameras."""
        for name, camera in self.cameras.items():
            camera.stop()
    
    def get_camera_stats(self, name: str) -> Optional[Dict]:
        """Get statistics for a specific camera."""
        if name in self.cameras:
            return self.cameras[name].get_stats()
        return None
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all cameras."""
        return {name: camera.get_stats() for name, camera in self.cameras.items()}


if __name__ == "__main__":
    # Test camera stream
    detector = HelmetDetector()
    alarm_system = AlarmSystem()
    config = CameraConfig()
    
    camera_stream = CameraStream(detector, alarm_system, config)
    
    if camera_stream.test_camera():
        print("Camera test successful")
    else:
        print("Camera test failed")
