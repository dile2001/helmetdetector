"""
Helmet Detection System - Core Detection Module

This module provides the main helmet detection functionality using YOLO models.
It can detect people and determine if they are wearing helmets or not.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class HelmetDetector:
    """
    Main helmet detection class that handles YOLO model loading and inference.
    """
    
    def __init__(self, model_path: str = "models/helmet_detector.pt", 
                 confidence_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize the helmet detector.
        
        Args:
            model_path: Path to the trained YOLO model
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.class_names = ['helmet', 'no_helmet']
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model."""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded custom model from {self.model_path}")
            else:
                # Use pre-trained YOLOv8n as fallback
                self.model = YOLO('yolov8n.pt')
                logger.warning(f"Custom model not found at {self.model_path}, using pre-trained YOLOv8n")
                logger.warning("For helmet detection, you need to train a custom model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect helmets in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, device=self.device)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Convert to center format
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [int(center_x), int(center_y)],
                            'size': [int(width), int(height)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                            'timestamp': time.time()
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_persons_and_helmets(self, image: np.ndarray) -> Dict:
        """
        Detect persons and their helmet status.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with person detections and helmet status
        """
        detections = self.detect(image)
        
        # Separate helmet and no_helmet detections
        helmet_detections = [d for d in detections if d['class_name'] == 'helmet']
        no_helmet_detections = [d for d in detections if d['class_name'] == 'no_helmet']
        
        # Count violations (people without helmets)
        violations = len(no_helmet_detections)
        
        return {
            'total_detections': len(detections),
            'helmet_detections': helmet_detections,
            'no_helmet_detections': no_helmet_detections,
            'violations': violations,
            'has_violations': violations > 0,
            'timestamp': time.time()
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on the image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        annotated_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on class
            if class_name == 'helmet':
                color = (0, 255, 0)  # Green for helmet
            elif class_name == 'no_helmet':
                color = (0, 0, 255)  # Red for no helmet
            else:
                color = (255, 0, 0)  # Blue for other classes
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def update_model(self, new_model_path: str):
        """Update the model with a new trained model."""
        self.model_path = new_model_path
        self._load_model()
        logger.info(f"Model updated to {new_model_path}")


class DetectionStats:
    """Track detection statistics over time."""
    
    def __init__(self):
        self.total_frames = 0
        self.total_violations = 0
        self.violation_history = []
        self.start_time = time.time()
    
    def update(self, detection_result: Dict):
        """Update statistics with new detection result."""
        self.total_frames += 1
        violations = detection_result.get('violations', 0)
        self.total_violations += violations
        
        if violations > 0:
            self.violation_history.append({
                'timestamp': detection_result['timestamp'],
                'violations': violations,
                'frame': self.total_frames
            })
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        runtime = time.time() - self.start_time
        violation_rate = self.total_violations / max(self.total_frames, 1)
        
        return {
            'total_frames': self.total_frames,
            'total_violations': self.total_violations,
            'violation_rate': violation_rate,
            'runtime_seconds': runtime,
            'fps': self.total_frames / max(runtime, 1),
            'recent_violations': self.violation_history[-10:] if self.violation_history else []
        }
    
    def reset(self):
        """Reset all statistics."""
        self.total_frames = 0
        self.total_violations = 0
        self.violation_history = []
        self.start_time = time.time()


if __name__ == "__main__":
    # Test the detector
    detector = HelmetDetector()
    
    # Test with a sample image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_image)
    print(f"Detections: {detections}")
