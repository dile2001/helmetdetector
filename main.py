"""
Helmet Detection System - Main Application

This is the main application file that ties everything together.
Provides command-line interface and starts the complete system.
"""

import argparse
import logging
import sys
import signal
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.helmet_detector import HelmetDetector
from src.alarm_system import AlarmSystem, AlarmConfig
from src.camera_stream import CameraStream, CameraConfig, MultiCameraManager
from src.web_interface import create_web_interface
from src.data_management import DatasetManager, ModelTrainer

logger = logging.getLogger(__name__)


class HelmetDetectionApp:
    """
    Main application class for the helmet detection system.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the helmet detection application.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.detector = None
        self.alarm_system = None
        self.camera_manager = None
        self.web_interface = None
        self.is_running = False
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/helmet_detection.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("Logging configured")
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize detector
            self.detector = HelmetDetector()
            logger.info("Helmet detector initialized")
            
            # Initialize alarm system
            alarm_config = AlarmConfig()
            self.alarm_system = AlarmSystem(alarm_config)
            logger.info("Alarm system initialized")
            
            # Initialize camera manager
            self.camera_manager = MultiCameraManager(self.detector, self.alarm_system)
            logger.info("Camera manager initialized")
            
            # Initialize web interface
            self.web_interface = create_web_interface(self.detector, self.alarm_system)
            logger.info("Web interface initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def add_camera(self, name: str, camera_id: int = 0, camera_url: str = None):
        """
        Add a camera to the system.
        
        Args:
            name: Camera name
            camera_id: Camera ID (for local cameras)
            camera_url: Camera URL (for IP cameras)
        """
        config = CameraConfig()
        config.camera_id = camera_id
        config.camera_url = camera_url
        
        self.camera_manager.add_camera(name, config)
        logger.info(f"Added camera: {name}")
    
    def start_camera(self, name: str) -> bool:
        """Start a specific camera."""
        return self.camera_manager.start_camera(name)
    
    def stop_camera(self, name: str):
        """Stop a specific camera."""
        self.camera_manager.stop_camera(name)
    
    def start_all_cameras(self):
        """Start all cameras."""
        self.camera_manager.start_all()
    
    def stop_all_cameras(self):
        """Stop all cameras."""
        self.camera_manager.stop_all()
    
    def test_alarm(self):
        """Test the alarm system."""
        self.alarm_system.test_alarm()
    
    def get_status(self) -> dict:
        """Get system status."""
        return {
            'detector': {
                'model_loaded': self.detector.model is not None,
                'model_path': self.detector.model_path,
                'confidence_threshold': self.detector.confidence_threshold
            },
            'alarm_system': self.alarm_system.get_status(),
            'cameras': self.camera_manager.get_all_stats()
        }
    
    def start(self):
        """Start the helmet detection system."""
        if self.is_running:
            logger.warning("System is already running")
            return
        
        self.is_running = True
        logger.info("Helmet Detection System started")
    
    def stop(self):
        """Stop the helmet detection system."""
        if not self.is_running:
            return
        
        logger.info("Stopping Helmet Detection System...")
        
        # Stop all cameras
        self.stop_all_cameras()
        
        self.is_running = False
        logger.info("Helmet Detection System stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Helmet Detection System")
    parser.add_argument("--mode", choices=["web", "camera", "train", "test"], 
                       default="web", help="Operation mode")
    parser.add_argument("--camera-id", type=int, default=0, 
                       help="Camera ID for camera mode")
    parser.add_argument("--camera-url", type=str, 
                       help="Camera URL for IP cameras")
    parser.add_argument("--model-path", type=str, default="models/helmet_detector.pt",
                       help="Path to trained model")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Web server host")
    parser.add_argument("--port", type=int, default=8000,
                       help="Web server port")
    parser.add_argument("--config", type=str,
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        # Create application
        app = HelmetDetectionApp(args.config)
        
        if args.mode == "web":
            # Start web interface
            app.start()
            
            # Add default camera
            app.add_camera("default", args.camera_id, args.camera_url)
            
            # Start camera
            app.start_camera("default")
            
            # Start web server
            import uvicorn
            web_app = app.web_interface.get_app()
            
            logger.info(f"Starting web server on {args.host}:{args.port}")
            uvicorn.run(web_app, host=args.host, port=args.port)
            
        elif args.mode == "camera":
            # Camera-only mode
            app.start()
            app.add_camera("default", args.camera_id, args.camera_url)
            
            if app.start_camera("default"):
                logger.info("Camera started successfully")
                
                # Keep running
                try:
                    while app.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
            else:
                logger.error("Failed to start camera")
                sys.exit(1)
        
        elif args.mode == "train":
            # Training mode
            dataset_manager = DatasetManager()
            trainer = ModelTrainer(dataset_manager)
            
            logger.info("Starting model training...")
            config_path = trainer.prepare_training()
            results = trainer.train(epochs=args.epochs)
            
            if results['success']:
                logger.info(f"Training completed successfully!")
                logger.info(f"Best model saved to: {results['best_model_path']}")
            else:
                logger.error("Training failed")
                sys.exit(1)
        
        elif args.mode == "test":
            # Test mode
            app.start()
            app.add_camera("default", args.camera_id, args.camera_url)
            
            if app.start_camera("default"):
                logger.info("Testing camera and detection...")
                
                # Test for 30 seconds
                start_time = time.time()
                while time.time() - start_time < 30:
                    stats = app.get_status()
                    camera_stats = stats['cameras'].get('default', {})
                    
                    if camera_stats.get('is_running'):
                        logger.info(f"Camera running: {camera_stats.get('total_frames', 0)} frames processed")
                    
                    time.sleep(5)
                
                logger.info("Test completed")
            else:
                logger.error("Failed to start camera for testing")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    
    finally:
        if 'app' in locals():
            app.stop()


if __name__ == "__main__":
    main()
