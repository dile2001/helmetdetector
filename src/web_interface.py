"""
Helmet Detection System - Web Interface

FastAPI-based web interface for monitoring helmet detection system.
Provides real-time video feed, statistics, and configuration management.
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import json
import logging
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
import base64
import io
from PIL import Image
import time

from .helmet_detector import HelmetDetector
from .alarm_system import AlarmSystem, AlarmConfig
from .camera_stream import CameraStream, CameraConfig, MultiCameraManager

logger = logging.getLogger(__name__)


class WebInterface:
    """
    Main web interface class for the helmet detection system.
    """
    
    def __init__(self, detector: HelmetDetector, alarm_system: AlarmSystem):
        """
        Initialize web interface.
        
        Args:
            detector: Helmet detector instance
            alarm_system: Alarm system instance
        """
        self.detector = detector
        self.alarm_system = alarm_system
        self.camera_manager = MultiCameraManager(detector, alarm_system)
        
        # FastAPI app
        self.app = FastAPI(title="Helmet Detection System", version="1.0.0")
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # Setup static files and templates
        self._setup_static()
    
    def _setup_static(self):
        """Setup static files and templates."""
        # Create directories if they don't exist
        Path("static").mkdir(exist_ok=True)
        Path("templates").mkdir(exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Setup templates
        self.templates = Jinja2Templates(directory="templates")
    
    def _setup_routes(self):
        """Setup all API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {"request": request})
        
        @self.app.get("/api/status")
        async def get_status():
            """Get system status."""
            return {
                "detector": {
                    "model_loaded": self.detector.model is not None,
                    "model_path": self.detector.model_path,
                    "confidence_threshold": self.detector.confidence_threshold
                },
                "alarm_system": self.alarm_system.get_status(),
                "cameras": self.camera_manager.get_all_stats()
            }
        
        @self.app.get("/api/cameras")
        async def get_cameras():
            """Get camera information."""
            return {
                "cameras": list(self.camera_manager.cameras.keys()),
                "stats": self.camera_manager.get_all_stats()
            }
        
        @self.app.post("/api/cameras/{camera_name}/start")
        async def start_camera(camera_name: str):
            """Start a camera."""
            success = self.camera_manager.start_camera(camera_name)
            return {"success": success, "message": f"Camera {camera_name} {'started' if success else 'failed to start'}"}
        
        @self.app.post("/api/cameras/{camera_name}/stop")
        async def stop_camera(camera_name: str):
            """Stop a camera."""
            self.camera_manager.stop_camera(camera_name)
            return {"success": True, "message": f"Camera {camera_name} stopped"}
        
        @self.app.post("/api/cameras/add")
        async def add_camera(request: Request):
            """Add a new camera."""
            data = await request.json()
            camera_name = data.get("name")
            camera_config = CameraConfig()
            
            # Update config with provided values
            if "camera_id" in data:
                camera_config.camera_id = data["camera_id"]
            if "camera_url" in data:
                camera_config.camera_url = data["camera_url"]
            if "width" in data:
                camera_config.width = data["width"]
            if "height" in data:
                camera_config.height = data["height"]
            if "fps" in data:
                camera_config.fps = data["fps"]
            
            self.camera_manager.add_camera(camera_name, camera_config)
            return {"success": True, "message": f"Camera {camera_name} added"}
        
        @self.app.get("/api/alarm/test")
        async def test_alarm():
            """Test alarm system."""
            self.alarm_system.test_alarm()
            return {"success": True, "message": "Alarm test triggered"}
        
        @self.app.get("/api/alarm/config")
        async def get_alarm_config():
            """Get alarm configuration."""
            return self.alarm_system.config.to_dict()
        
        @self.app.post("/api/alarm/config")
        async def update_alarm_config(request: Request):
            """Update alarm configuration."""
            data = await request.json()
            self.alarm_system.config.from_dict(data)
            self.alarm_system.save_config()
            return {"success": True, "message": "Alarm configuration updated"}
        
        @self.app.post("/api/detect")
        async def detect_image(file: UploadFile = File(...)):
            """Detect helmets in uploaded image."""
            try:
                # Read image
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Run detection
                detection_result = self.detector.detect_persons_and_helmets(image_array)
                
                # Draw detections
                annotated_image = self.detector.draw_detections(
                    image_array, 
                    detection_result['helmet_detections'] + detection_result['no_helmet_detections']
                )
                
                # Convert back to RGB
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                # Encode image
                pil_image = Image.fromarray(annotated_image)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                return {
                    "success": True,
                    "detection_result": detection_result,
                    "annotated_image": f"data:image/jpeg;base64,{image_base64}"
                }
                
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(1)
                    
                    # Get latest detection from first camera
                    if self.camera_manager.cameras:
                        first_camera = list(self.camera_manager.cameras.values())[0]
                        latest_detection = first_camera.get_latest_detection()
                        
                        if latest_detection:
                            # Convert frame to base64
                            frame = latest_detection['frame']
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format="JPEG", quality=80)
                            frame_base64 = base64.b64encode(buffer.getvalue()).decode()
                            
                            message = {
                                "type": "detection",
                                "timestamp": latest_detection['timestamp'],
                                "detection_result": latest_detection['detection_result'],
                                "frame": f"data:image/jpeg;base64,{frame_base64}"
                            }
                            
                            await websocket.send_text(json.dumps(message))
                    
                    # Send status update
                    status = {
                        "type": "status",
                        "timestamp": time.time(),
                        "status": {
                            "cameras": self.camera_manager.get_all_stats(),
                            "alarm_system": self.alarm_system.get_status()
                        }
                    }
                    
                    await websocket.send_text(json.dumps(status))
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def broadcast_message(self, message: Dict):
        """Broadcast message to all connected WebSocket clients."""
        if self.active_connections:
            message_str = json.dumps(message)
            disconnected = []
            
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except:
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.active_connections.remove(connection)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


def create_web_interface(detector: HelmetDetector, alarm_system: AlarmSystem) -> WebInterface:
    """Create and configure web interface."""
    return WebInterface(detector, alarm_system)


if __name__ == "__main__":
    # Test web interface
    detector = HelmetDetector()
    alarm_system = AlarmSystem()
    web_interface = create_web_interface(detector, alarm_system)
    
    print("Web interface created successfully")
    print("Run with: uvicorn src.web_interface:web_interface.app --reload")
