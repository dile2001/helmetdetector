#!/usr/bin/env python3
"""
Helmet Detection System - Installation Script

This script sets up the helmet detection system environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    # Check if pip is available
    if not run_command("pip --version", "Checking pip availability"):
        print("pip not found. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def setup_directories():
    """Create necessary directories."""
    print("Setting up directories...")
    
    directories = [
        "models",
        "datasets/images/train",
        "datasets/images/val", 
        "datasets/images/test",
        "datasets/labels/train",
        "datasets/labels/val",
        "datasets/labels/test",
        "config",
        "logs",
        "static",
        "templates",
        "logs/violations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True


def create_sample_configs():
    """Create sample configuration files."""
    print("Creating sample configuration files...")
    
    # Alarm config
    alarm_config = {
        "enabled": True,
        "audio_enabled": True,
        "visual_enabled": True,
        "email_enabled": False,
        "sms_enabled": False,
        "webhook_enabled": False,
        "audio_file": "static/alarm.wav",
        "audio_volume": 0.7,
        "audio_duration": 3.0,
        "flash_duration": 2.0,
        "flash_color": [255, 0, 0],
        "email_recipients": [],
        "sms_recipients": [],
        "webhook_url": "",
        "cooldown_duration": 5.0,
        "max_alarms_per_minute": 10
    }
    
    import json
    with open("config/alarm_config.json", "w") as f:
        json.dump(alarm_config, f, indent=2)
    print("✓ Created alarm configuration")
    
    # Camera config
    camera_config = {
        "camera_id": 0,
        "camera_url": None,
        "width": 640,
        "height": 480,
        "fps": 30,
        "buffer_size": 10,
        "detection_interval": 1,
        "save_violations": True,
        "save_path": "logs/violations",
        "max_fps": 30,
        "skip_frames": False
    }
    
    with open("config/camera_config.json", "w") as f:
        json.dump(camera_config, f, indent=2)
    print("✓ Created camera configuration")
    
    return True


def download_sample_alarm():
    """Download a sample alarm sound."""
    print("Setting up sample alarm sound...")
    
    # Create a simple beep sound using pygame
    try:
        import pygame
        import numpy as np
        
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Generate a simple beep sound
        sample_rate = 22050
        duration = 1.0
        frequency = 800
        
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2))
        
        for i in range(frames):
            arr[i][0] = 32767 * np.sin(2 * np.pi * frequency * i / sample_rate)
            arr[i][1] = 32767 * np.sin(2 * np.pi * frequency * i / sample_rate)
        
        sound = pygame.sndarray.make_sound(arr.astype(np.int16))
        pygame.mixer.Sound.save(sound, "static/alarm.wav")
        
        print("✓ Created sample alarm sound")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create alarm sound: {e}")
        return False


def create_startup_scripts():
    """Create startup scripts."""
    print("Creating startup scripts...")
    
    # Windows batch file
    windows_script = """@echo off
echo Starting Helmet Detection System...
python main.py --mode web --host 0.0.0.0 --port 8000
pause
"""
    
    with open("start_windows.bat", "w") as f:
        f.write(windows_script)
    print("✓ Created Windows startup script")
    
    # Linux/Mac shell script
    unix_script = """#!/bin/bash
echo "Starting Helmet Detection System..."
python3 main.py --mode web --host 0.0.0.0 --port 8000
"""
    
    with open("start_unix.sh", "w") as f:
        f.write(unix_script)
    
    # Make executable
    os.chmod("start_unix.sh", 0o755)
    print("✓ Created Unix startup script")
    
    return True


def main():
    """Main installation function."""
    print("=" * 60)
    print("Helmet Detection System - Installation")
    print("=" * 60)
    
    success = True
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Setup directories
    if not setup_directories():
        success = False
    
    # Create sample configs
    if not create_sample_configs():
        success = False
    
    # Download sample alarm
    if not download_sample_alarm():
        success = False
    
    # Create startup scripts
    if not create_startup_scripts():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Installation completed successfully!")
        print("\nNext steps:")
        print("1. Add your helmet detection images to datasets/images/")
        print("2. Train a model: python main.py --mode train")
        print("3. Start the system: python main.py --mode web")
        print("4. Open browser to http://localhost:8000")
    else:
        print("✗ Installation completed with errors!")
        print("Please check the error messages above and fix any issues.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
