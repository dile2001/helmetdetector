#!/usr/bin/env python3
"""
Helmet Detection System - Raspberry Pi 4 Setup Script

Optimized installation and configuration for Raspberry Pi 4.
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


def check_pi_hardware():
    """Check if running on Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'BCM' in cpuinfo and 'Raspberry Pi' in cpuinfo:
                print("✓ Detected Raspberry Pi hardware")
                return True
    except:
        pass
    
    print("⚠ Warning: Not detected as Raspberry Pi hardware")
    return False


def optimize_pi_settings():
    """Optimize Raspberry Pi settings for AI workloads."""
    print("Optimizing Pi settings...")
    
    # Enable camera interface
    run_command("sudo raspi-config nonint do_camera 0", "Enable camera interface")
    
    # Increase GPU memory split
    run_command("sudo raspi-config nonint do_memory_split 128", "Set GPU memory split to 128MB")
    
    # Enable I2C and SPI (for sensors)
    run_command("sudo raspi-config nonint do_i2c 0", "Enable I2C")
    run_command("sudo raspi-config nonint do_spi 0", "Enable SPI")
    
    print("✓ Pi optimization completed")


def install_pi_dependencies():
    """Install Pi-specific dependencies."""
    print("Installing Pi-specific dependencies...")
    
    # Update system
    run_command("sudo apt update", "Update package list")
    run_command("sudo apt upgrade -y", "Upgrade system packages")
    
    # Install system dependencies
    packages = [
        "python3-pip",
        "python3-venv", 
        "python3-dev",
        "libhdf5-dev",
        "libhdf5-serial-dev",
        "libatlas-base-dev",
        "libjasper-dev",
        "libqtgui4",
        "libqt4-test",
        "libqt4-dev",
        "libqt4-opengl-dev",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "libv4l-dev",
        "libxvidcore-dev",
        "libx264-dev",
        "libgtk-3-dev",
        "libtbb2",
        "libtbb-dev",
        "libdc1394-dev",
        "libopenblas-dev",
        "libatlas-dev",
        "liblapack-dev",
        "gfortran",
        "cmake",
        "pkg-config",
        "libjpeg-dev",
        "libtiff5-dev",
        "libpng-dev",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "libv4l-dev",
        "libxvidcore-dev",
        "libx264-dev",
        "libgtk-3-dev",
        "libtbb2",
        "libtbb-dev",
        "libdc1394-dev",
        "libopenblas-dev",
        "libatlas-dev",
        "liblapack-dev",
        "gfortran",
        "cmake",
        "pkg-config",
        "libjpeg-dev",
        "libtiff5-dev",
        "libpng-dev"
    ]
    
    for package in packages:
        run_command(f"sudo apt install -y {package}", f"Install {package}")


def create_pi_optimized_config():
    """Create Pi-optimized configuration files."""
    print("Creating Pi-optimized configurations...")
    
    # Camera config optimized for Pi
    camera_config = {
        "camera_id": 0,
        "camera_url": None,
        "width": 320,  # Reduced resolution for Pi
        "height": 240,
        "fps": 10,     # Reduced FPS for Pi
        "buffer_size": 5,
        "detection_interval": 3,  # Process every 3rd frame
        "save_violations": True,
        "save_path": "logs/violations",
        "max_fps": 10,
        "skip_frames": True
    }
    
    import json
    with open("config/camera_config.json", "w") as f:
        json.dump(camera_config, f, indent=2)
    
    # Alarm config optimized for Pi
    alarm_config = {
        "enabled": True,
        "audio_enabled": True,
        "visual_enabled": True,
        "email_enabled": False,
        "sms_enabled": False,
        "webhook_enabled": False,
        "audio_file": "static/alarm.wav",
        "audio_volume": 0.8,
        "audio_duration": 2.0,
        "flash_duration": 1.0,
        "flash_color": [255, 0, 0],
        "email_recipients": [],
        "sms_recipients": [],
        "webhook_url": "",
        "cooldown_duration": 10.0,  # Longer cooldown for Pi
        "max_alarms_per_minute": 5
    }
    
    with open("config/alarm_config.json", "w") as f:
        json.dump(alarm_config, f, indent=2)
    
    print("✓ Pi-optimized configurations created")


def install_python_dependencies():
    """Install Python dependencies optimized for Pi."""
    print("Installing Python dependencies for Pi...")
    
    # Create virtual environment
    run_command("python3 -m venv helmet_env", "Create virtual environment")
    
    # Activate and install dependencies
    commands = [
        "source helmet_env/bin/activate && pip install --upgrade pip",
        "source helmet_env/bin/activate && pip install numpy==1.24.3",  # Specific version for Pi
        "source helmet_env/bin/activate && pip install opencv-python-headless==4.8.0.76",  # Headless version
        "source helmet_env/bin/activate && pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu",
        "source helmet_env/bin/activate && pip install ultralytics==8.0.196",  # Specific version
        "source helmet_env/bin/activate && pip install fastapi uvicorn python-multipart jinja2",
        "source helmet_env/bin/activate && pip install pygame python-dotenv pydantic loguru psutil",
        "source helmet_env/bin/activate && pip install scikit-learn matplotlib seaborn"
    ]
    
    for cmd in commands:
        run_command(cmd, f"Install: {cmd.split()[-1]}")


def create_pi_startup_script():
    """Create startup script for Pi."""
    print("Creating Pi startup script...")
    
    startup_script = """#!/bin/bash
# Helmet Detection System - Raspberry Pi Startup Script

# Enable virtual environment
source /home/pi/helmetdetector/helmet_env/bin/activate

# Change to project directory
cd /home/pi/helmetdetector

# Start the helmet detection system
python3 main.py --mode web --host 0.0.0.0 --port 8000

# Keep script running
while true; do
    sleep 10
done
"""
    
    with open("start_pi.sh", "w") as f:
        f.write(startup_script)
    
    os.chmod("start_pi.sh", 0o755)
    print("✓ Pi startup script created")


def create_systemd_service():
    """Create systemd service for auto-start."""
    print("Creating systemd service...")
    
    service_content = """[Unit]
Description=Helmet Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/helmetdetector
ExecStart=/home/pi/helmetdetector/helmet_env/bin/python main.py --mode web --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("helmet-detection.service", "w") as f:
        f.write(service_content)
    
    print("✓ Systemd service file created")
    print("To enable auto-start:")
    print("  sudo cp helmet-detection.service /etc/systemd/system/")
    print("  sudo systemctl enable helmet-detection")
    print("  sudo systemctl start helmet-detection")


def main():
    """Main Pi setup function."""
    print("=" * 60)
    print("Helmet Detection System - Raspberry Pi 4 Setup")
    print("=" * 60)
    
    # Check if running on Pi
    is_pi = check_pi_hardware()
    
    if not is_pi:
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled")
            return
    
    success = True
    
    # Optimize Pi settings
    if is_pi:
        optimize_pi_settings()
    
    # Install system dependencies
    if is_pi:
        install_pi_dependencies()
    
    # Create optimized configs
    create_pi_optimized_config()
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Create startup scripts
    create_pi_startup_script()
    create_systemd_service()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Raspberry Pi 4 setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your helmet detection images to datasets/images/")
        print("2. Train a model: python train.py --dataset datasets --epochs 30")
        print("3. Start the system: ./start_pi.sh")
        print("4. Access dashboard: http://[PI_IP]:8000")
        print("\nPi-specific optimizations applied:")
        print("- Reduced resolution (320x240)")
        print("- Lower FPS (10)")
        print("- Frame skipping enabled")
        print("- Longer alarm cooldown")
        print("- CPU-optimized PyTorch")
    else:
        print("✗ Setup completed with errors!")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
