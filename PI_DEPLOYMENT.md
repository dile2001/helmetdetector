# Raspberry Pi 4 Deployment Guide

This guide covers deploying the Helmet Detection System on Raspberry Pi 4 hardware.

## 🍓 **Hardware Requirements**

### **Minimum Requirements:**
- Raspberry Pi 4 (4GB RAM)
- 32GB+ MicroSD Card (Class 10)
- Pi Camera Module v2 or USB webcam
- 5V 3A USB-C power supply
- Heat sink/fan (recommended for sustained operation)

### **Recommended Setup:**
- Raspberry Pi 4 (8GB RAM)
- 64GB+ MicroSD Card (Class 10)
- Pi Camera Module v2
- Active cooling solution
- Ethernet connection (for stability)

## ⚡ **Performance Expectations**

### **Pi 4 Specifications:**
- **CPU**: ARM Cortex-A72 quad-core 1.5GHz
- **RAM**: 4GB/8GB LPDDR4
- **Expected Performance**:
  - **Detection FPS**: 5-15 FPS
  - **Resolution**: 320x240 (optimized)
  - **Model Size**: YOLOv8n (nano) recommended
  - **Processing**: Every 3rd frame

### **Optimizations Applied:**
- Reduced input resolution (320x240)
- Lower FPS (10 FPS)
- Frame skipping enabled
- CPU-optimized PyTorch
- Headless OpenCV
- Longer alarm cooldowns

## 🚀 **Installation Steps**

### **1. Prepare Raspberry Pi OS**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
```

### **2. Clone and Setup Project**

```bash
# Clone the project
git clone <your-repo-url> helmetdetector
cd helmetdetector

# Run Pi-specific setup
python3 setup_pi.py
```

### **3. Manual Installation (Alternative)**

```bash
# Install system dependencies
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libhdf5-dev libatlas-base-dev libjasper-dev
sudo apt install -y libqtgui4 libqt4-test libqt4-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libv4l-dev libxvidcore-dev libx264-dev
sudo apt install -y libgtk-3-dev libtbb2 libtbb-dev
sudo apt install -y libdc1394-dev libopenblas-dev
sudo apt install -y libatlas-dev liblapack-dev gfortran
sudo apt install -y cmake pkg-config libjpeg-dev libtiff5-dev libpng-dev

# Create virtual environment
python3 -m venv helmet_env
source helmet_env/bin/activate

# Install PyTorch CPU version (optimized for Pi)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements_pi.txt
```

## 📊 **Performance Optimization**

### **Pi-Specific Configurations:**

**Camera Settings** (`config/camera_config.json`):
```json
{
  "width": 320,
  "height": 240,
  "fps": 10,
  "detection_interval": 3,
  "skip_frames": true
}
```

**Detection Settings**:
- Use YOLOv8n (nano) model
- Confidence threshold: 0.5-0.7
- Process every 3rd frame
- Enable frame skipping

### **System Optimizations:**

```bash
# Increase GPU memory split
sudo raspi-config nonint do_memory_split 128

# Enable hardware acceleration
echo 'gpu_mem=128' | sudo tee -a /boot/config.txt

# Optimize for AI workloads
echo 'arm_freq=1800' | sudo tee -a /boot/config.txt
echo 'over_voltage=2' | sudo tee -a /boot/config.txt
```

## 🎯 **Training on Pi 4**

### **Model Training Considerations:**

**Recommended Settings:**
```bash
# Train with Pi-optimized settings
python train.py --dataset datasets --epochs 30 --batch-size 4 --imgsz 320
```

**Training Performance:**
- **Training Time**: 2-4 hours for 30 epochs
- **Batch Size**: 4 (due to memory constraints)
- **Image Size**: 320x320 (reduced for Pi)
- **Epochs**: 30-50 (sufficient for helmet detection)

### **Training Tips:**
1. Use smaller dataset initially (100-200 images)
2. Train overnight when Pi is not in use
3. Monitor temperature during training
4. Use SSD for faster I/O

## 🔧 **Camera Setup**

### **Pi Camera Module:**

```bash
# Enable camera interface
sudo raspi-config nonint do_camera 0

# Test camera
libcamera-hello --list-cameras
libcamera-vid -t 5000 -o test.h264
```

### **USB Webcam:**

```bash
# List available cameras
ls /dev/video*

# Test USB camera
ffmpeg -f v4l2 -i /dev/video0 -t 10 -f null -
```

## 🌐 **Network Configuration**

### **Static IP Setup:**

```bash
# Edit network configuration
sudo nano /etc/dhcpcd.conf

# Add static IP configuration
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1
```

### **Access Dashboard:**

Once running, access the dashboard at:
- **Local**: http://localhost:8000
- **Network**: http://[PI_IP]:8000
- **Example**: http://192.168.1.100:8000

## 🔄 **Auto-Start Configuration**

### **Systemd Service:**

```bash
# Copy service file
sudo cp helmet-detection.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable helmet-detection
sudo systemctl start helmet-detection

# Check status
sudo systemctl status helmet-detection
```

### **Manual Startup:**

```bash
# Start system manually
./start_pi.sh

# Or use main script
source helmet_env/bin/activate
python main.py --mode web --host 0.0.0.0 --port 8000
```

## 📈 **Monitoring and Maintenance**

### **Performance Monitoring:**

```bash
# Monitor CPU usage
htop

# Monitor temperature
vcgencmd measure_temp

# Monitor memory
free -h

# Monitor disk usage
df -h
```

### **Log Files:**

```bash
# View system logs
sudo journalctl -u helmet-detection -f

# View application logs
tail -f logs/helmet_detection.log

# View alarm logs
tail -f logs/alarms.log
```

## 🚨 **Troubleshooting**

### **Common Issues:**

**1. Camera Not Detected:**
```bash
# Check camera interface
sudo raspi-config nonint get_camera

# Enable camera
sudo raspi-config nonint do_camera 0
```

**2. Low Performance:**
- Reduce image resolution
- Increase detection interval
- Enable frame skipping
- Check temperature

**3. Memory Issues:**
- Reduce batch size
- Use smaller model
- Close unnecessary processes

**4. Network Issues:**
- Check static IP configuration
- Verify firewall settings
- Test with Ethernet connection

### **Performance Tuning:**

```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Optimize GPU memory
sudo raspi-config nonint do_memory_split 128
```

## 📱 **Mobile Access**

### **Access from Mobile Devices:**

1. Connect Pi to same network
2. Find Pi IP address: `hostname -I`
3. Open browser on mobile: `http://[PI_IP]:8000`
4. Bookmark for easy access

### **Remote Monitoring:**

- Use SSH for remote access: `ssh pi@[PI_IP]`
- Enable VNC for desktop access
- Use mobile apps for camera monitoring

## 🔋 **Power Management**

### **Power Optimization:**

```bash
# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-powersave

# Optimize for headless operation
sudo raspi-config nonint do_boot_behaviour B2
```

### **UPS Integration:**

For critical applications, consider:
- UPS HAT for power backup
- Battery monitoring
- Graceful shutdown scripts

## 📊 **Expected Performance**

### **Pi 4 (4GB) Performance:**
- **Detection FPS**: 5-10 FPS
- **Resolution**: 320x240
- **Memory Usage**: 2-3GB
- **CPU Usage**: 80-90%
- **Temperature**: 60-70°C (with cooling)

### **Pi 4 (8GB) Performance:**
- **Detection FPS**: 8-15 FPS
- **Resolution**: 320x240
- **Memory Usage**: 3-4GB
- **CPU Usage**: 70-85%
- **Temperature**: 55-65°C (with cooling)

## 🎯 **Production Deployment**

### **Recommended Setup:**
1. **Hardware**: Pi 4 8GB with active cooling
2. **Storage**: 64GB+ high-endurance SD card
3. **Network**: Ethernet connection
4. **Power**: UPS backup
5. **Monitoring**: Temperature and performance alerts

### **Deployment Checklist:**
- [ ] Pi OS updated and optimized
- [ ] Camera interface enabled
- [ ] Static IP configured
- [ ] Model trained and tested
- [ ] Auto-start service enabled
- [ ] Performance monitoring setup
- [ ] Backup procedures in place
- [ ] Remote access configured

This setup provides a robust, production-ready helmet detection system optimized for Raspberry Pi 4 hardware!
