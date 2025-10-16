# Helmet Detection System - Configuration Files

This directory contains configuration files for the helmet detection system.

## Files

- `alarm_config.json` - Alarm system configuration
- `camera_config.json` - Camera settings configuration

## Alarm Configuration

The alarm configuration controls how the system responds to helmet violations:

```json
{
  "enabled": true,                    // Enable/disable alarm system
  "audio_enabled": true,              // Enable audio alarms
  "visual_enabled": true,             // Enable visual alarms
  "email_enabled": false,             // Enable email notifications
  "sms_enabled": false,               // Enable SMS notifications
  "webhook_enabled": false,           // Enable webhook notifications
  "audio_file": "static/alarm.wav",   // Path to alarm sound file
  "audio_volume": 0.7,                // Audio volume (0.0 to 1.0)
  "audio_duration": 3.0,              // Alarm duration in seconds
  "flash_duration": 2.0,              // Visual flash duration
  "flash_color": [255, 0, 0],         // RGB color for visual alarm
  "email_recipients": [],             // List of email addresses
  "sms_recipients": [],               // List of phone numbers
  "webhook_url": "",                  // Webhook endpoint URL
  "cooldown_duration": 5.0,           // Seconds between alarms
  "max_alarms_per_minute": 10         // Rate limit for alarms
}
```

## Camera Configuration

The camera configuration controls video capture settings:

```json
{
  "camera_id": 0,                     // Camera device ID
  "camera_url": null,                 // IP camera URL (optional)
  "width": 640,                       // Video width
  "height": 480,                      // Video height
  "fps": 30,                          // Frames per second
  "buffer_size": 10,                   // Frame buffer size
  "detection_interval": 1,             // Process every N frames
  "save_violations": true,             // Save violation images
  "save_path": "logs/violations",     // Path to save violations
  "max_fps": 30,                      // Maximum processing FPS
  "skip_frames": false                // Skip frames for performance
}
```

## Usage

Configuration files are automatically loaded when the system starts. Changes to these files require a system restart to take effect.

You can also modify configurations through the web interface at `http://localhost:8000`.
