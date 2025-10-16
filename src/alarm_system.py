"""
Helmet Detection System - Alarm System

This module handles alarm triggering when helmet violations are detected.
Supports multiple alarm types: audio, visual, and external notifications.
"""

import pygame
import threading
import time
import logging
from typing import Dict, List, Optional, Callable
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AlarmType(Enum):
    """Types of alarms available."""
    AUDIO = "audio"
    VISUAL = "visual"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"


class AlarmConfig:
    """Configuration for alarm settings."""
    
    def __init__(self):
        self.enabled = True
        self.audio_enabled = True
        self.visual_enabled = True
        self.email_enabled = False
        self.sms_enabled = False
        self.webhook_enabled = False
        
        # Audio settings
        self.audio_file = "static/alarm.wav"
        self.audio_volume = 0.7
        self.audio_duration = 3.0  # seconds
        
        # Visual settings
        self.flash_duration = 2.0  # seconds
        self.flash_color = (255, 0, 0)  # Red
        
        # Notification settings
        self.email_recipients = []
        self.sms_recipients = []
        self.webhook_url = ""
        
        # Cooldown settings
        self.cooldown_duration = 5.0  # seconds between alarms
        self.max_alarms_per_minute = 10
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'enabled': self.enabled,
            'audio_enabled': self.audio_enabled,
            'visual_enabled': self.visual_enabled,
            'email_enabled': self.email_enabled,
            'sms_enabled': self.sms_enabled,
            'webhook_enabled': self.webhook_enabled,
            'audio_file': self.audio_file,
            'audio_volume': self.audio_volume,
            'audio_duration': self.audio_duration,
            'flash_duration': self.flash_duration,
            'flash_color': self.flash_color,
            'email_recipients': self.email_recipients,
            'sms_recipients': self.sms_recipients,
            'webhook_url': self.webhook_url,
            'cooldown_duration': self.cooldown_duration,
            'max_alarms_per_minute': self.max_alarms_per_minute
        }
    
    def from_dict(self, config_dict: Dict):
        """Load config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class AlarmSystem:
    """
    Main alarm system that handles triggering alarms when violations are detected.
    """
    
    def __init__(self, config: Optional[AlarmConfig] = None):
        """
        Initialize the alarm system.
        
        Args:
            config: Alarm configuration object
        """
        self.config = config or AlarmConfig()
        self.last_alarm_time = 0
        self.alarm_count_this_minute = 0
        self.minute_start_time = time.time()
        
        # Initialize pygame for audio
        if self.config.audio_enabled:
            self._init_audio()
        
        # Alarm callbacks
        self.alarm_callbacks: List[Callable] = []
        
        # Load config from file if exists
        self._load_config()
    
    def _init_audio(self):
        """Initialize pygame audio system."""
        try:
            pygame.mixer.init()
            logger.info("Audio system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            self.config.audio_enabled = False
    
    def _load_config(self):
        """Load configuration from file."""
        config_file = Path("config/alarm_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                self.config.from_dict(config_dict)
                logger.info("Alarm configuration loaded from file")
            except Exception as e:
                logger.error(f"Failed to load alarm config: {e}")
    
    def save_config(self):
        """Save configuration to file."""
        config_file = Path("config/alarm_config.json")
        config_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info("Alarm configuration saved")
        except Exception as e:
            logger.error(f"Failed to save alarm config: {e}")
    
    def add_alarm_callback(self, callback: Callable):
        """Add a custom alarm callback function."""
        self.alarm_callbacks.append(callback)
    
    def _check_cooldown(self) -> bool:
        """Check if enough time has passed since last alarm."""
        current_time = time.time()
        
        # Reset minute counter if needed
        if current_time - self.minute_start_time >= 60:
            self.alarm_count_this_minute = 0
            self.minute_start_time = current_time
        
        # Check cooldown
        if current_time - self.last_alarm_time < self.config.cooldown_duration:
            return False
        
        # Check rate limit
        if self.alarm_count_this_minute >= self.config.max_alarms_per_minute:
            logger.warning("Alarm rate limit exceeded")
            return False
        
        return True
    
    def trigger_alarm(self, violation_data: Dict):
        """
        Trigger alarm for helmet violations.
        
        Args:
            violation_data: Dictionary containing violation information
        """
        if not self.config.enabled:
            return
        
        if not self._check_cooldown():
            logger.debug("Alarm skipped due to cooldown")
            return
        
        logger.warning(f"Helmet violation detected! Violations: {violation_data.get('violations', 0)}")
        
        # Update timing
        self.last_alarm_time = time.time()
        self.alarm_count_this_minute += 1
        
        # Trigger different alarm types
        if self.config.audio_enabled:
            self._trigger_audio_alarm()
        
        if self.config.visual_enabled:
            self._trigger_visual_alarm()
        
        if self.config.email_enabled:
            self._trigger_email_alarm(violation_data)
        
        if self.config.sms_enabled:
            self._trigger_sms_alarm(violation_data)
        
        if self.config.webhook_enabled:
            self._trigger_webhook_alarm(violation_data)
        
        # Call custom callbacks
        for callback in self.alarm_callbacks:
            try:
                callback(violation_data)
            except Exception as e:
                logger.error(f"Alarm callback failed: {e}")
    
    def _trigger_audio_alarm(self):
        """Trigger audio alarm."""
        try:
            if Path(self.config.audio_file).exists():
                sound = pygame.mixer.Sound(self.config.audio_file)
                sound.set_volume(self.config.audio_volume)
                sound.play()
                logger.info("Audio alarm triggered")
            else:
                # Fallback: system beep
                pygame.mixer.Sound.play(pygame.mixer.Sound(buffer=bytes([128] * 1000)))
                logger.info("System beep alarm triggered")
        except Exception as e:
            logger.error(f"Audio alarm failed: {e}")
    
    def _trigger_visual_alarm(self):
        """Trigger visual alarm (screen flash)."""
        try:
            # This would typically flash the screen or show a visual alert
            # Implementation depends on the GUI framework being used
            logger.info("Visual alarm triggered")
        except Exception as e:
            logger.error(f"Visual alarm failed: {e}")
    
    def _trigger_email_alarm(self, violation_data: Dict):
        """Trigger email notification."""
        try:
            # This would send an email notification
            # Implementation would depend on email service (SMTP, SendGrid, etc.)
            logger.info(f"Email alarm triggered for {len(self.config.email_recipients)} recipients")
        except Exception as e:
            logger.error(f"Email alarm failed: {e}")
    
    def _trigger_sms_alarm(self, violation_data: Dict):
        """Trigger SMS notification."""
        try:
            # This would send an SMS notification
            # Implementation would depend on SMS service (Twilio, etc.)
            logger.info(f"SMS alarm triggered for {len(self.config.sms_recipients)} recipients")
        except Exception as e:
            logger.error(f"SMS alarm failed: {e}")
    
    def _trigger_webhook_alarm(self, violation_data: Dict):
        """Trigger webhook notification."""
        try:
            # This would send a webhook notification
            # Implementation would use requests library
            logger.info(f"Webhook alarm triggered to {self.config.webhook_url}")
        except Exception as e:
            logger.error(f"Webhook alarm failed: {e}")
    
    def test_alarm(self):
        """Test all enabled alarm systems."""
        test_data = {
            'violations': 1,
            'timestamp': time.time(),
            'message': 'Test alarm'
        }
        self.trigger_alarm(test_data)
    
    def get_status(self) -> Dict:
        """Get current alarm system status."""
        return {
            'enabled': self.config.enabled,
            'audio_enabled': self.config.audio_enabled,
            'visual_enabled': self.config.visual_enabled,
            'email_enabled': self.config.email_enabled,
            'sms_enabled': self.config.sms_enabled,
            'webhook_enabled': self.config.webhook_enabled,
            'last_alarm_time': self.last_alarm_time,
            'alarm_count_this_minute': self.alarm_count_this_minute,
            'cooldown_active': not self._check_cooldown()
        }


class AlarmLogger:
    """Log alarm events to file."""
    
    def __init__(self, log_file: str = "logs/alarms.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log_alarm(self, violation_data: Dict):
        """Log alarm event."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        violations = violation_data.get('violations', 0)
        
        log_entry = f"{timestamp} - Helmet violation detected: {violations} violations\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Failed to log alarm: {e}")


if __name__ == "__main__":
    # Test the alarm system
    config = AlarmConfig()
    config.audio_enabled = True
    config.visual_enabled = True
    
    alarm_system = AlarmSystem(config)
    
    # Test alarm
    test_data = {
        'violations': 2,
        'timestamp': time.time()
    }
    
    alarm_system.trigger_alarm(test_data)
    print("Test alarm triggered")
