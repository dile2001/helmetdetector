#!/usr/bin/env python3
"""
Helmet Detection System - Quick Test Script

This script provides a quick way to test the helmet detection system
without setting up the full web interface.
"""

import cv2
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.helmet_detector import HelmetDetector
from src.alarm_system import AlarmSystem


def test_detector():
    """Test the helmet detector with webcam."""
    print("Testing Helmet Detector...")
    print("Press 'q' to quit, 't' to test alarm")
    
    # Initialize components
    detector = HelmetDetector()
    alarm_system = AlarmSystem()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print("Camera opened successfully")
    print("Starting detection...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Process every 5th frame for performance
        if frame_count % 5 == 0:
            # Run detection
            detection_result = detector.detect_persons_and_helmets(frame)
            
            # Draw detections
            annotated_frame = detector.draw_detections(
                frame, 
                detection_result['helmet_detections'] + detection_result['no_helmet_detections']
            )
            
            # Show violation count
            violations = detection_result['violations']
            if violations > 0:
                cv2.putText(annotated_frame, f"VIOLATIONS: {violations}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Trigger alarm
                alarm_system.trigger_alarm(detection_result)
            else:
                cv2.putText(annotated_frame, "SAFE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Helmet Detection Test', annotated_frame)
        else:
            cv2.imshow('Helmet Detection Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            print("Testing alarm...")
            alarm_system.test_alarm()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Test completed. Processed {frame_count} frames in {time.time() - start_time:.1f} seconds")
    return True


def test_image_detection():
    """Test detection on a single image."""
    print("Testing Image Detection...")
    
    detector = HelmetDetector()
    
    # Create a test image (you can replace this with a real image path)
    test_image_path = "test_image.jpg"
    
    if not Path(test_image_path).exists():
        print(f"Test image not found: {test_image_path}")
        print("Please add a test image named 'test_image.jpg' to the project directory")
        return False
    
    # Load and process image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Could not load image: {test_image_path}")
        return False
    
    # Run detection
    detection_result = detector.detect_persons_and_helmets(image)
    
    # Draw detections
    annotated_image = detector.draw_detections(
        image, 
        detection_result['helmet_detections'] + detection_result['no_helmet_detections']
    )
    
    # Save result
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, annotated_image)
    
    # Print results
    print(f"Detection Results:")
    print(f"  Total detections: {detection_result['total_detections']}")
    print(f"  Helmets detected: {len(detection_result['helmet_detections'])}")
    print(f"  No helmet detected: {len(detection_result['no_helmet_detections'])}")
    print(f"  Violations: {detection_result['violations']}")
    print(f"  Result saved to: {output_path}")
    
    return True


def test_alarm_system():
    """Test the alarm system."""
    print("Testing Alarm System...")
    
    alarm_system = AlarmSystem()
    
    # Test alarm
    print("Triggering test alarm...")
    alarm_system.test_alarm()
    
    # Wait a moment
    time.sleep(2)
    
    # Test with violation data
    violation_data = {
        'violations': 2,
        'timestamp': time.time(),
        'message': 'Test violation'
    }
    
    print("Triggering violation alarm...")
    alarm_system.trigger_alarm(violation_data)
    
    # Get status
    status = alarm_system.get_status()
    print(f"Alarm system status: {status}")
    
    return True


def main():
    """Main test function."""
    print("=" * 60)
    print("Helmet Detection System - Quick Test")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        print("Available tests:")
        print("  camera    - Test with live camera feed")
        print("  image     - Test with single image")
        print("  alarm     - Test alarm system")
        print("  all       - Run all tests")
        print()
        test_type = input("Enter test type (camera/image/alarm/all): ").lower()
    
    success = True
    
    if test_type in ['camera', 'all']:
        print("\n" + "=" * 40)
        print("Camera Test")
        print("=" * 40)
        if not test_detector():
            success = False
    
    if test_type in ['image', 'all']:
        print("\n" + "=" * 40)
        print("Image Test")
        print("=" * 40)
        if not test_image_detection():
            success = False
    
    if test_type in ['alarm', 'all']:
        print("\n" + "=" * 40)
        print("Alarm Test")
        print("=" * 40)
        if not test_alarm_system():
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests completed successfully!")
    else:
        print("✗ Some tests failed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
