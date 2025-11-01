#!/usr/bin/env python3
"""
Simple MVP Object Detection Script

This is the minimal viable product (MVP) script for real-time object detection
as outlined in the project requirements. It demonstrates the core concept:
"Can I see labeled objects from my webcam in real-time?"

Usage:
    python detect.py

Press 'q' to quit.

Author: Aravind Itte
Date: November 2024
"""

import cv2
from ultralytics import YOLO
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function implementing the MVP object detection workflow
    """
    logger.info("Initializing Real-Time Object Detection MVP")
    
    # Load the pre-trained YOLOv8 nano model
    try:
        logger.info("Loading YOLOv8 nano model...")
        model = YOLO("yolov8n.pt")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Initialize video capture from default webcam (index 0)
    try:
        logger.info("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Webcam initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize webcam: {e}")
        return
    
    # Main detection loop
    logger.info("Starting real-time detection. Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame from webcam
            success, frame = cap.read()
            
            if not success:
                logger.warning("Failed to read frame from webcam")
                break
            
            # Perform object detection
            results = model(frame)
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                
                # Draw FPS on frame
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display the annotated frame
            cv2.imshow('Real-Time Object Detection MVP', annotated_frame)
            
            # Check for quit key ('q')
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed")
                break
    
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during detection: {e}")
    
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        if frame_count > 0 and elapsed_time > 0:
            final_fps = frame_count / elapsed_time
            logger.info(f"Session complete. Processed {frame_count} frames at {final_fps:.1f} FPS")

if __name__ == "__main__":
    main()
