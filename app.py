#!/usr/bin/env python3
"""
Real-Time Object Detection System

A comprehensive web application for real-time object detection using YOLOv8.
Supports webcam streaming, video uploads, and image processing.

Author: Aravind Itte
Date: November 2024
"""

import os
import cv2
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import threading
import time
import queue
import json
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from datetime import datetime
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', '52428800'))  # 50MB

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration from environment variables
CONFIG = {
    'MODEL_NAME': os.environ.get('MODEL_NAME', 'yolov8n.pt'),
    'CONFIDENCE_THRESHOLD': float(os.environ.get('CONFIDENCE_THRESHOLD', '0.30')),
    'IOU_THRESHOLD': float(os.environ.get('IOU_THRESHOLD', '0.45')),
    'INFER_WIDTH': int(os.environ.get('INFER_WIDTH', '640')),
    'INFER_HEIGHT': int(os.environ.get('INFER_HEIGHT', '384')),
    'FRAME_QUEUE_MAX': int(os.environ.get('FRAME_QUEUE_MAX', '3')),
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# Global variables
model = None
processing_active = False
frame_queue = queue.Queue(maxsize=CONFIG['FRAME_QUEUE_MAX'])
current_connections = 0
statistics = {
    'total_frames_processed': 0,
    'total_detections': 0,
    'average_fps': 0,
    'average_processing_time': 0,
    'last_detection_time': None
}

class ObjectDetector:
    """
    Core object detection class handling YOLO model operations
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model with error handling"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            logger.info(f"Model loaded successfully: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            return False
    
    def detect_objects(self, image, conf_threshold=None, iou_threshold=None):
        """Perform object detection on image"""
        if self.model is None:
            return None, None
        
        try:
            conf_threshold = conf_threshold or CONFIG['CONFIDENCE_THRESHOLD']
            iou_threshold = iou_threshold or CONFIG['IOU_THRESHOLD']
            
            # Run inference
            results = self.model(
                image, 
                conf=conf_threshold, 
                iou=iou_threshold,
                imgsz=(CONFIG['INFER_HEIGHT'], CONFIG['INFER_WIDTH'])
            )
            
            return results, None
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return None, str(e)
    
    def draw_detections(self, image, results):
        """Draw bounding boxes and labels on image"""
        if not results or len(results) == 0:
            return image, 0
        
        detection_count = 0
        annotated_frame = image.copy()
        
        try:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = result.names[class_id]
                        
                        # Choose color based on class
                        color = self._get_color_for_class(class_id)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label with confidence
                        label = f"{class_name}: {confidence:.2f}"
                        
                        # Calculate label size and position
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - label_height - baseline),
                            (x1 + label_width, y1),
                            color,
                            cv2.FILLED
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )
                        
                        detection_count += 1
            
            return annotated_frame, detection_count
        
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return image, 0
    
    def _get_color_for_class(self, class_id):
        """Get consistent color for each object class"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 128, 0),  # Olive
        ]
        return colors[class_id % len(colors)]

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return None

def decode_base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        return None

def process_webcam_frame(frame_data):
    """Process a single webcam frame"""
    global statistics
    
    try:
        start_time = time.time()
        
        # Decode frame
        frame = decode_base64_to_image(frame_data)
        if frame is None:
            return None, "Failed to decode frame"
        
        # Perform detection
        results, error = detector.detect_objects(frame)
        if error:
            return None, error
        
        # Draw detections
        annotated_frame, detection_count = detector.draw_detections(frame, results)
        
        # Update statistics
        processing_time = time.time() - start_time
        statistics['total_frames_processed'] += 1
        statistics['total_detections'] += detection_count
        statistics['average_processing_time'] = (
            (statistics['average_processing_time'] * (statistics['total_frames_processed'] - 1) + processing_time) / 
            statistics['total_frames_processed']
        )
        
        if detection_count > 0:
            statistics['last_detection_time'] = datetime.now().isoformat()
        
        # Calculate FPS
        fps = 1.0 / processing_time if processing_time > 0 else 0
        statistics['average_fps'] = (
            (statistics['average_fps'] * (statistics['total_frames_processed'] - 1) + fps) / 
            statistics['total_frames_processed']
        )
        
        # Encode result
        result_base64 = encode_image_to_base64(annotated_frame)
        
        return {
            'image': result_base64,
            'detections': detection_count,
            'fps': round(fps, 2),
            'processing_time': round(processing_time * 1000, 2)  # Convert to ms
        }, None
        
    except Exception as e:
        logger.error(f"Error processing webcam frame: {str(e)}")
        return None, str(e)

def process_video_file(video_path, emit_callback):
    """Process uploaded video file frame by frame"""
    global processing_active
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            emit_callback('error', {'message': 'Failed to open video file'})
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        
        frame_count = 0
        processing_active = True
        
        while processing_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results, error = detector.detect_objects(frame)
            if not error and results:
                annotated_frame, detection_count = detector.draw_detections(frame, results)
            else:
                annotated_frame = frame
                detection_count = 0
            
            # Encode and emit frame
            frame_base64 = encode_image_to_base64(annotated_frame)
            if frame_base64:
                progress = (frame_count + 1) / total_frames * 100
                emit_callback('video_frame', {
                    'image': frame_base64,
                    'frame_number': frame_count + 1,
                    'total_frames': total_frames,
                    'progress': round(progress, 2),
                    'detections': detection_count
                })
            
            frame_count += 1
            
            # Add small delay to prevent overwhelming the client
            time.sleep(0.03)  # ~30 FPS playback
        
        cap.release()
        
        if processing_active:
            emit_callback('video_complete', {
                'total_frames_processed': frame_count,
                'message': 'Video processing completed successfully'
            })
        
        processing_active = False
        
        # Clean up video file
        if os.path.exists(video_path):
            os.remove(video_path)
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        emit_callback('error', {'message': f'Error processing video: {str(e)}'})
        processing_active = False

# Initialize detector
detector = ObjectDetector(CONFIG['MODEL_NAME'])

# Flask routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = detector.model is not None
    return jsonify({
        'status': 'healthy' if model_status else 'unhealthy',
        'model_loaded': model_status,
        'model_name': CONFIG['MODEL_NAME'],
        'connections': current_connections,
        'statistics': statistics,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        
        # Save file
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Check if it's an image or video
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            # Process image immediately
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Failed to load image'}), 400
            
            results, error = detector.detect_objects(image)
            if error:
                return jsonify({'error': error}), 500
            
            annotated_image, detection_count = detector.draw_detections(image, results)
            result_base64 = encode_image_to_base64(annotated_image)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'type': 'image',
                'result': result_base64,
                'detections': detection_count,
                'message': f'Detected {detection_count} objects in the image'
            })
        
        else:
            # For video, return filepath for WebSocket processing
            return jsonify({
                'type': 'video',
                'filepath': filepath,
                'message': 'Video uploaded successfully. Start processing via WebSocket.'
            })
    
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    global current_connections
    current_connections += 1
    logger.info(f"Client connected. Total connections: {current_connections}")
    emit('connection_status', {
        'status': 'connected',
        'message': 'Connected to real-time object detection server',
        'model_name': CONFIG['MODEL_NAME']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global current_connections, processing_active
    current_connections -= 1
    processing_active = False
    logger.info(f"Client disconnected. Total connections: {current_connections}")

@socketio.on('webcam_frame')
def handle_webcam_frame(data):
    """Handle webcam frame from client"""
    try:
        frame_data = data.get('image')
        if not frame_data:
            emit('error', {'message': 'No image data received'})
            return
        
        # Add frame to queue (non-blocking)
        try:
            frame_queue.put_nowait(frame_data)
        except queue.Full:
            # Skip frame if queue is full
            pass
        
        # Process frame from queue
        if not frame_queue.empty():
            frame_to_process = frame_queue.get()
            result, error = process_webcam_frame(frame_to_process)
            
            if error:
                emit('error', {'message': error})
            elif result:
                emit('processed_frame', result)
    
    except Exception as e:
        logger.error(f"Error handling webcam frame: {str(e)}")
        emit('error', {'message': f'Processing error: {str(e)}'})

@socketio.on('process_video')
def handle_process_video(data):
    """Handle video processing request"""
    global processing_active
    
    try:
        filepath = data.get('filepath')
        if not filepath or not os.path.exists(filepath):
            emit('error', {'message': 'Video file not found'})
            return
        
        if processing_active:
            emit('error', {'message': 'Another video is currently being processed'})
            return
        
        # Start video processing in a separate thread
        def emit_callback(event, data):
            socketio.emit(event, data)
        
        thread = threading.Thread(
            target=process_video_file,
            args=(filepath, emit_callback)
        )
        thread.start()
        
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        emit('error', {'message': f'Failed to start video processing: {str(e)}'})

@socketio.on('stop_processing')
def handle_stop_processing():
    """Handle stop processing request"""
    global processing_active
    processing_active = False
    emit('processing_stopped', {'message': 'Processing stopped by user'})
    logger.info("Processing stopped by user request")

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check if model is loaded
    if detector.model is None:
        logger.error("Failed to load YOLO model. Exiting...")
        exit(1)
    
    # Get host and port from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Real-Time Object Detection Server on {host}:{port}")
    logger.info(f"Model: {CONFIG['MODEL_NAME']}")
    logger.info(f"Debug mode: {debug}")
    
    # Start the application
    socketio.run(app, host=host, port=port, debug=debug)
