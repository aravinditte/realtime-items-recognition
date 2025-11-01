#!/usr/bin/env python3
"""
Real-Time Object Detection Web Application
Supports webcam, video upload, and image upload detection using YOLO models.

Author: Aravind Itte
Technology Stack: Flask, SocketIO, OpenCV, Ultralytics YOLO, TensorFlow
"""

import os
import cv2
import base64
import numpy as np
import eventlet
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import threading
import time
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'realtime-object-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables
model = None
model_loaded = False
client_connections = {}
processing_active = {}

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    if file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    return False

def load_model():
    """Load YOLO model with error handling"""
    global model, model_loaded
    try:
        logger.info("Loading YOLO model...")
        # Try YOLOv8n first (smallest model for faster inference)
        model = YOLO('yolov8n.pt')
        model_loaded = True
        logger.info("YOLO model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        model_loaded = False
        try:
            # Fallback to pre-trained model download
            logger.info("Attempting to download YOLOv8n model...")
            model = YOLO('yolov8n.pt')
            model_loaded = True
            logger.info("YOLO model downloaded and loaded successfully!")
        except Exception as e2:
            logger.error(f"Failed to download model: {e2}")
            model_loaded = False

def detect_objects(frame):
    """Perform object detection on a frame"""
    global model, model_loaded
    
    if not model_loaded or model is None:
        return frame, []
    
    try:
        # Run YOLO detection
        results = model(frame, conf=0.4, iou=0.45, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Draw bounding box with better styling
                    color = (0, 255, 0)  # Green color
                    thickness = 2
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Draw label background
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Store detection info
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return frame, detections
    
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return frame, []

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for image/video processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type and extension
    is_video = allowed_file(file.filename, 'video')
    is_image = allowed_file(file.filename, 'image')
    
    if not (is_video or is_image):
        return jsonify({'error': 'Unsupported file format'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join('uploads', unique_filename)
        
        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        if is_image:
            # Process image
            frame = cv2.imread(filepath)
            if frame is not None:
                processed_frame, detections = detect_objects(frame)
                
                # Encode processed frame
                _, buffer = cv2.imencode('.jpg', processed_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Clean up
                os.remove(filepath)
                
                return jsonify({
                    'type': 'image',
                    'image': img_base64,
                    'detections': detections,
                    'count': len(detections)
                })
        
        elif is_video:
            # Return video path for streaming processing
            return jsonify({
                'type': 'video',
                'filepath': filepath,
                'message': 'Video uploaded successfully. Use WebSocket for real-time processing.'
            })
    
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': 'Failed to process file'}), 500
    
    return jsonify({'error': 'Unknown error occurred'}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    client_connections[client_id] = {
        'connected_at': datetime.now(),
        'active': True
    }
    processing_active[client_id] = False
    
    logger.info(f"Client {client_id} connected")
    emit('connection_status', {
        'status': 'connected',
        'model_loaded': model_loaded,
        'client_id': client_id
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in client_connections:
        del client_connections[client_id]
    if client_id in processing_active:
        processing_active[client_id] = False
        del processing_active[client_id]
    
    logger.info(f"Client {client_id} disconnected")

@socketio.on('webcam_frame')
def handle_webcam_frame(data):
    """Handle webcam frame from client browser"""
    client_id = request.sid
    
    try:
        # Decode base64 image from browser
        image_data = data['frame'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Process frame for object detection
            processed_frame, detections = detect_objects(frame)
            
            # Encode processed frame back to base64
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send processed frame back to client
            emit('processed_frame', {
                'frame': f"data:image/jpeg;base64,{processed_base64}",
                'detections': detections,
                'timestamp': time.time()
            })
        
    except Exception as e:
        logger.error(f"Error processing webcam frame: {e}")
        emit('error', {'message': f'Frame processing error: {str(e)}'})

@socketio.on('process_video')
def handle_process_video(data):
    """Process uploaded video file with real-time detection"""
    client_id = request.sid
    filepath = data.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        emit('error', {'message': 'Video file not found'})
        return
    
    processing_active[client_id] = True
    
    def video_stream():
        """Video processing thread with real-time detection"""
        cap = cv2.VideoCapture(filepath)
        
        if not cap.isOpened():
            socketio.emit('error', {'message': 'Cannot open video file'}, room=client_id)
            processing_active[client_id] = False
            return
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate delay to maintain video speed
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # Default 30 FPS
        
        while processing_active.get(client_id, False):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame with object detection
            processed_frame, detections = detect_objects(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit frame to client with detection info
            socketio.emit('video_frame', {
                'frame': f"data:image/jpeg;base64,{frame_base64}",
                'detections': detections,
                'progress': (frame_count / total_frames) * 100,
                'frame_number': frame_count,
                'total_frames': total_frames
            }, room=client_id)
            
            # Control playback speed
            eventlet.sleep(frame_delay)
        
        cap.release()
        
        # Clean up video file
        try:
            os.remove(filepath)
        except:
            pass
        
        socketio.emit('video_complete', {}, room=client_id)
        logger.info(f"Video processing completed for client {client_id}")
    
    # Start video processing thread
    eventlet.spawn(video_stream)

@socketio.on('stop_processing')
def handle_stop_processing():
    """Stop any active processing"""
    client_id = request.sid
    processing_active[client_id] = False
    emit('processing_stopped', {'status': 'stopped'})

if __name__ == '__main__':
    # Load YOLO model on startup
    load_model()
    
    # Start the application
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Real-Time Object Detection Server on {host}:{port}")
    logger.info(f"Model loaded: {model_loaded}")
    
    socketio.run(app, host=host, port=port, debug=False)
