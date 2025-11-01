#!/usr/bin/env python3
"""
Real-Time Object Detection Web Application

Fixes:
- Remove invalid 'transports' kwarg from socketio.run() that was causing TypeError
- Force WebSocket transport via client-side configuration instead
- Use PORT=5000 as default (can be overridden by Render's PORT env var)
- Keep frame queueing, rate limiting, and preprocessing improvements
- Add configurable detection thresholds via environment variables
"""
import os
import cv2
import base64
import numpy as np
import eventlet
from eventlet.queue import LightQueue, Empty
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
CONF = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.30'))
IOU = float(os.environ.get('IOU_THRESHOLD', '0.45'))
MAX_Q = int(os.environ.get('FRAME_QUEUE_MAX', '3'))
TARGET_W = int(os.environ.get('INFER_WIDTH', '640'))
TARGET_H = int(os.environ.get('INFER_HEIGHT', '384'))
CORS = os.environ.get('CORS_ORIGINS', '*')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'realtime-object-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Initialize SocketIO with optimized settings
socketio = SocketIO(
    app, 
    cors_allowed_origins=CORS, 
    async_mode='eventlet',
    logger=False, 
    engineio_logger=False, 
    ping_timeout=30, 
    ping_interval=25
)

# Global variables
model = None
model_loaded = False
processing_active = {}
frame_queues = {}

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return False

def load_model():
    """Load YOLO model with error handling"""
    global model, model_loaded
    try:
        logger.info("Loading YOLO model...")
        model_name = os.environ.get('MODEL_NAME', 'yolov8n.pt')
        model = YOLO(model_name)
        model_loaded = True
        logger.info(f"YOLO model ({model_name}) loaded successfully!")
    except Exception as e:
        logger.exception(f"Error loading YOLO model: {e}")
        model_loaded = False

def preprocess_frame(frame):
    """Resize frame to target dimensions for consistent inference"""
    return cv2.resize(frame, (TARGET_W, TARGET_H))

def detect_objects(frame):
    """Perform object detection on a frame with proper scaling"""
    if not model_loaded or model is None:
        return frame, []
    
    try:
        # Preprocess frame
        resized_frame = preprocess_frame(frame)
        
        # Run YOLO detection on resized frame
        results = model(resized_frame, conf=CONF, iou=IOU, verbose=False)
        
        detections = []
        annotated_frame = frame.copy()
        
        # Scale factors to map detections back to original frame size
        orig_h, orig_w = frame.shape[:2]
        resized_h, resized_w = resized_frame.shape[:2]
        scale_x, scale_y = orig_w / resized_w, orig_h / resized_h
        
        # Process detection results
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                # Get box coordinates and metadata
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names.get(class_id, str(class_id))
                
                # Scale coordinates back to original frame size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = f"{class_name} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - text_height - 6)), 
                            (x1 + text_width + 6, y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Store detection info
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return annotated_frame, detections
    
    except Exception as e:
        logger.exception(f"Detection error: {e}")
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
        'timestamp': datetime.now().isoformat(),
        'config': {
            'confidence_threshold': CONF,
            'iou_threshold': IOU,
            'input_size': f"{TARGET_W}x{TARGET_H}"
        }
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
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', unique_filename)
        file.save(filepath)
        
        if is_image:
            # Process image immediately
            frame = cv2.imread(filepath)
            if frame is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            annotated_frame, detections = detect_objects(frame)
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except Exception:
                pass
            
            return jsonify({
                'type': 'image',
                'image': img_base64,
                'detections': detections,
                'count': len(detections)
            })
        
        else:  # is_video
            return jsonify({
                'type': 'video',
                'filepath': filepath,
                'message': 'Video uploaded successfully. Starting real-time processing...'
            })
    
    except Exception as e:
        logger.exception(f"Error processing upload: {e}")
        return jsonify({'error': 'Failed to process file'}), 500

@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    sid = request.sid
    processing_active[sid] = False
    frame_queues[sid] = LightQueue(maxsize=MAX_Q)
    
    logger.info(f"Client {sid} connected")
    emit('connection_status', {
        'status': 'connected',
        'model_loaded': model_loaded,
        'client_id': sid
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    sid = request.sid
    processing_active.pop(sid, None)
    frame_queues.pop(sid, None)
    logger.info(f"Client {sid} disconnected")

@socketio.on('webcam_frame')
def on_webcam_frame(data):
    """Handle webcam frame from browser - queue with backpressure"""
    sid = request.sid
    queue = frame_queues.get(sid)
    
    if not queue:
        return
    
    try:
        # Drop oldest frame if queue is full to prevent overflow
        if queue.qsize() >= MAX_Q:
            try:
                queue.get_nowait()
            except Empty:
                pass
        
        # Add new frame to queue
        queue.put_nowait(data['frame'])
    except Exception:
        pass
    
    # Start frame processing worker if not already running
    if not processing_active.get(sid):
        processing_active[sid] = True
        eventlet.spawn_n(process_frames_worker, sid)

def process_frames_worker(sid):
    """Background worker to process webcam frames with rate limiting"""
    queue = frame_queues.get(sid)
    if not queue:
        return
    
    last_emit_time = 0.0
    target_interval = 0.1  # 10 FPS max
    
    while processing_active.get(sid) and queue:
        try:
            # Get frame from queue with timeout
            frame_data = queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            # Decode base64 frame
            image_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Process frame for object detection
            annotated_frame, detections = detect_objects(frame)
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Rate limit emissions
            current_time = time.time()
            if current_time - last_emit_time >= target_interval:
                socketio.emit('processed_frame', {
                    'frame': f"data:image/jpeg;base64,{processed_base64}",
                    'detections': detections,
                    'timestamp': current_time
                }, room=sid)
                last_emit_time = current_time
                
        except Exception as e:
            logger.exception(f"Frame processing error: {e}")
            socketio.emit('error', {'message': 'Frame processing error'}, room=sid)

@socketio.on('process_video')
def on_process_video(data):
    """Handle video processing request"""
    sid = request.sid
    filepath = data.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        emit('error', {'message': 'Video file not found'})
        return
    
    processing_active[sid] = True
    eventlet.spawn_n(process_video_worker, sid, filepath)

def process_video_worker(sid, filepath):
    """Background worker to process video frames"""
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        socketio.emit('error', {'message': 'Cannot open video file'}, room=sid)
        processing_active[sid] = False
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / fps
    frame_count = 0
    
    try:
        while processing_active.get(sid) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame with object detection
            annotated_frame, detections = detect_objects(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit processed frame
            socketio.emit('video_frame', {
                'frame': f"data:image/jpeg;base64,{frame_base64}",
                'detections': detections,
                'progress': (frame_count / total_frames * 100) if total_frames > 0 else 0,
                'frame_number': frame_count,
                'total_frames': total_frames
            }, room=sid)
            
            # Control playback speed
            eventlet.sleep(frame_delay)
            
    finally:
        cap.release()
        # Clean up video file
        try:
            os.remove(filepath)
        except Exception:
            pass
        
        socketio.emit('video_complete', {}, room=sid)
        logger.info(f"Video processing completed for client {sid}")

@socketio.on('stop_processing')
def on_stop_processing():
    """Stop any active processing for client"""
    sid = request.sid
    processing_active[sid] = False
    emit('processing_stopped', {'status': 'stopped'})

if __name__ == '__main__':
    # Load YOLO model on startup
    load_model()
    
    # Use PORT=5000 as default, can be overridden by Render's PORT env var
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Real-Time Object Detection Server on {host}:{port}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"Detection config: conf={CONF}, iou={IOU}, input_size={TARGET_W}x{TARGET_H}")
    
    # Start the application (removed invalid 'transports' parameter)
    socketio.run(app, host=host, port=port, debug=False)