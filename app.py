#!/usr/bin/env python3
"""
Integrate the new realtime engine, tracker, and overlay into the Flask-SocketIO server.
"""
import os
import cv2
import base64
import numpy as np
import eventlet
from eventlet.queue import LightQueue, Empty
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import logging
import uuid
from datetime import datetime
import time
from werkzeug.utils import secure_filename

from engine import RealtimeEngine

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Realtime config (no external envs required)
FRAME_QUEUE_MAX = 3
EMIT_FPS_CAP = 12  # allow slightly higher than 10 if CPU permits
CANVAS_W, CANVAS_H = 640, 480

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'realtime-object-recognition-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=False,
    engineio_logger=False,
    ping_timeout=30,
    ping_interval=25,
    allow_upgrades=False,
    cookie=False
)

# Engine
engine = None

# State
processing_active = {}
frame_queues = {}

ALLOWED_VIDEO = {'mp4','avi','mov','mkv','webm'}
ALLOWED_IMAGE = {'jpg','jpeg','png','bmp','tiff'}

def allowed(fname, kind):
    if '.' not in fname:
        return False
    ext = fname.rsplit('.',1)[1].lower()
    return (kind=='video' and ext in ALLOWED_VIDEO) or (kind=='image' and ext in ALLOWED_IMAGE)

@app.before_first_request
def init_engine():
    global engine
    if engine is None:
        logger.info('Initializing realtime engine (YOLO+SORT)...')
        engine = RealtimeEngine('yolov8n.pt')
        logger.info('Realtime engine ready.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status':'healthy','engine_ready': engine is not None,'ts': datetime.now().isoformat()})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error':'No file provided'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error':'No file selected'}), 400
    is_video = allowed(f.filename, 'video')
    is_image = allowed(f.filename, 'image')
    if not (is_video or is_image):
        return jsonify({'error':'Unsupported format'}), 400
    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', f"{uuid.uuid4()}_{secure_filename(f.filename)}")
    f.save(path)
    if is_image:
        img = cv2.imread(path)
        if img is None:
            return jsonify({'error':'Invalid image'}), 400
        b64, tracks = engine.process_frame(img)
        try: os.remove(path)
        except: pass
        return jsonify({'type':'image','image': b64,'detections': tracks,'count': len(tracks)})
    else:
        return jsonify({'type':'video','filepath': path})

@socketio.on('connect')
def on_connect():
    sid = request.sid
    processing_active[sid] = False
    frame_queues[sid] = LightQueue(maxsize=FRAME_QUEUE_MAX)
    emit('connection_status', {'status':'connected','engine_ready': engine is not None})

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    processing_active.pop(sid, None)
    frame_queues.pop(sid, None)

@socketio.on('webcam_frame')
def on_webcam_frame(data):
    sid = request.sid
    q = frame_queues.get(sid)
    if not q:
        return
    try:
        # drop oldest to maintain realtime
        while q.qsize() >= FRAME_QUEUE_MAX:
            try: q.get_nowait()
            except Empty: break
        q.put_nowait(data['frame'])
    except Exception:
        pass
    if not processing_active.get(sid):
        processing_active[sid] = True
        eventlet.spawn_n(webcam_worker, sid)

def webcam_worker(sid):
    q = frame_queues.get(sid)
    if not q:
        return
    last_emit = 0.0
    emit_interval = 1.0 / EMIT_FPS_CAP
    while processing_active.get(sid) and q:
        try:
            frame_data = q.get(timeout=1.0)
        except Empty:
            continue
        try:
            image_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            b64, tracks = engine.process_frame(frame)
            now = time.time()
            if now - last_emit >= emit_interval:
                socketio.emit('processed_frame', {'frame': f"data:image/jpeg;base64,{b64}", 'detections': tracks, 'realtime': True}, room=sid)
                last_emit = now
        except Exception as e:
            socketio.emit('error', {'message': 'Realtime processing error'}, room=sid)

@socketio.on('process_video')
def on_process_video(data):
    sid = request.sid
    path = data.get('filepath')
    if not path or not os.path.exists(path):
        emit('error', {'message': 'Video not found'})
        return
    processing_active[sid] = True
    eventlet.spawn_n(video_worker, sid, path)

def video_worker(sid, path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        socketio.emit('error', {'message':'Cannot open video'}, room=sid)
        processing_active[sid] = False
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = max(0.033, 1.0 / fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    stride = 1  # set to 2 for speed mode
    try:
        while processing_active.get(sid) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if (idx % stride) != 0:
                continue
            b64, tracks = engine.process_frame(frame)
            socketio.emit('video_frame', {
                'frame': f"data:image/jpeg;base64,{b64}",
                'detections': tracks,
                'progress': (idx/total*100) if total else 0,
                'frame_number': idx,
                'total_frames': total,
                'realtime': True
            }, room=sid)
            eventlet.sleep(delay)
    finally:
        cap.release()
        try: os.remove(path)
        except: pass
        socketio.emit('video_complete', {'message':'Video processing complete'}, room=sid)

@socketio.on('stop_processing')
def stop_processing():
    sid = request.sid
    processing_active[sid] = False
    emit('processing_stopped', {'status':'stopped'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    logger.info(f"Starting realtime server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)
