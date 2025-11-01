#!/usr/bin/env python3
"""
Realtime object recognition server with decoupled pipeline and explicit model_loaded status.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAME_QUEUE_MAX = 12
INFER_WORKERS = 2
EMIT_FPS_CAP = 12
JPEG_QUALITY_OUT = 70
WATCHDOG_SECS = 5.0

app = Flask(__name__)
app.config['SECRET_KEY'] = 'realtime-object-recognition-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

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

engine = None

def ensure_engine():
    global engine
    if engine is None:
        logger.info('Initializing realtime engine (YOLO+SORT)...')
        engine = RealtimeEngine('yolov8n.pt')
        logger.info('Realtime engine ready.')
        # Broadcast readiness to all
        try:
            socketio.emit('connection_status', {'status':'connected','engine_ready': True, 'model_loaded': True})
        except Exception:
            pass
    return engine

class ClientState:
    def __init__(self):
        self.input_q = LightQueue(maxsize=FRAME_QUEUE_MAX)
        self.output_q = LightQueue(maxsize=FRAME_QUEUE_MAX)
        self.processing = False
        self.last_emit_ts = 0.0
        self.last_processed_ts = time.time()
        self.stop = False

clients = {}

ALLOWED_VIDEO = {'mp4','avi','mov','mkv','webm'}
ALLOWED_IMAGE = {'jpg','jpeg','png','bmp','tiff'}

def allowed(fname, kind):
    if '.' not in fname:
        return False
    ext = fname.rsplit('.',1)[1].lower()
    return (kind=='video' and ext in ALLOWED_VIDEO) or (kind=='image' and ext in ALLOWED_IMAGE)

@app.route('/')
def index():
    ensure_engine()
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status':'healthy','engine_ready': ensure_engine() is not None,'model_loaded': True,'ts': datetime.now().isoformat()})

@app.route('/upload', methods=['POST'])
def upload():
    eng = ensure_engine()
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
        b64, tracks = eng.process_frame(img)
        try: os.remove(path)
        except: pass
        return jsonify({'type':'image','image': b64,'detections': tracks,'count': len(tracks)})
    else:
        return jsonify({'type':'video','filepath': path})

@socketio.on('connect')
def on_connect():
    ready = ensure_engine() is not None
    sid = request.sid
    clients[sid] = ClientState()
    emit('connection_status', {'status':'connected','engine_ready': ready, 'model_loaded': ready})

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    st = clients.get(sid)
    if st:
        st.stop = True
    clients.pop(sid, None)

@socketio.on('webcam_frame')
def on_webcam_frame(data):
    ensure_engine()
    sid = request.sid
    st = clients.get(sid)
    if not st:
        return
    try:
        while st.input_q.qsize() >= FRAME_QUEUE_MAX:
            try: st.input_q.get_nowait()
            except Empty: break
        st.input_q.put_nowait(('webcam', data['frame']))
    except Exception:
        pass
    if not st.processing:
        st.processing = True
        eventlet.spawn_n(infer_workers, sid, st)
        eventlet.spawn_n(emitter_loop, sid, st)

@socketio.on('process_video')
def on_process_video(data):
    ensure_engine()
    sid = request.sid
    st = clients.get(sid)
    if not st:
        return
    path = data.get('filepath')
    if not path or not os.path.exists(path):
        emit('error', {'message':'Video not found'})
        return
    st.stop = False
    if not st.processing:
        st.processing = True
        eventlet.spawn_n(infer_workers, sid, st)
        eventlet.spawn_n(emitter_loop, sid, st)
    eventlet.spawn_n(video_producer, sid, st, path)

def _choose_stride(total_frames:int, fps:float) -> int:
    duration = (total_frames / fps) if fps > 0 else 10.0
    if duration <= 12:
        return 2
    elif duration <= 30:
        return 2
    elif duration <= 60:
        return 3
    else:
        return 4

def video_producer(sid, st:ClientState, path:str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        socketio.emit('error', {'message':'Cannot open video'}, room=sid)
        st.stop = True
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = _choose_stride(total, fps)
    idx = 0
    last_progress_emit = 0.0
    socketio.emit('video_meta', {'fps': fps, 'total_frames': total, 'stride': stride}, room=sid)
    logger.info(f"Video processing: total={total}, fps={fps}, stride={stride}")
    try:
        while not st.stop and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % stride != 0:
                continue
            try:
                if st.input_q.qsize() >= FRAME_QUEUE_MAX:
                    continue
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                st.input_q.put_nowait(('video', base64.b64encode(buf).decode('utf-8')))
            except Exception:
                pass
            now = time.time()
            if now - last_progress_emit > 0.33:
                pct = (idx/total*100) if total else 0
                socketio.emit('video_progress', {'progress': pct, 'frame_number': idx}, room=sid)
                last_progress_emit = now
            if now - st.last_processed_ts > WATCHDOG_SECS and stride < 5:
                stride += 1
                socketio.emit('video_info', {'message': f'Auto-increasing stride to {stride} for speed'}, room=sid)
    finally:
        cap.release()
        try: os.remove(path)
        except: pass
        st.stop = True
        socketio.emit('video_complete', {'message':'Video processing complete'}, room=sid)

def infer_workers(sid, st:ClientState):
    eng = ensure_engine()
    def worker_loop():
        while not st.stop:
            try:
                items = []
                item = st.input_q.get(timeout=1.0)
                items.append(item)
                try:
                    items.append(st.input_q.get_nowait())
                except Empty:
                    pass
            except Empty:
                continue
            for kind, payload in items:
                try:
                    if kind == 'webcam':
                        image_data = payload.split(',')[1] if ',' in payload else payload
                        img_bytes = base64.b64decode(image_data)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    else:
                        img_bytes = base64.b64decode(payload)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    b64, tracks = eng.process_frame(frame)
                    st.last_processed_ts = time.time()
                    try:
                        while st.output_q.qsize() >= FRAME_QUEUE_MAX:
                            try: st.output_q.get_nowait()
                            except Empty: break
                        st.output_q.put_nowait((b64, tracks))
                    except Exception:
                        pass
                except Exception:
                    continue
    for _ in range(INFER_WORKERS):
        eventlet.spawn_n(worker_loop)

def emitter_loop(sid, st:ClientState):
    emit_interval = 1.0 / EMIT_FPS_CAP
    while not st.stop:
        try:
            b64, tracks = st.output_q.get(timeout=1.0)
        except Empty:
            continue
        now = time.time()
        if now - st.last_emit_ts < emit_interval:
            continue
        try:
            socketio.emit('processed_frame', {'frame': f"data:image/jpeg;base64,{b64}", 'detections': tracks, 'realtime': True, 'model_loaded': True}, room=sid)
            # Emit a lightweight status ping to flip any pending UI indicators
            socketio.emit('connection_status', {'status':'connected','engine_ready': True, 'model_loaded': True}, room=sid)
            st.last_emit_ts = now
        except Exception:
            pass

@socketio.on('stop_processing')
def stop_processing():
    sid = request.sid
    st = clients.get(sid)
    if st:
        st.stop = True
    emit('processing_stopped', {'status':'stopped'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    logger.info(f"Starting realtime server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)