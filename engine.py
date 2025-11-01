#!/usr/bin/env python3
"""
Realtime object recognition server (YOLOv8n) with SORT tracking and overlay.
This module overrides key parts to integrate tracker and draw live overlays.
"""
import os
import cv2
import base64
import numpy as np
import time
from ultralytics import YOLO
from tracker import SortTracker
from overlay import draw_tracks, draw_info_panel

# Defaults (no external envs required)
CONF = 0.30
IOU = 0.45
INFER_W, INFER_H = 640, 384

class RealtimeEngine:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
        self.tracker = SortTracker(max_age=10, min_hits=2, iou_threshold=0.3)
        self.last_ts = time.time()
        self.fps = 0
        self.counts = {}

    def _run_yolo(self, frame):
        resized = cv2.resize(frame, (INFER_W, INFER_H))
        results = self.model(resized, conf=CONF, iou=IOU, verbose=False)
        h, w = frame.shape[:2]
        sx, sy = w/INFER_W, h/INFER_H
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                conf = float(b.conf[0].cpu().numpy())
                cls  = int(b.cls[0].cpu().numpy())
                name = self.model.names.get(cls, str(cls))
                dets.append({'bbox':[x1,y1,x2,y2],'cls':name,'conf':conf})
        return dets

    def process_frame(self, frame):
        # 1) Detection
        dets = self._run_yolo(frame)
        # 2) Tracking
        tracks = self.tracker.update(dets)
        # 3) Update simple counts
        for t in tracks:
            name = t.get('cls') or 'obj'
            self.counts[name] = self.counts.get(name, 0) + 0  # keep cumulative if needed
        # 4) FPS
        now = time.time()
        dt = now - self.last_ts
        if dt > 0:
            self.fps = 1.0/dt
        self.last_ts = now
        # 5) Overlay
        annotated = frame.copy()
        annotated = draw_tracks(annotated, tracks)
        annotated = draw_info_panel(annotated, {'fps':self.fps, 'active':len(tracks)})
        # 6) Encode
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf).decode('utf-8')
        return b64, tracks
