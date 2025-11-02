# Real-Time Object Detection (Local Only)

A minimal setup to run real-time object detection locally using YOLOv8 and OpenCV.

## Prerequisites

- Python 3.8+ (3.11 recommended)
- Webcam connected to your machine
- pip installed

## Setup

```bash
# Clone the repository
git clone https://github.com/aravinditte/realtime-object-recognition.git
cd realtime-object-recognition

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the MVP (Webcam)

```bash
python detect.py
```

- A window will open showing your webcam with bounding boxes and labels.
- Press `q` to quit.

## Run the Web App (Optional, Local Only)

```bash
# Default
python app.py

# Or with environment overrides
HOST=127.0.0.1 PORT=5000 MODEL_NAME=yolov8n.pt python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Troubleshooting

- If the webcam does not open, try changing the camera index in `detect.py` from `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
- If model download is slow, ensure internet connectivity; the first run downloads `yolov8n.pt`.
- On Linux, if OpenCV errors mention GL, install: `sudo apt-get install libgl1 libglib2.0-0 libsm6 libxext6`.

## Uninstall / Clean

```bash
deactivate  # exit venv
rm -rf venv uploads logs __pycache__
```
