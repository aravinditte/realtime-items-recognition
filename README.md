# Real-Time Object Detection Web Application

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![YOLO](https://img.shields.io/badge/YOLO-v8/v11-orange.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A powerful, real-time object detection web application that recognizes and tracks objects in live camera feeds, uploaded videos, and images using state-of-the-art YOLO models.

## Features

- **Real-Time Detection**: Live camera feed processing with WebSocket streaming
- **Video Processing**: Upload and analyze video files with object tracking
- **Image Analysis**: Single image processing with bounding box annotations
- **Advanced AI**: Powered by YOLOv8/YOLOv11 models for 80+ object classes
- **High Performance**: Optimized for low-latency real-time processing
- **Responsive Design**: Modern UI that works on all devices
- **Secure**: Containerized deployment with security best practices
- **Zero Cost Deployment**: Multiple free hosting options available

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/aravinditte/realtime-items-recognition.git
cd realtime-items-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open your browser and navigate to `http://localhost:5000`

### Docker

```bash
# Build and run with Docker
docker build -t realtime-object-detection .
docker run -p 5000:5000 realtime-object-detection

# Or use Docker Compose
docker-compose up
```

## Technology Stack

### Backend
- **Framework**: Flask 3.0+ with SocketIO for real-time communication
- **AI/ML**: Ultralytics YOLO (v8/v11), OpenCV, TensorFlow
- **Async**: Eventlet for WebSocket handling
- **Security**: Non-root containers, input validation, CORS protection

### Frontend
- **UI Framework**: Bootstrap 5.3 with custom CSS
- **Real-time**: Socket.IO client for WebSocket communication
- **Icons**: Font Awesome 6.4
- **Animations**: Animate.css for smooth transitions

### Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Monitoring**: Health checks and logging

## Supported Object Classes

The application can detect 80+ object classes from the COCO dataset including:

- **People**: person
- **Vehicles**: car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: bicycle, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard
- **Electronics**: cell phone, laptop, mouse, remote, keyboard, microwave, oven
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza
- **Furniture**: chair, couch, potted plant, bed, dining table, toilet
- **Sports**: sports ball, kite, baseball bat, baseball glove, skateboard, surfboard
- **And many more...**

## Usage Modes

### 1. Live Camera Detection
- Click "Start Camera" to begin real-time detection
- View live feed with bounding boxes and labels
- Monitor FPS and detection statistics
- Support for multiple camera sources

### 2. Video Upload Processing
- Drag & drop or browse video files (MP4, AVI, MOV, MKV, WebM)
- Real-time processing with progress tracking
- Frame-by-frame analysis with object tracking
- Maximum file size: 50MB

### 3. Image Analysis
- Upload images (JPG, PNG, BMP, TIFF)
- Instant processing with bounding box annotations
- Confidence scores and object classification
- Download processed results

## Configuration

### Environment Variables

```bash
# Application Settings
FLASK_ENV=production
HOST=0.0.0.0
PORT=5000
SECRET_KEY=your-secret-key

# AI Model Settings
MODEL_NAME=yolov8n.pt  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Performance Settings
MAX_CONTENT_LENGTH=52428800  # 50MB
FRAME_SKIP=2  # Process every nth frame for performance
```

### Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6MB | Fastest | Good | Real-time, mobile |
| YOLOv8s | 22MB | Fast | Better | Balanced performance |
| YOLOv8m | 52MB | Medium | High | High accuracy needs |
| YOLOv8l | 87MB | Slow | Higher | Production quality |
| YOLOv8x | 136MB | Slowest | Highest | Maximum accuracy |

## Performance Optimization

### System Requirements

- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB RAM, 2 CPU cores
- **GPU**: Optional but recommended for better performance
- **Storage**: 2GB free space (including model downloads)

### Optimization Tips

1. **Model Selection**: Use YOLOv8n for real-time applications
2. **Frame Processing**: Adjust frame skip rate based on CPU capacity
3. **Resolution**: Lower input resolution for better FPS
4. **Batch Processing**: Process multiple objects in single inference
5. **Caching**: Enable model caching for faster startup

## Security Features

- **Container Security**: Non-root user execution
- **Input Validation**: File type and size restrictions
- **CORS Protection**: Configurable cross-origin policies
- **Rate Limiting**: WebSocket connection limits
- **Health Monitoring**: Automated health checks
- **Secure Headers**: Security-focused HTTP headers

## Monitoring & Analytics

### Built-in Metrics
- Real-time FPS monitoring
- Detection accuracy statistics
- Processing time analytics
- Connection status tracking
- Error rate monitoring

### Health Endpoints
- `GET /health` - Application health status
- WebSocket connection monitoring

## Testing

```bash
# Test model loading
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Test Flask app
python -c "from app import app; print('App imports successfully')"

# Load testing (install hey first)
hey -n 1000 -c 10 http://localhost:5000/health
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

## API Documentation

### REST Endpoints

```
GET  /                 - Main application page
GET  /health          - Health check endpoint
POST /upload          - File upload for processing
```

### WebSocket Events

```
# Client to Server
start_webcam          - Start camera streaming
stop_webcam           - Stop camera streaming
process_video         - Process uploaded video

# Server to Client
connection_status     - Connection established
frame_data           - Processed frame data
webcam_status        - Camera status updates
error                - Error notifications
video_complete       - Video processing finished
```

## Troubleshooting

### Common Issues

1. **Camera Access Denied**
   ```bash
   # Check camera permissions in browser
   # Ensure HTTPS for production deployment
   ```

2. **Model Loading Fails**
   ```bash
   # Clear model cache
   rm -rf ~/.cache/torch/hub/
   
   # Reinstall ultralytics
   pip uninstall ultralytics
   pip install ultralytics
   ```

3. **WebSocket Connection Issues**
   ```bash
   # Check CORS settings
   # Verify SocketIO version compatibility
   # Test with different browsers
   ```

4. **Memory Issues**
   ```bash
   # Use smaller model (yolov8n.pt)
   # Reduce frame processing rate
   # Increase system memory
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Aravind Itte**
- GitHub: [@aravinditte](https://github.com/aravinditte)
- LinkedIn: [Connect with me](https://linkedin.com/in/aravinditte)
- Email: aravinditte@gmail.com

## Acknowledgments

- [Ultralytics](https://ultralytics.com) for the amazing YOLO models
- [OpenCV](https://opencv.org) for computer vision capabilities
- [Flask](https://flask.palletsprojects.com) for the web framework
- [Socket.IO](https://socket.io) for real-time communication
- [Bootstrap](https://getbootstrap.com) for the responsive UI

---

<p align="center">
  Made with love by <a href="https://github.com/aravinditte">Aravind Itte</a>
</p>
