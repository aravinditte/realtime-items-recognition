#!/usr/bin/env python3
"""
Production runner for Real-Time Object Detection System

This script provides a production-ready way to run the application with
proper configuration, logging, and error handling.

Usage:
    python run.py [--host HOST] [--port PORT] [--debug] [--model MODEL]

Author: Aravind Itte
Date: November 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import signal
import threading
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app import app, socketio, detector, logger as app_logger
except ImportError as e:
    print(f"Error importing application: {e}")
    print("Make sure to run 'python setup.py' first to install dependencies")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionRunner:
    """
    Production application runner with monitoring and graceful shutdown
    """
    
    def __init__(self):
        self.running = False
        self.start_time = None
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def health_monitor(self):
        """Monitor application health"""
        while self.running:
            try:
                # Check if model is loaded
                if detector.model is None:
                    logger.warning("YOLO model is not loaded")
                
                # Check memory usage (basic check)
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > 2048:  # 2GB threshold
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                
                # Log uptime every hour
                uptime = time.time() - self.start_time
                if uptime > 0 and int(uptime) % 3600 == 0:  # Every hour
                    hours = int(uptime // 3600)
                    logger.info(f"Application uptime: {hours} hours")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(60)
    
    def run(self, host='0.0.0.0', port=5000, debug=False, model=None):
        """Run the application"""
        logger.info("Starting Real-Time Object Detection System...")
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Check if model is loaded
        if detector.model is None:
            logger.error("YOLO model failed to load. Cannot start application.")
            return False
        
        # Log configuration
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        logger.info(f"Debug: {debug}")
        logger.info(f"Model: {detector.model_name}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        self.running = True
        self.start_time = time.time()
        
        # Start health monitor in background
        if not debug:
            health_thread = threading.Thread(target=self.health_monitor, daemon=True)
            health_thread.start()
            logger.info("Health monitor started")
        
        try:
            # Start the application
            logger.info(f"Application starting on http://{host}:{port}")
            socketio.run(app, host=host, port=port, debug=debug)
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            return False
        
        finally:
            self.shutdown()
        
        return True
    
    def shutdown(self):
        """Graceful shutdown"""
        if self.running:
            logger.info("Shutting down application...")
            self.running = False
            
            # Calculate uptime
            if self.start_time:
                uptime = time.time() - self.start_time
                logger.info(f"Total uptime: {uptime:.1f} seconds")
            
            logger.info("Application shutdown complete")

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Real-Time Object Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --host 127.0.0.1 --port 8000
  python run.py --debug --model yolov8s.pt
  python run.py --host 0.0.0.0 --port 5000 --model yolov8n.pt

Environment Variables:
  HOST              - Server host (default: 0.0.0.0)
  PORT              - Server port (default: 5000)
  FLASK_ENV         - Flask environment (development/production)
  MODEL_NAME        - YOLO model to use (default: yolov8n.pt)
  SECRET_KEY        - Flask secret key
        """
    )
    
    parser.add_argument(
        '--host', 
        default=os.environ.get('HOST', '0.0.0.0'),
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=int(os.environ.get('PORT', 5000)),
        help='Port to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        default=os.environ.get('FLASK_ENV') == 'development',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--model', 
        default=os.environ.get('MODEL_NAME', 'yolov8n.pt'),
        help='YOLO model to use (default: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--check-health', 
        action='store_true',
        help='Check application health and exit'
    )
    
    parser.add_argument(
        '--version', 
        action='version',
        version='Real-Time Object Detection System v1.0.0'
    )
    
    return parser

def check_health():
    """Check application health"""
    try:
        import requests
        response = requests.get('http://localhost:5000/health', timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"Status: {health_data.get('status', 'unknown')}")
            print(f"Model loaded: {health_data.get('model_loaded', False)}")
            print(f"Connections: {health_data.get('connections', 0)}")
            return True
        else:
            print(f"Health check failed: HTTP {response.status_code}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle health check
    if args.check_health:
        success = check_health()
        sys.exit(0 if success else 1)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create runner and start application
    runner = ProductionRunner()
    
    try:
        success = runner.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            model=args.model
        )
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
