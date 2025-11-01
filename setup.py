#!/usr/bin/env python3
"""
Setup script for Real-Time Object Detection System

This script handles the initial setup and configuration of the system,
including model downloads, dependency verification, and environment setup.

Usage:
    python setup.py

Author: Aravind Itte
Date: November 2024
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemSetup:
    """
    System setup and configuration class
    """
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system().lower()
        self.required_python = (3, 8)
        self.project_root = Path(__file__).parent
        
    def check_python_version(self):
        """Check if Python version meets requirements"""
        logger.info(f"Checking Python version: {self.python_version[:2]}")
        
        if self.python_version[:2] < self.required_python:
            logger.error(f"Python {self.required_python[0]}.{self.required_python[1]}+ is required")
            logger.error(f"Current version: {self.python_version[0]}.{self.python_version[1]}")
            return False
        
        logger.info("Python version check passed")
        return True
    
    def check_system_dependencies(self):
        """Check system-level dependencies"""
        logger.info("Checking system dependencies...")
        
        if self.platform == 'linux':
            required_packages = [
                'libgl1-mesa-glx',
                'libglib2.0-0', 
                'libsm6',
                'libxext6',
                'libxrender1'
            ]
            
            logger.info("On Linux systems, ensure these packages are installed:")
            for package in required_packages:
                logger.info(f"  - {package}")
            
            logger.info("Install with: sudo apt-get install " + ' '.join(required_packages))
        
        elif self.platform == 'darwin':
            logger.info("On macOS, ensure Xcode command line tools are installed")
            logger.info("Install with: xcode-select --install")
        
        elif self.platform == 'windows':
            logger.info("On Windows, ensure Visual C++ Redistributable is installed")
        
        return True
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False
        
        try:
            # Upgrade pip first
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])
            
            # Install requirements
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ])
            
            logger.info("Python dependencies installed successfully")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def download_models(self):
        """Download YOLO models"""
        logger.info("Downloading YOLO models...")
        
        models = ['yolov8n.pt', 'yolov8s.pt']
        
        try:
            # Import after dependencies are installed
            from ultralytics import YOLO
            
            for model_name in models:
                logger.info(f"Downloading {model_name}...")
                model = YOLO(model_name)
                logger.info(f"{model_name} downloaded successfully")
            
            logger.info("All models downloaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download models: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating project directories...")
        
        directories = [
            'uploads',
            'models',
            'logs',
            'temp'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create .gitkeep files to ensure directories are tracked
        for directory in directories:
            gitkeep_file = self.project_root / directory / '.gitkeep'
            gitkeep_file.touch()
        
        logger.info("Project directories created successfully")
        return True
    
    def setup_environment(self):
        """Setup environment variables"""
        logger.info("Setting up environment configuration...")
        
        env_example = self.project_root / '.env.example'
        env_file = self.project_root / '.env'
        
        if not env_file.exists() and env_example.exists():
            # Copy example environment file
            import shutil
            shutil.copy(env_example, env_file)
            logger.info("Created .env file from .env.example")
            logger.info("Please review and update .env file with your configuration")
        
        return True
    
    def verify_installation(self):
        """Verify the installation"""
        logger.info("Verifying installation...")
        
        try:
            # Test imports
            import cv2
            import numpy as np
            from flask import Flask
            from ultralytics import YOLO
            
            logger.info(f"OpenCV version: {cv2.__version__}")
            logger.info(f"NumPy version: {np.__version__}")
            
            # Test YOLO model loading
            model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded successfully")
            
            # Test basic detection
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(test_image)
            logger.info("Test detection completed successfully")
            
            logger.info("Installation verification passed")
            return True
        
        except Exception as e:
            logger.error(f"Installation verification failed: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process"""
        logger.info("Starting Real-Time Object Detection System setup...")
        
        steps = [
            ("Python version check", self.check_python_version),
            ("System dependencies check", self.check_system_dependencies),
            ("Directory creation", self.create_directories),
            ("Python dependencies installation", self.install_python_dependencies),
            ("Model download", self.download_models),
            ("Environment setup", self.setup_environment),
            ("Installation verification", self.verify_installation)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Step: {step_name}")
            logger.info(f"{'='*50}")
            
            if not step_function():
                logger.error(f"Setup failed at step: {step_name}")
                return False
        
        logger.info(f"\n{'='*50}")
        logger.info("Setup completed successfully!")
        logger.info(f"{'='*50}")
        logger.info("\nNext steps:")
        logger.info("1. Review the .env file and update configuration if needed")
        logger.info("2. Run the MVP script: python detect.py")
        logger.info("3. Run the web application: python app.py")
        logger.info("4. Open your browser and go to http://localhost:5000")
        logger.info("\nFor Docker deployment:")
        logger.info("1. Build the image: docker build -t realtime-object-detection .")
        logger.info("2. Run the container: docker run -p 5000:5000 realtime-object-detection")
        
        return True

def main():
    """Main setup function"""
    setup = SystemSetup()
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
