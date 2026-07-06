# RescueVision: Drone-Based Human & Animal Detection System (Nidar)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-yellow.svg" alt="YOLO">
</div>

## Problem Statement
During natural disasters (earthquakes, floods, landslides), locating survivors—both human and animal—is a time-critical challenge. Ground access is often blocked by destroyed infrastructure or hazardous terrain, making traditional search-and-rescue operations slow, dangerous, and sometimes impossible, leading to a loss of critical time for victims.

## Solution
RescueVision (Nidar) is an AI-powered disaster response system designed to be deployed on drones. It leverages state-of-the-art computer vision (YOLO) to detect and classify humans and animals in real-time from aerial imagery. By integrating GPS tracking, the system maps detected entities and provides precise coordinates (rescue waypoints), enabling faster, safer, and more targeted rescue operations.

## Architecture
1. **Input Layer**: Drone camera feed (Real-time video) or pre-captured aerial images.
2. **Processing Core**: YOLOv8 Object Detection Model (Custom trained for disaster scenarios).
3. **Tracking & Logic Module**: Unique object tracking to prevent duplicate counting and a GPS Location Tracker to map coordinates of detected victims.
4. **Output/UI Layer**: OpenCV-based annotated video feed, interactive CLI for mode selection, and generation of rescue reports and waypoints.

## Key Features
- **Real-Time Detection**: Swiftly identifies humans and animals using high-performance YOLO models.
- **Unique Object Tracking**: Intelligent tracking algorithms follow detected entities to ensure accurate counts and prevent duplicate logging of the same victim.
- **GPS Location Integration**: Logs the precise geographical coordinates of detected targets to facilitate immediate rescue dispatch.
- **Multi-Mode Operation**: Supports real-time camera feeds, processing of pre-clicked images, and model testing environments.
- **Priority Assessment**: Automatically assigns priority levels to detections (e.g., humans as high priority, animals as medium).
- **Comprehensive Reporting**: Generates detailed logs, captures, and unique object summaries for post-mission analysis.

## Folder Structure
```text
.
├── disaster_drone_models/    # Custom trained YOLO models for disaster rescue
├── output/                   # Generated reports, logs, and captured images
│   ├── captures/             # Saved detection frames
│   └── logs/                 # Detection logs and GPS waypoint records
├── runs/                     # YOLO training/validation output directories
├── __pycache__/              # Python compiled files
├── basic_camera_test.py      # Script to test basic camera functionality
├── benchmark.py              # Performance benchmarking script
├── complete_system.py        # Main execution script integrating detection and tracking
├── dataset_manager.py        # Utility to manage and format the dataset
├── enhanced_detection.py     # Script containing advanced detection logic
├── gps_location_tracker.py   # Module handling GPS data mapping and waypoint generation
├── image_test.py             # Script for testing model on static images
├── train_model.py            # Script used to train the custom YOLO model
├── yolov8n.pt                # Base YOLOv8 nano model weights
└── README.md                 # Project documentation
```