from ultralytics import YOLO
import cv2
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import math
import os
import glob
from tkinter import filedialog, messagebox
import tkinter as tk
from gps_location_tracker import GPSLocationTracker


class CompleteDroneDetectionSystem:
    def __init__(self, model_path=r"C:\Users\vedan\Documents\My_Projects\Nidar_drone_project\disaster_drone_models\models\disaster_rescue_v1\weights\best.pt", confidence_threshold=0.5,enable_gps=True):
        """Complete detection system for drone deployment with unique tracking"""
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Available model files:")
            # Try to find alternative model files
            possible_paths = [
                "best.pt",
                "yolov8n.pt", 
                "yolov8s.pt",
                "yolov8m.pt",
                "disaster_drone_models/streaming_human_animal_v1/weights/best.pt",
                "runs/detect/train/weights/best.pt"
            ]
            
            found_model = None
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"  Found: {path}")
                    if found_model is None:
                        found_model = path
                else:
                    print(f"  Not found: {path}")
            
            if found_model:
                print(f"Using alternative model: {found_model}")
                model_path = found_model
            else:
                print("No model found! Please ensure you have a trained model or download a YOLO model.")
                print("You can download a basic YOLO model by running:")
                print("  from ultralytics import YOLO; YOLO('yolov8n.pt')")
                raise FileNotFoundError(f"No valid model found")

        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully from: {model_path}")
            print(f"Model classes: {self.model.names}")
            print(f"Number of classes: {len(self.model.names)}")
            
            # Check if this is our custom trained model or a standard YOLO model
            model_classes = list(self.model.names.values()) if hasattr(self.model.names, 'values') else list(self.model.names)
            self.is_custom_model = 'human' in model_classes and 'animal' in model_classes
            
            if self.is_custom_model:
                print("🎯 Custom disaster drone model detected!")
                print("Classes: human=0, animal=1")
            else:
                print("🔄 Standard YOLO model detected - will map to disaster classes")
                print("Available classes:", model_classes)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        print("=" * 50)
    
        self.confidence_threshold = confidence_threshold
        
        # Priority classes for disaster rescue - Updated for both custom and standard models
        if self.is_custom_model:
            # Custom trained model classes
            self.priority_classes = {
                0: {'name': 'human', 'priority': 1, 'color': (0, 0, 255)},   # Red - Highest
                1: {'name': 'animal', 'priority': 2, 'color': (0, 255, 255)}  # Yellow
            }
        else:
            # Standard COCO/YOLO model classes - map relevant classes to our priorities
            self.priority_classes = {
                0: {'name': 'person', 'priority': 1, 'color': (0, 0, 255)},    # person - Red
                14: {'name': 'bird', 'priority': 2, 'color': (0, 255, 255)},   # bird - Yellow  
                15: {'name': 'cat', 'priority': 2, 'color': (0, 255, 255)},    # cat - Yellow
                16: {'name': 'dog', 'priority': 2, 'color': (0, 255, 255)},    # dog - Yellow
                17: {'name': 'horse', 'priority': 2, 'color': (0, 255, 255)},  # horse - Yellow
                18: {'name': 'sheep', 'priority': 2, 'color': (0, 255, 255)},  # sheep - Yellow
                19: {'name': 'cow', 'priority': 2, 'color': (0, 255, 255)},    # cow - Yellow
                20: {'name': 'elephant', 'priority': 2, 'color': (0, 255, 255)}, # elephant - Yellow
                21: {'name': 'bear', 'priority': 2, 'color': (0, 255, 255)},   # bear - Yellow
                22: {'name': 'zebra', 'priority': 2, 'color': (0, 255, 255)},  # zebra - Yellow
                23: {'name': 'giraffe', 'priority': 2, 'color': (0, 255, 255)}, # giraffe - Yellow
            }
        
        # Unique tracking system
        self.unique_objects = {}  # Store unique detected objects
        self.next_object_id = 1
        self.tracking_threshold = 100  # Max distance for same object (pixels)
        self.max_missed_frames = 30   # Max frames to keep tracking inactive objects
        
        # Detection tracking
        self.detection_history = []
        self.active_detections = {}
        
        # Image processing settings
        self.image_display_time = 3000  # Time to display each image in milliseconds
        
        # Create output directories
        self.setup_directories()

        #GPS
        self.enable_gps = enable_gps
        if self.enable_gps:
            self.gps_tracker = GPSLocationTracker(output_dir=str(self.output_dir))
            print("GPS Location Tracking: ENABLED")
        else:
            self.gps_tracker = None
            print("GPS Location Tracking: DISABLED")

    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path('output')
        self.logs_dir = self.output_dir / 'logs'
        self.captures_dir = self.output_dir / 'captures'
        
        for dir_path in [self.logs_dir, self.captures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def choose_input_mode(self):
        """Interactive function to choose between camera and images"""
        print("\n" + "="*60)
        print("DISASTER RESCUE DETECTION SYSTEM - INPUT MODE SELECTION")
        print("="*60)
        print("Choose input mode:")
        print("1. Real-time Camera Detection")
        print("2. Process Pre-clicked Images")
        print("3. Test Model with Sample Detection")
        print("="*60)
        
        while True:
            try:
                choice = input("Enter your choice (1, 2, or 3): ").strip()
                if choice == '1':
                    return 'camera'
                elif choice == '2':
                    return 'images'
                elif choice == '3':
                    return 'test'
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\nExiting...")
                return None
    
    def test_model(self):
        """Test the model with a simple detection to verify it works"""
        print("\n=== MODEL TESTING MODE ===")
        print("This will test your model with the camera to verify it's working correctly.")
        print("Controls: 'q' to quit, 's' to save test image")
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera for testing")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame_count += 1
                
                # Run detection with verbose output for testing
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Process and display results
                annotated_frame = frame.copy()
                detection_count = 0
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        
                        # Check if this class is in our priority classes
                        if class_id in self.priority_classes:
                            detection_count += 1
                            
                            # Draw bounding box
                            bbox = box.xyxy.cpu().numpy().tolist()[0]
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            color = self.priority_classes[class_id]['color']
                            name = self.priority_classes[class_id]['name']
                            priority = self.priority_classes[class_id]['priority']
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Label
                            label = f"{name} {confidence:.2f} (P{priority})"
                            cv2.putText(annotated_frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw test info
                test_info = [
                    f"MODEL TEST MODE",
                    f"Frame: {frame_count}",
                    f"Detections: {detection_count}",
                    f"Model: {'Custom Disaster' if self.is_custom_model else 'Standard YOLO'}",
                    f"Confidence: {self.confidence_threshold}",
                    f"Press 'q' to quit, 's' to save"
                ]
                
                # Background for info
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                for i, info in enumerate(test_info):
                    color = (0, 255, 0) if detection_count > 0 and i == 2 else (255, 255, 255)
                    cv2.putText(annotated_frame, info, (20, 35 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.imshow('Model Test - Disaster Detection System', annotated_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save test image
                    test_path = self.captures_dir / f"model_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(test_path), annotated_frame)
                    print(f"Test image saved: {test_path}")
                
                # Print detection info every 30 frames
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: {detection_count} relevant detections found")
                    
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Model test completed. Processed {frame_count} frames.")
    
    def get_camera_source(self):
        """Get camera source from user"""
        print("\nCamera Detection Mode Selected")
        print("Options:")
        print("0 - Default camera")
        print("1 - External camera")
        print("Or enter video file path")
        
        source = input("Enter camera source (default: 0): ").strip()
        if source == '':
            source = '0'
        
        return int(source) if source.isdigit() else source
    
    def get_image_sources(self):
        """Get image folder or files from user with file explorer"""
        print("\nImage Processing Mode Selected")
        print("Options:")
        print("1. Select a folder containing images")
        print("2. Select specific image files")
        
        while True:
            try:
                choice = input("Enter your choice (1 or 2): ").strip()
                if choice == '1':
                    return self.select_folder_with_explorer()
                elif choice == '2':
                    return self.select_files_with_explorer()
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except KeyboardInterrupt:
                return []
    
    def select_folder_with_explorer(self):
        """Open file explorer to select a folder"""
        print("Opening file explorer to select folder...")
        
        # Create a temporary root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.lift()      # Bring to front
        root.attributes('-topmost', True)  # Keep on top
        
        try:
            folder_path = filedialog.askdirectory(
                title="Select folder containing images",
                initialdir=os.getcwd()
            )
            
            if not folder_path:
                print("No folder selected.")
                return []
            
            print(f"Selected folder: {folder_path}")
            
            # Find all supported image files
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            image_files = []
            
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if not image_files:
                messagebox.showwarning("No Images Found", 
                                     f"No supported image files found in the selected folder.\n"
                                     f"Supported formats: {', '.join([ext[2:] for ext in extensions])}")
                return []
            
            print(f"Found {len(image_files)} images in the selected folder")
            return sorted(image_files)
            
        except Exception as e:
            print(f"Error selecting folder: {str(e)}")
            return []
        finally:
            root.destroy()
    
    def select_files_with_explorer(self):
        """Open file explorer to select specific image files"""
        print("Opening file explorer to select image files...")
        
        # Create a temporary root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.lift()      # Bring to front
        root.attributes('-topmost', True)  # Keep on top
        
        try:
            image_files = filedialog.askopenfilenames(
                title="Select image files",
                initialdir=os.getcwd(),
                filetypes=[
                    ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                    ("JPEG Files", "*.jpg *.jpeg"),
                    ("PNG Files", "*.png"),
                    ("BMP Files", "*.bmp"),
                    ("TIFF Files", "*.tiff *.tif"),
                    ("All Files", "*.*")
                ]
            )
            
            if not image_files:
                print("No files selected.")
                return []
            
            print(f"Selected {len(image_files)} image files:")
            for i, file_path in enumerate(image_files, 1):
                print(f"  {i}. {os.path.basename(file_path)}")
            
            return list(image_files)
            
        except Exception as e:
            print(f"Error selecting files: {str(e)}")
            return []
        finally:
            root.destroy()
    
    def calculate_distance(self, center1, center2):
        """Calculate Euclidean distance between two centers"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_bbox_area(self, bbox):
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def find_matching_object(self, detection, frame_number):
        """Find if this detection matches an existing tracked object"""
        detection_center = detection['center']
        detection_class = detection['class_id']
        detection_area = self.calculate_bbox_area(detection['bbox'])
        
        best_match_id = None
        min_distance = float('inf')
        
        for obj_id, obj_data in self.unique_objects.items():
            # Only match same class
            if obj_data['class_id'] != detection_class:
                continue
            
            # Check if object was seen recently (within max_missed_frames)
            if frame_number - obj_data['last_frame'] > self.max_missed_frames:
                continue
            
            # Calculate distance from last known position
            distance = self.calculate_distance(detection_center, obj_data['last_center'])
            
            # Check area similarity (helps distinguish between different objects)
            area_ratio = min(detection_area, obj_data['last_area']) / max(detection_area, obj_data['last_area'])
            
            # Consider it a match if distance is small and area is similar
            if distance < self.tracking_threshold and area_ratio > 0.5 and distance < min_distance:
                min_distance = distance
                best_match_id = obj_id
        
        return best_match_id
    
    def update_or_create_object(self, detection, frame_number):
        """Update existing object or create new unique object"""
        matching_id = self.find_matching_object(detection, frame_number)
        
        if matching_id is not None:
            # Update existing object
            self.unique_objects[matching_id].update({
                'last_center': detection['center'],
                'last_frame': frame_number,
                'last_confidence': detection['confidence'],
                'last_bbox': detection['bbox'],
                'last_area': self.calculate_bbox_area(detection['bbox']),
                'total_detections': self.unique_objects[matching_id]['total_detections'] + 1,
                'last_seen': detection['timestamp'],
                'max_confidence': max(self.unique_objects[matching_id]['max_confidence'], detection['confidence'])
            })
            detection['object_id'] = matching_id
            detection['is_new'] = False
        else:
            # Create new unique object
            obj_id = self.next_object_id
            self.next_object_id += 1
            
            self.unique_objects[obj_id] = {
                'id': obj_id,
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'priority': detection['priority'],
                'first_seen': detection['timestamp'],
                'last_seen': detection['timestamp'],
                'first_frame': frame_number,
                'last_frame': frame_number,
                'first_center': detection['center'],
                'last_center': detection['center'],
                'last_confidence': detection['confidence'],
                'last_bbox': detection['bbox'],
                'last_area': self.calculate_bbox_area(detection['bbox']),
                'total_detections': 1,
                'max_confidence': detection['confidence']
            }
            
            detection['object_id'] = obj_id
            detection['is_new'] = True
            
            # Log new unique detection
            priority_text = "HIGH PRIORITY" if detection['priority'] == 1 else "MEDIUM PRIORITY"
            print(f"🆕 NEW {detection['class_name'].upper()} DETECTED - ID: {obj_id} ({priority_text})")
    
    def process_images(self, image_files):
        """Process pre-clicked images"""
        if not image_files:
            print("No images to process.")
            return
        
        print(f"\n=== PROCESSING {len(image_files)} IMAGES ===")
        print("Controls:")
        print("  'n' or SPACE - Next image")
        print("  'p' - Previous image") 
        print("  's' - Save current detection")
        print("  'r' - Generate report")
        print("  'u' - Show unique objects summary")
        print("  'q' - Quit")
        print("  Or wait 3 seconds for auto-advance")
        
        current_index = 0
        new_high_priority_count = 0
        
        try:
            while current_index < len(image_files):
                image_path = image_files[current_index]
                print(f"\nProcessing: {os.path.basename(image_path)} ({current_index + 1}/{len(image_files)})")
                
                # Load image
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Error: Could not load image '{image_path}'")
                    current_index += 1
                    continue
                
                # Run detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Process detections with unique tracking
                current_detections = self.process_detections_with_tracking(results[0], current_index + 1)
                
                # Count new high priority detections
                new_high_priority = len([d for d in current_detections if d['priority'] == 1 and d['is_new']])
                new_high_priority_count += new_high_priority
                
                # Draw enhanced annotations
                annotated_frame = self.draw_enhanced_annotations_with_tracking(frame, current_detections)
                
                # Display status (modified for image processing)
                self.draw_image_status_info(annotated_frame, current_index + 1, len(image_files), 
                                          new_high_priority_count, os.path.basename(image_path))
                
                # Auto-save new high priority detections
                if new_high_priority > 0:
                    self.save_high_priority_detection(annotated_frame, current_detections, image_path)
                
                # Display image
                cv2.imshow('Disaster Rescue Detection System - Image Processing', annotated_frame)
                
                # Handle user input with timeout
                key = cv2.waitKey(self.image_display_time) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('n') or key == ord(' ') or key == 255:  # Next (n, space, or timeout)
                    current_index += 1
                elif key == ord('p'):  # Previous
                    current_index = max(0, current_index - 1)
                elif key == ord('s'):
                    self.save_detection_frame(annotated_frame, current_detections, image_path)
                elif key == ord('r'):
                    self.generate_report()
                elif key == ord('u'):
                    self.show_unique_objects_summary()
        
        except KeyboardInterrupt:
            print("\nImage processing stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            self.generate_final_report(len(image_files), new_high_priority_count)
    
    def draw_image_status_info(self, frame, current_image, total_images, new_high_priority_count, filename):
        """Draw status info for image processing mode"""
        unique_persons = len([obj for obj in self.unique_objects.values() 
                            if 'person' in obj['class_name'] or 'human' in obj['class_name']])
        unique_animals = len([obj for obj in self.unique_objects.values() 
                            if obj['class_name'] not in ['person', 'human']])
        
        status_info = [
            f'Image: {current_image}/{total_images}',
            f'File: {filename[:30]}...' if len(filename) > 30 else f'File: {filename}',
            f'Unique Persons: {unique_persons}',
            f'Unique Animals: {unique_animals}',
            f'New High Priority: {new_high_priority_count}',
            f'Total Unique Objects: {len(self.unique_objects)}',
            f'Model: {"Custom" if self.is_custom_model else "Standard"}',
            f'Mode: IMAGE PROCESSING',
            f'Time: {datetime.now().strftime("%H:%M:%S")}'
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, info in enumerate(status_info):
            y_pos = 35 + (i * 25)
            if 'Unique Persons:' in info and unique_persons > 0:
                color = (0, 0, 255)  # Red for persons
            elif 'Unique Animals:' in info and unique_animals > 0:
                color = (0, 255, 255)  # Yellow for animals
            elif 'IMAGE PROCESSING' in info:
                color = (0, 255, 0)  # Green for mode
            else:
                color = (255, 255, 255)  # White for others
                
            cv2.putText(frame, info, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def detect_and_track(self, source=0, duration_minutes=None):
        """Main detection loop with unique tracking for camera/video"""
        
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open source: {source}")
            return
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60) if duration_minutes else float('inf')
        
        frame_count = 0
        new_high_priority_count = 0
        
        print("=== DISASTER RESCUE DETECTION WITH UNIQUE TRACKING ===")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Generate report")
        print("  'c' - Clear detection history")
        print("  'u' - Show unique objects summary")
        
        try:
            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Process detections with unique tracking
                current_detections = self.process_detections_with_tracking(results[0], frame_count)
                
                # Count new high priority detections
                new_high_priority = len([d for d in current_detections if d['priority'] == 1 and d['is_new']])
                new_high_priority_count += new_high_priority
                
                # Draw enhanced annotations
                annotated_frame = self.draw_enhanced_annotations_with_tracking(frame, current_detections)
                
                # Display status
                self.draw_status_info_with_tracking(annotated_frame, frame_count, new_high_priority_count)
                
                cv2.imshow('Disaster Rescue Detection System - Real-time Camera', annotated_frame)
                
                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_detection_frame(annotated_frame, current_detections)
                elif key == ord('r'):
                    self.generate_report()
                elif key == ord('c'):
                    self.clear_history()
                elif key == ord('u'):
                    self.show_unique_objects_summary()
                
                # Auto-save new high priority detections only
                if new_high_priority > 0:
                    self.save_high_priority_detection(annotated_frame, current_detections)
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.generate_final_report(frame_count, new_high_priority_count)
    
        

    def process_detections_with_tracking(self, result, frame_number):
        """
        MODIFIED: Process detections with GPS location logging
        """
        current_detections = []
    
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Frame {frame_number}: Found {len(result.boxes)} raw detections")
            
            for i, box in enumerate(result.boxes):
                try:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    if class_id in self.priority_classes and confidence >= self.confidence_threshold:
                        bbox = box.xyxy.cpu().numpy().tolist()[0]
                        
                        detection = {
                            'timestamp': datetime.now(),
                            'frame': frame_number,
                            'class_id': class_id,
                            'class_name': self.priority_classes[class_id]['name'],
                            'confidence': confidence,
                            'priority': self.priority_classes[class_id]['priority'],
                            'bbox': bbox,
                            'center': self.calculate_center(bbox)
                        }
                        
                        # Apply unique tracking
                        self.update_or_create_object(detection, frame_number)
                        current_detections.append(detection)
                        
                        # NEW: Log GPS location for new unique detections
                        if detection['is_new'] and self.gps_tracker:
                            location_record = self.gps_tracker.log_detection_location(
                                detection, frame_number
                            )
                            
                            # Print location info for high priority detections
                            if detection['priority'] == 1 or detection['priority'] == 1  and location_record:
                                print(f"🚨 RESCUE LOCATION: {detection['class_name'].upper()} "
                                      f"at GPS {location_record['gps_location']['latitude']:.6f}, "
                                      f"{location_record['gps_location']['longitude']:.6f}")
                        
                        # Add to history if it's a new unique detection
                        if detection['is_new']:
                            self.detection_history.append(detection)
                            
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue
        
        return current_detections
    
    def draw_enhanced_annotations_with_tracking(self, frame, detections):
        """
        MODIFIED: Enhanced annotations with GPS status
        """
        annotated_frame = frame.copy()
        
        # Add GPS status indicator
        if self.gps_tracker:
            gps_location = self.gps_tracker.get_current_location()
            if gps_location:
                gps_status = f"GPS: {gps_location['latitude']:.6f}, {gps_location['longitude']:.6f}"
                gps_color = (0, 255, 0) if gps_location['accuracy'] < 5 else (0, 255, 255)
            else:
                gps_status = "GPS: NO SIGNAL"
                gps_color = (0, 0, 255)
                
            cv2.putText(annotated_frame, gps_status, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 1)
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            color = self.priority_classes[detection['class_id']]['color']
            
            if detection['is_new']:
                thickness = 4
                line_type = cv2.LINE_8
            else:
                thickness = 2
                line_type = cv2.LINE_4
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness, line_type)
            
            priority_text = "HIGH" if detection['priority'] == 1 else "MED"
            status_text = "NEW!" if detection['is_new'] else "TRACKED"
            location_text = "📍" if detection['is_new'] and self.gps_tracker else ""
            
            label = f"ID:{detection['object_id']} {detection['class_name']} ({priority_text}) {detection['confidence']:.2f} [{status_text}] {location_text}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            bg_color = (0, 255, 0) if detection['is_new'] else color
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), bg_color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            center = tuple(map(int, detection['center']))
            cv2.circle(annotated_frame, center, 5, color, -1)
            cv2.putText(annotated_frame, str(detection['object_id']), 
                       (center[0] - 10, center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def draw_status_info_with_tracking(self, frame, frame_count, new_high_priority_count):
        """
        MODIFIED: Status info with GPS and rescue waypoints
        """
        unique_persons = len([obj for obj in self.unique_objects.values() 
                            if 'person' in obj['class_name'] or 'human' in obj['class_name']])
        unique_animals = len([obj for obj in self.unique_objects.values() 
                            if obj['class_name'] not in ['person', 'human']])
        
        # Get GPS and waypoint info
        gps_status = "NO GPS"
        waypoints_count = 0
        if self.gps_tracker:
            current_gps = self.gps_tracker.get_current_location()
            if current_gps:
                gps_status = f"GPS OK (±{current_gps['accuracy']:.1f}m)"
            waypoints_count = len(self.gps_tracker.get_rescue_waypoints('PENDING'))
        
        status_info = [
            f'Frame: {frame_count}',
            f'Unique Persons: {unique_persons}',
            f'Unique Animals: {unique_animals}',
            f'New High Priority: {new_high_priority_count}',
            f'Rescue Waypoints: {waypoints_count}',
            f'GPS Status: {gps_status}',
            f'Total Unique Objects: {len(self.unique_objects)}',
            f'Mode: REAL-TIME + GPS',
            f'Time: {datetime.now().strftime("%H:%M:%S")}'
        ]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, info in enumerate(status_info):
            y_pos = 35 + (i * 25)
            if 'Unique Persons:' in info and unique_persons > 0:
                color = (0, 0, 255)  # Red for persons
            elif 'Unique Animals:' in info and unique_animals > 0:
                color = (0, 255, 255)  # Yellow for animals
            elif 'Rescue Waypoints:' in info and waypoints_count > 0:
                color = (0, 255, 0)  # Green for waypoints
            elif 'GPS Status:' in info:
                color = (0, 255, 0) if "GPS OK" in info else (0, 0, 255)
            else:
                color = (255, 255, 255)
                
            cv2.putText(frame, info, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def save_detection_frame(self, frame, detections, source_path=None):
        """Save current frame with detection data including unique IDs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        if source_path:
            img_filename = f"detection_{os.path.splitext(os.path.basename(source_path))[0]}_{timestamp}.jpg"
        else:
            img_filename = f"detection_{timestamp}.jpg"
        
        img_path = self.captures_dir / img_filename
        cv2.imwrite(str(img_path), frame)
        
        # Save detection data with unique tracking info
        detection_data = {
            'timestamp': timestamp,
            'source_file': source_path if source_path else 'camera',
            'current_detections': [{
                'object_id': d['object_id'],
                'class_name': d['class_name'],
                'confidence': d['confidence'],
                'priority': d['priority'],
                'bbox': d['bbox'],
                'is_new': d['is_new']
            } for d in detections],
            'unique_objects_summary': {
                'total_unique_objects': len(self.unique_objects),
                'unique_persons': len([obj for obj in self.unique_objects.values() 
                                     if 'person' in obj['class_name'] or 'human' in obj['class_name']]),
                'unique_animals': len([obj for obj in self.unique_objects.values() 
                                     if obj['class_name'] not in ['person', 'human']])
            }
        }
        
        json_filename = f"detection_{os.path.splitext(os.path.basename(source_path))[0]}_{timestamp}.json" if source_path else f"detection_{timestamp}.json"
        with open(self.captures_dir / json_filename, 'w') as f:
            json.dump(detection_data, f, indent=2, default=str)
        
        print(f"Saved detection: {img_path}")
    
    def save_high_priority_detection(self, frame, detections, source_path=None):
        """Automatically save NEW high priority detections only"""
        new_high_priority = [d for d in detections if d['priority'] == 1 and d['is_new']]
        if not new_high_priority:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if source_path:
            alert_filename = f"ALERT_NEW_PERSON_{os.path.splitext(os.path.basename(source_path))[0]}_{timestamp}.jpg"
        else:
            alert_filename = f"ALERT_NEW_PERSON_{timestamp}.jpg"
        
        alert_path = self.captures_dir / alert_filename
        cv2.imwrite(str(alert_path), frame)
        
        for detection in new_high_priority:
            print(f"🚨 NEW HIGH PRIORITY ALERT - {detection['class_name'].upper()} ID:{detection['object_id']} DETECTED! 🚨")
        
        print(f"Alert saved: {alert_path}")
    
    def show_unique_objects_summary(self):
        """Display summary of all unique objects detected"""
        if not self.unique_objects:
            print("No unique objects detected yet")
            return
        
        print("\n=== UNIQUE OBJECTS SUMMARY ===")
        for obj_id, obj_data in self.unique_objects.items():
            duration = obj_data['last_frame'] - obj_data['first_frame']
            print(f"ID {obj_id}: {obj_data['class_name']} | "
                  f"Priority: {obj_data['priority']} | "
                  f"Frames: {obj_data['first_frame']}-{obj_data['last_frame']} ({duration} frames) | "
                  f"Max Conf: {obj_data['max_confidence']:.3f}")
        print("="*50)
    
    def generate_report(self):
        """Generate current session report with unique tracking"""
        if not self.unique_objects:
            print("No unique objects detected to report")
            return
        
        # Count unique objects by class and priority
        class_counts = {}
        priority_counts = {1: 0, 2: 0, 3: 0}
        
        for obj_data in self.unique_objects.values():
            class_name = obj_data['class_name']
            priority = obj_data['priority']
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            priority_counts[priority] += 1
        
        print("\n=== UNIQUE DETECTION REPORT ===")
        print(f"Total Unique Objects: {len(self.unique_objects)}")
        print(f"High Priority Objects: {priority_counts[1]}")
        print(f"Medium Priority Objects: {priority_counts[2]}")
        
        print("\nUnique Objects by Class:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        print("\nDetailed Object List:")
        for obj_id, obj_data in self.unique_objects.items():
            duration = obj_data['last_frame'] - obj_data['first_frame']
            print(f"  ID {obj_id}: {obj_data['class_name']} | "
                  f"Tracked for {duration} frames | "
                  f"Max confidence: {obj_data['max_confidence']:.3f}")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'custom' if self.is_custom_model else 'standard',
            'total_unique_objects': len(self.unique_objects),
            'priority_counts': priority_counts,
            'class_counts': class_counts,
            'unique_objects': [{
                'id': obj['id'],
                'class_name': obj['class_name'],
                'priority': obj['priority'],
                'first_seen': obj['first_seen'].isoformat(),
                'last_seen': obj['last_seen'].isoformat(),
                'tracking_duration_frames': obj['last_frame'] - obj['first_frame'],
                'max_confidence': obj['max_confidence'],
                'total_frame_detections': obj['total_detections']
            } for obj in self.unique_objects.values()]
        }
        
        report_path = self.logs_dir / f"unique_tracking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Report saved to: {report_path}")
    

    def generate_rescue_report(self):
        """
        NEW: Generate comprehensive rescue report with GPS locations
        """
        print("\n=== RESCUE OPERATIONS REPORT ===")
        
        # Standard detection report
        self.generate_report()
        
        # GPS and location report
        if self.gps_tracker:
            print("\n=== GPS LOCATION REPORT ===")
            
            waypoints = self.gps_tracker.get_rescue_waypoints()
            if waypoints:
                print(f"Total Rescue Waypoints: {len(waypoints)}")
                print(f"Pending Rescues: {len([w for w in waypoints if w['status'] == 'PENDING'])}")
                
                print("\nHigh Priority Locations:")
                for waypoint in waypoints:
                    if waypoint['status'] == 'PENDING':
                        print(f"  Waypoint #{waypoint['id']}: {waypoint['type']} "
                              f"at {waypoint['latitude']:.6f}, {waypoint['longitude']:.6f}")
                        print(f"    Google Maps: https://maps.google.com/?q="
                              f"{waypoint['latitude']},{waypoint['longitude']}")
                
                # Generate KML for Google Earth
                kml_file = self.gps_tracker.generate_kml_file()
                print(f"\nGoogle Earth KML file: {kml_file}")
                
                # Generate location summary
                self.gps_tracker.generate_location_summary()
            else:
                print("No rescue waypoints recorded")
        else:
            print("GPS tracking was disabled")
    
    def clear_history(self):
        """Clear detection history and unique objects"""
        self.detection_history.clear()
        self.unique_objects.clear()
        self.next_object_id = 1
        print("Detection history and unique objects cleared")
    
    def generate_final_report(self, total_frames, new_high_priority_count):
        """Generate final session report with unique tracking stats"""
        print("\n" + "="*60)
        print("FINAL SESSION REPORT - UNIQUE TRACKING")
        print("="*60)
        print(f"Model Type: {'Custom Disaster Model' if self.is_custom_model else 'Standard YOLO Model'}")
        print(f"Total frames/images processed: {total_frames}")
        print(f"Total unique objects detected: {len(self.unique_objects)}")
        print(f"New high priority alerts: {new_high_priority_count}")
        
        # Breakdown by class
        persons = [obj for obj in self.unique_objects.values() 
                  if 'person' in obj['class_name'] or 'human' in obj['class_name']]
        animals = [obj for obj in self.unique_objects.values() 
                  if obj['class_name'] not in ['person', 'human']]
        
        print(f"Unique persons detected: {len(persons)}")
        print(f"Unique animals detected: {len(animals)}")
        
        if self.unique_objects:
            # Calculate tracking statistics
            total_detections = sum(obj['total_detections'] for obj in self.unique_objects.values())
            avg_tracking_duration = np.mean([obj['last_frame'] - obj['first_frame'] for obj in self.unique_objects.values()])
            avg_confidence = np.mean([obj['max_confidence'] for obj in self.unique_objects.values()])
            
            print(f"Total frame detections (all objects): {total_detections}")
            print(f"Average tracking duration: {avg_tracking_duration:.1f} frames")
            print(f"Average max confidence: {avg_confidence:.3f}")
            
            # Show longest tracked objects
            if len(self.unique_objects) > 0:
                longest_tracked = max(self.unique_objects.values(), 
                                    key=lambda x: x['last_frame'] - x['first_frame'])
                print(f"Longest tracked object: ID {longest_tracked['id']} ({longest_tracked['class_name']}) "
                      f"for {longest_tracked['last_frame'] - longest_tracked['first_frame']} frames")
        
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        # Save final summary
        self.save_final_summary(total_frames, new_high_priority_count)
    
    def save_final_summary(self, total_frames, new_high_priority_count):
        """Save final summary to JSON file"""
        summary = {
            'session_end': datetime.now().isoformat(),
            'model_type': 'custom' if self.is_custom_model else 'standard',
            'total_frames': total_frames,
            'unique_objects_detected': len(self.unique_objects),
            'new_high_priority_alerts': new_high_priority_count,
            'unique_persons': len([obj for obj in self.unique_objects.values() 
                                 if 'person' in obj['class_name'] or 'human' in obj['class_name']]),
            'unique_animals': len([obj for obj in self.unique_objects.values() 
                                 if obj['class_name'] not in ['person', 'human']]),
            'tracking_parameters': {
                'tracking_threshold': self.tracking_threshold,
                'max_missed_frames': self.max_missed_frames,
                'confidence_threshold': self.confidence_threshold
            },
            'priority_classes_used': list(self.priority_classes.keys())
        }
        
        summary_path = self.logs_dir / f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Session summary saved to: {summary_path}")

    def cleanup_and_shutdown(self):
        """
        NEW: Proper cleanup including GPS tracking
        """
        print("\nShutting down detection system...")
        
        if self.gps_tracker:
            self.gps_tracker.stop_gps_tracking()
        
        # Generate final reports
        self.generate_rescue_report()
        
        print("System shutdown complete")

def main():
    parser = argparse.ArgumentParser(description='Disaster Rescue Detection System with Unique Tracking')
    parser.add_argument('--model', default=r'C:\Users\vedan\Documents\My_Projects\Nidar_drone_project\disaster_drone_models\models\disaster_rescue_v1\weights\best.pt', 
                       help='Model path (will auto-detect available models)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--duration', type=int, help='Duration in minutes (camera mode only)')
    parser.add_argument('--tracking-threshold', type=int, default=100, help='Max distance for same object tracking (pixels)')
    parser.add_argument('--max-missed-frames', type=int, default=30, help='Max frames to keep tracking inactive objects')
    parser.add_argument('--image-display-time', type=int, default=3000, help='Time to display each image in ms (image mode)')
    parser.add_argument('--mode', choices=['camera', 'images', 'test'], help='Force input mode (skip interactive selection)')
    parser.add_argument('--source', help='Camera source (for camera mode) or image folder/files (for image mode)')
    parser.add_argument('--enable-gps', action='store_true', default=True, help='Enable GPS tracking')
    parser.add_argument('--disable-gps', action='store_true', help='Disable GPS tracking')
    
    args = parser.parse_args()
    
    # Determine GPS setting
    enable_gps = args.enable_gps and not args.disable_gps

    # Initialize system with improved error handling
    try:
        detector = CompleteDroneDetectionSystem(
            model_path=args.model,
            confidence_threshold=args.conf,
            enable_gps=enable_gps
        )
        print(f"GPS Tracking: {'ENABLED' if enable_gps else 'DISABLED'}")

    except Exception as e:
        print(f"Failed to initialize detection system: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have trained your model first")
        print("2. Or download a basic YOLO model: from ultralytics import YOLO; YOLO('yolov8n.pt')")
        print("3. Check the model path is correct")
        return
    
    # Update parameters if provided
    detector.tracking_threshold = args.tracking_threshold
    detector.max_missed_frames = args.max_missed_frames
    detector.image_display_time = args.image_display_time
    
    print(f"Detection parameters: Confidence = {args.conf}, "
          f"Tracking threshold = {detector.tracking_threshold}px, "
          f"Max missed frames = {detector.max_missed_frames}")
    
    # Determine input mode
    if args.mode:
        input_mode = args.mode
        print(f"Mode forced via argument: {input_mode}")
    else:
        input_mode = detector.choose_input_mode()
    
    if input_mode is None:
        print("Exiting...")
        return
    
    try:
        if input_mode == 'test':
            # Test model mode
            print("Starting model test mode...")
            detector.test_model()
            
        elif input_mode == 'camera':
            # Camera/video mode
            if args.source:
                source = int(args.source) if args.source.isdigit() else args.source
            else:
                source = detector.get_camera_source()
            
            print(f"Starting camera detection with source: {source}")
            detector.detect_and_track(source=source, duration_minutes=args.duration)
            
        elif input_mode == 'images':
            # Image processing mode
            if args.source:
                # If source is provided, treat it as a folder or file pattern
                if os.path.isdir(args.source):
                    # It's a directory
                    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
                    image_files = []
                    for ext in extensions:
                        image_files.extend(glob.glob(os.path.join(args.source, ext)))
                        image_files.extend(glob.glob(os.path.join(args.source, ext.upper())))
                    image_files = sorted(image_files)
                elif os.path.isfile(args.source):
                    # It's a single file
                    image_files = [args.source]
                else:
                    # Try as a glob pattern
                    image_files = sorted(glob.glob(args.source))
                
                if not image_files:
                    print(f"No valid images found for source: {args.source}")
                    return
                    
                print(f"Found {len(image_files)} images from source: {args.source}")
            else:
                # Interactive mode
                image_files = detector.get_image_sources()
            
            if image_files:
                print(f"Starting image processing with {len(image_files)} images")
                detector.process_images(image_files)
            else:
                print("No images selected for processing")
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure proper cleanup
        if 'detector' in locals():
            detector.cleanup_and_shutdown()   

if __name__ == "__main__":
    main()