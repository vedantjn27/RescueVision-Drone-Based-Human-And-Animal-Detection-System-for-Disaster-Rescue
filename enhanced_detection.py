from ultralytics import YOLO
import cv2
import time
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np

class DisasterDetectionSystem:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the disaster detection system"""
        self.model = YOLO(model_path)
        self.detection_log = []
        self.performance_log = []
        
        # Create logs directory
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
        # Detection classes we care about for disaster rescue
        self.rescue_classes = {
            0: 'person',           # Highest priority
            15: 'cat',             # Animals
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe'
        }
    
    def detect_from_camera(self, duration_minutes=5):
        """Run detection from camera for specified duration"""
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        frame_count = 0
        fps_list = []
        
        print(f"Starting {duration_minutes}-minute detection session...")
        print("Press 'q' to quit early, 's' to save current frame")
        
        while time.time() < end_time:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Process results
            detections = self.process_results(results[0], frame_count)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_list.append(fps)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add performance info
            self.draw_info(annotated_frame, fps, len(detections), frame_count)
            
            cv2.imshow('Disaster Detection System', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_frame(annotated_frame, detections)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save session results
        self.save_session_results(frame_count, fps_list)
        
    def process_results(self, result, frame_number):
        """Process YOLO results and log important detections"""
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                
                # Only log rescue-relevant classes
                if class_id in self.rescue_classes:
                    detection = {
                        'timestamp': datetime.now().isoformat(),
                        'frame': frame_number,
                        'class_id': class_id,
                        'class_name': self.rescue_classes[class_id],
                        'confidence': confidence,
                        'bbox': box.xyxy.cpu().numpy().tolist()[0],
                        'priority': 'HIGH' if class_id == 0 else 'MEDIUM'  # Humans = high priority
                    }
                    
                    detections.append(detection)
                    self.detection_log.append(detection)
        
        return detections
    
    def draw_info(self, frame, fps, detection_count, frame_number):
        """Draw performance and detection info on frame"""
        info_text = [
            f'FPS: {fps:.1f}',
            f'Frame: {frame_number}',
            f'Detections: {detection_count}',
            f'Total Logged: {len(self.detection_log)}'
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 30 + (i * 30)
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def save_frame(self, frame, detections):
        """Save current frame with detection info"""
        timestamp = int(time.time())
        frame_path = f"results/frame_{timestamp}.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Save detection info
        if detections:
            json_path = f"results/detections_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(detections, f, indent=2)
        
        print(f"Saved: {frame_path}")
    
    def save_session_results(self, total_frames, fps_list):
        """Save complete session results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detection log
        csv_path = self.logs_dir / f"detection_log_{timestamp}.csv"
        if self.detection_log:
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ['timestamp', 'frame', 'class_name', 'confidence', 'priority']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for detection in self.detection_log:
                    writer.writerow({
                        'timestamp': detection['timestamp'],
                        'frame': detection['frame'],
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence'],
                        'priority': detection['priority']
                    })
        
        # Save performance summary
        summary = {
            'session_timestamp': timestamp,
            'total_frames': total_frames,
            'total_detections': len(self.detection_log),
            'avg_fps': np.mean(fps_list) if fps_list else 0,
            'person_detections': len([d for d in self.detection_log if d['class_name'] == 'person']),
            'animal_detections': len([d for d in self.detection_log if d['class_name'] != 'person'])
        }
        
        summary_path = self.logs_dir / f"session_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== SESSION COMPLETE ===")
        print(f"Frames processed: {total_frames}")
        print(f"Total detections: {len(self.detection_log)}")
        print(f"Person detections: {summary['person_detections']}")
        print(f"Animal detections: {summary['animal_detections']}")
        print(f"Average FPS: {summary['avg_fps']:.2f}")
        print(f"Results saved to: {csv_path}")

def main():
    detector = DisasterDetectionSystem()
    
    print("Disaster Detection System")
    print("1. Quick test (1 minute)")
    print("2. Standard test (5 minutes)")
    print("3. Custom duration")
    
    choice = input("Choose option (1-3): ")
    
    if choice == '1':
        detector.detect_from_camera(1)
    elif choice == '2':
        detector.detect_from_camera(5)
    elif choice == '3':
        minutes = int(input("Enter duration in minutes: "))
        detector.detect_from_camera(minutes)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()