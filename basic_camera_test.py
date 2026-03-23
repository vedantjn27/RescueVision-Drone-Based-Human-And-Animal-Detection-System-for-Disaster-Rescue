from ultralytics import YOLO
import cv2
import time
import os

def test_basic_detection():
    """Test basic human detection with laptop camera"""
    
    # Load pre-trained YOLO model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # This will auto-download on first run
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera initialized. Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Run detection every 3rd frame for better performance
        if frame_count % 3 == 0:
            results = model(frame, verbose=False)
            
            # Count person detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0:  # person class
                            detection_count += 1
            
            # Draw results
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
        
        # Add info text
        cv2.putText(annotated_frame, f'Frames: {frame_count} | Detections: {detection_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Disaster Detection - Basic Test', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"results/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTest completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total person detections: {detection_count}")

if __name__ == "__main__":
    test_basic_detection()