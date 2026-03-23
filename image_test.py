from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def test_image_detection():
    """Test detection on sample images"""
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Test images directory
    test_dir = Path(__file__).parent / "test_images"
    results_dir = Path('results/image_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_dir.exists():
        print(f"Please create '{test_dir}' directory and add test images")
        return
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(test_dir.glob(f'*{ext}'))
        image_files.extend(test_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        print("Please add some test images (jpg, png, etc.)")
        return
    
    print(f"Found {len(image_files)} images to test")
    
    detection_results = []
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Run detection
        results = model(img)
        
        # Count detections
        person_count = 0
        total_detections = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                total_detections = len(boxes)
                for box in boxes:
                    if int(box.cls) == 0:  # person class
                        person_count += 1
        
        detection_results.append({
            'image': img_path.name,
            'persons': person_count,
            'total_objects': total_detections
        })
        
        # Save annotated image
        annotated = results[0].plot()
        output_path = results_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)
    
    # Print summary
    print("\n=== DETECTION SUMMARY ===")
    for result in detection_results:
        print(f"{result['image']}: {result['persons']} persons, {result['total_objects']} total objects")
    
    total_persons = sum(r['persons'] for r in detection_results)
    print(f"\nTotal persons detected across all images: {total_persons}")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    test_image_detection()