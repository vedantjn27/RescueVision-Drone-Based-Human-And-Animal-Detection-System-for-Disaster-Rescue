# Setup Guide for RescueVision (Nidar)

## Prerequisites
- Python 3.8 or higher installed on your system.
- A functional webcam or external video feed source for real-time detection.
- (Optional) Pre-clicked aerial images for testing the image processing mode.

## Installation

1. **Navigate to the project directory:**
   Ensure you are in the root directory of the project.
   ```bash
   cd "path/to/Nidar- Drone Usage for Human and Animals Detection In Disaster Affected Areas"
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Model Preparation:**
   Ensure that your trained YOLO model weights are available. The system looks for custom models in `disaster_drone_models/models/disaster_rescue_v1/weights/best.pt` or falls back to standard models like `yolov8n.pt` in the root directory.

## Running the Application

1. **Start the Main System:**
   Execute the core script to launch the application:
   ```bash
   python complete_system.py
   ```

2. **Select Input Mode:**
   The interactive CLI will prompt you to choose an operational mode:
   - `1`: **Real-time Camera Detection** (Uses webcam or drone video feed)
   - `2`: **Process Pre-clicked Images** (Batch processes a folder of images)
   - `3`: **Test Model with Sample Detection** (Verifies the model is functioning)

3. **In-App Controls:**
   While the detection window is open, you can use the following keyboard controls:
   - `q` : Quit the application
   - `s` : Save the current detection frame
   - `r` : Generate a detailed report
   - `u` : Show a summary of unique tracked objects
   - `c` : Clear the detection history (Camera mode only)
   - `Space/n` : Next image (Image mode only)
   - `p` : Previous image (Image mode only)

## Outputs
All captured frames, detection logs, and GPS waypoints will be automatically saved in the `output/` directory for further review.
