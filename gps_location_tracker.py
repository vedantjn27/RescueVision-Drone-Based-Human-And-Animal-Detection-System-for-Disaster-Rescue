import json
import time
from datetime import datetime
from pathlib import Path
import math
import threading
import queue
import logging

# For GPS simulation (replace with actual GPS hardware interface)
import random

class GPSLocationTracker:
    def __init__(self, output_dir="output"):
        """
        GPS Location Tracker for Drone Detection System
        
        This class provides real-time GPS tracking and location logging
        for detected humans and animals during rescue operations.
        """
        self.output_dir = Path(output_dir)
        self.locations_dir = self.output_dir / "locations"
        self.locations_dir.mkdir(parents=True, exist_ok=True)
        
        # GPS tracking variables
        self.current_gps = None
        self.gps_queue = queue.Queue()
        self.gps_thread = None
        self.gps_running = False
        
        # Detection locations storage
        self.detection_locations = {}
        self.rescue_waypoints = []
        
        # GPS simulation parameters (replace with real GPS interface)
        self.simulate_gps = True
        self.base_lat = 12.9716  # Bangalore latitude for simulation
        self.base_lon = 77.5946  # Bangalore longitude for simulation
        self.gps_accuracy = 3.0   # meters
        
        # Setup logging
        self.setup_logging()
        
        # Start GPS tracking
        self.start_gps_tracking()
        
        print("GPS Location Tracker initialized")
        print(f"Location logs will be saved to: {self.locations_dir}")
    
    def setup_logging(self):
        """Setup location logging"""
        log_file = self.locations_dir / f"gps_log_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def start_gps_tracking(self):
        """Start GPS tracking thread"""
        self.gps_running = True
        self.gps_thread = threading.Thread(target=self._gps_worker, daemon=True)
        self.gps_thread.start()
        print("GPS tracking started")
    
    def stop_gps_tracking(self):
        """Stop GPS tracking"""
        self.gps_running = False
        if self.gps_thread:
            self.gps_thread.join(timeout=1)
        print("GPS tracking stopped")
    
    def _gps_worker(self):
        """GPS worker thread - simulates or reads real GPS data"""
        while self.gps_running:
            try:
                if self.simulate_gps:
                    # Simulate drone movement (replace with actual GPS reading)
                    gps_data = self._simulate_gps_data()
                else:
                    # TODO: Replace with actual GPS hardware interface
                    # gps_data = self._read_real_gps()
                    gps_data = self._simulate_gps_data()
                
                self.current_gps = gps_data
                
                # Log GPS data every 5 seconds
                if int(time.time()) % 5 == 0:
                    self.logger.info(f"Drone GPS: {gps_data}")
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"GPS tracking error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _simulate_gps_data(self):
        """
        Simulate GPS data for testing
        Replace this with actual GPS hardware interface
        """
        # Simulate drone movement in a search pattern
        current_time = time.time()
        
        # Create a realistic search pattern movement
        offset_lat = math.sin(current_time / 10.0) * 0.001  # ~100m movement
        offset_lon = math.cos(current_time / 10.0) * 0.001
        
        # Add some random GPS noise
        noise_lat = random.uniform(-0.00005, 0.00005)  # ~5m noise
        noise_lon = random.uniform(-0.00005, 0.00005)
        
        gps_data = {
            'timestamp': datetime.now(),
            'latitude': self.base_lat + offset_lat + noise_lat,
            'longitude': self.base_lon + offset_lon + noise_lon,
            'altitude': 50.0 + random.uniform(-2, 2),  # 50m flight height with noise
            'accuracy': self.gps_accuracy + random.uniform(-1, 1),
            'heading': (current_time * 10) % 360,  # Slowly rotating
            'speed': 5.0 + random.uniform(-1, 1),  # 5 m/s with variation
            'satellites': random.randint(8, 12)
        }
        
        return gps_data
    
    def _read_real_gps(self):
        """
        Read GPS data from actual hardware
        TODO: Implement based on your GPS hardware
        
        Example interfaces:
        - Serial GPS (NMEA): Use pyserial + pynmea2
        - USB GPS: Use gpsd
        - Drone telemetry: Use pymavlink or specific drone SDK
        """
        # Example implementation for NMEA GPS:
        """
        import serial
        import pynmea2
        
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        line = ser.readline().decode('ascii', errors='replace')
        
        if line.startswith('$GPGGA'):
            msg = pynmea2.parse(line)
            return {
                'timestamp': datetime.now(),
                'latitude': float(msg.latitude),
                'longitude': float(msg.longitude),
                'altitude': float(msg.altitude),
                'accuracy': float(msg.horizontal_dil) if msg.horizontal_dil else 5.0,
                'satellites': int(msg.num_sats) if msg.num_sats else 0
            }
        """
        pass
    
    def log_detection_location(self, detection_data, frame_number=None):
        """
        Log the GPS location when a human or animal is detected
        
        Args:
            detection_data: Detection information from YOLO
            frame_number: Current frame number (optional)
        """
        if not self.current_gps:
            print("Warning: No GPS data available for location logging")
            return None
        
        # Create location record
        location_record = {
            'detection_id': detection_data.get('object_id'),
            'detection_type': detection_data.get('class_name'),
            'confidence': detection_data.get('confidence'),
            'priority': detection_data.get('priority'),
            'is_new_detection': detection_data.get('is_new', False),
            'frame_number': frame_number,
            'timestamp': datetime.now().isoformat(),
            'gps_location': {
                'latitude': self.current_gps['latitude'],
                'longitude': self.current_gps['longitude'],
                'altitude': self.current_gps['altitude'],
                'accuracy': self.current_gps['accuracy'],
                'heading': self.current_gps.get('heading'),
                'speed': self.current_gps.get('speed')
            },
            'image_coordinates': {
                'bbox': detection_data.get('bbox'),
                'center': detection_data.get('center')
            }
        }
        
        # Store in memory
        detection_id = detection_data.get('object_id')
        if detection_id not in self.detection_locations:
            self.detection_locations[detection_id] = []
        
        self.detection_locations[detection_id].append(location_record)
        
        # Save to file immediately for new high-priority detections
        if detection_data.get('priority') == 1 or detection_data.get('priority') == 2  and detection_data.get('is_new'):
            self._save_priority_location(location_record)
            self._add_to_rescue_waypoints(location_record)
        
        # Log the detection
        self.logger.info(f"DETECTION LOCATION - ID:{detection_id} "
                        f"{detection_data.get('class_name')} at "
                        f"{self.current_gps['latitude']:.6f}, "
                        f"{self.current_gps['longitude']:.6f}")
        
        return location_record

    def _save_priority_location(self, location_record):
        """Save high priority detection location immediately"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"PRIORITY_LOCATION_{location_record['detection_type']}_{location_record['detection_id']}_{timestamp}.json"
        
        filepath = self.locations_dir / filename
        with open(filepath, 'w') as f:
            json.dump(location_record, f, indent=2, default=str)
        
        # Also create a simple text alert
        alert_filename = f"RESCUE_ALERT_{timestamp}.txt"
        alert_path = self.locations_dir / alert_filename
        
        with open(alert_path, 'w') as f:
            f.write(f"RESCUE ALERT - {location_record['detection_type'].upper()} DETECTED\n")
            f.write(f"Time: {location_record['timestamp']}\n")
            f.write(f"Detection ID: {location_record['detection_id']}\n")
            f.write(f"Location: {location_record['gps_location']['latitude']:.6f}, "
                   f"{location_record['gps_location']['longitude']:.6f}\n")
            f.write(f"Altitude: {location_record['gps_location']['altitude']:.1f}m\n")
            f.write(f"Accuracy: ±{location_record['gps_location']['accuracy']:.1f}m\n")
            f.write(f"Confidence: {location_record['confidence']:.3f}\n")
            f.write(f"\nGoogle Maps: https://maps.google.com/?q="
                   f"{location_record['gps_location']['latitude']},{location_record['gps_location']['longitude']}\n")
        
        print(f"🚨 PRIORITY LOCATION SAVED: {filepath}")
        print(f"📍 GPS: {location_record['gps_location']['latitude']:.6f}, "
              f"{location_record['gps_location']['longitude']:.6f}")
    
    def _add_to_rescue_waypoints(self, location_record):
        """Add location to rescue waypoints list"""
        waypoint = {
            'id': len(self.rescue_waypoints) + 1,
            'detection_id': location_record['detection_id'],
            'type': location_record['detection_type'],
            'latitude': location_record['gps_location']['latitude'],
            'longitude': location_record['gps_location']['longitude'],
            'altitude': location_record['gps_location']['altitude'],
            'priority': location_record['priority'],
            'timestamp': location_record['timestamp'],
            'status': 'PENDING'  # PENDING, IN_PROGRESS, RESCUED, VERIFIED
        }
        
        self.rescue_waypoints.append(waypoint)
        self._save_waypoints()
        
        print(f"📍 Added to rescue waypoints: Waypoint #{waypoint['id']}")
    
    def _save_waypoints(self):
        """Save current rescue waypoints to file"""
        waypoints_file = self.locations_dir / "rescue_waypoints.json"
        
        waypoints_data = {
            'generated_at': datetime.now().isoformat(),
            'total_waypoints': len(self.rescue_waypoints),
            'pending_rescues': len([w for w in self.rescue_waypoints if w['status'] == 'PENDING']),
            'waypoints': self.rescue_waypoints
        }
        
        with open(waypoints_file, 'w') as f:
            json.dump(waypoints_data, f, indent=2, default=str)
    
    def get_current_location(self):
        """Get current GPS location"""
        return self.current_gps
    
    def get_detection_locations(self, detection_id=None):
        """Get stored detection locations"""
        if detection_id:
            return self.detection_locations.get(detection_id, [])
        return self.detection_locations
    
    def get_rescue_waypoints(self, status_filter=None):
        """Get rescue waypoints, optionally filtered by status"""
        if status_filter:
            return [w for w in self.rescue_waypoints if w['status'] == status_filter]
        return self.rescue_waypoints
    
    def update_waypoint_status(self, waypoint_id, new_status):
        """Update the status of a rescue waypoint"""
        for waypoint in self.rescue_waypoints:
            if waypoint['id'] == waypoint_id:
                waypoint['status'] = new_status
                waypoint['updated_at'] = datetime.now().isoformat()
                self._save_waypoints()
                print(f"Waypoint #{waypoint_id} status updated to: {new_status}")
                return True
        return False
    
    def generate_kml_file(self):
        """Generate KML file for Google Earth visualization"""
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Rescue Locations</name>
    <description>Detected humans and animals requiring rescue</description>
    
    <Style id="humanIcon">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.2</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/shapes/man.png</href>
            </Icon>
        </IconStyle>
    </Style>
    
    <Style id="animalIcon">
        <IconStyle>
            <color>ff00ffff</color>
            <scale>1.0</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
            </Icon>
        </IconStyle>
    </Style>
    
'''
        
        for waypoint in self.rescue_waypoints:
            style_id = "humanIcon" if "human" in waypoint['type'] or "person" in waypoint['type'] else "animalIcon"
            
            kml_content += f'''    <Placemark>
        <name>{waypoint['type'].title()} #{waypoint['detection_id']}</name>
        <description>
            Detection ID: {waypoint['detection_id']}
            Type: {waypoint['type']}
            Priority: {waypoint['priority']}
            Status: {waypoint['status']}
            Time: {waypoint['timestamp']}
            Altitude: {waypoint['altitude']:.1f}m
        </description>
        <styleUrl>#{style_id}</styleUrl>
        <Point>
            <coordinates>{waypoint['longitude']:.6f},{waypoint['latitude']:.6f},{waypoint['altitude']:.1f}</coordinates>
        </Point>
    </Placemark>
    
'''
        
        kml_content += '''</Document>
</kml>'''
        
        kml_file = self.locations_dir / f"rescue_locations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.kml"
        with open(kml_file, 'w') as f:
            f.write(kml_content)
        
        print(f"KML file generated: {kml_file}")
        print("Open this file in Google Earth to visualize rescue locations")
        return kml_file
    
    def calculate_distance_to_detection(self, detection_location):
        """Calculate distance from current drone position to detection"""
        if not self.current_gps:
            return None
        
        # Using Haversine formula for distance calculation
        lat1 = math.radians(self.current_gps['latitude'])
        lon1 = math.radians(self.current_gps['longitude'])
        lat2 = math.radians(detection_location['gps_location']['latitude'])
        lon2 = math.radians(detection_location['gps_location']['longitude'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        earth_radius = 6371000
        distance = earth_radius * c
        
        return distance
    
    def generate_location_summary(self):
        """Generate summary of all detection locations"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_detections': len(self.detection_locations),
            'total_waypoints': len(self.rescue_waypoints),
            'current_gps': self.current_gps,
            'detection_summary': {}
        }
        
        # Summarize by detection type
        for detection_id, locations in self.detection_locations.items():
            if locations:
                latest_location = locations[-1]
                summary['detection_summary'][detection_id] = {
                    'type': latest_location['detection_type'],
                    'priority': latest_location['priority'],
                    'first_detected': locations[0]['timestamp'],
                    'last_detected': latest_location['timestamp'],
                    'total_location_updates': len(locations),
                    'latest_location': latest_location['gps_location']
                }
        
        # Save summary
        summary_file = self.locations_dir / f"location_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Location summary saved: {summary_file}")
        return summary
    
    