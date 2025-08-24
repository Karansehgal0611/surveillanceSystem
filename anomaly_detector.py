import numpy as np
from collections import deque,defaultdict
import cv2

class AbandonedObjectDetector:
    def __init__(self, abandonment_time=8.0, max_distance=150):
        self.abandonment_time = abandonment_time  # seconds
        self.max_distance = max_distance  # pixels
        self.tracked_objects = {}  # object_id -> {stationary_time, position, last_owner, frames_stationary}
        
    def detect_abandoned_objects(self, tracked_objects, frame_count, fps):
        anomalies = []
        current_time = frame_count / fps
        
        # Track portable objects that could be abandoned
        portable_objects = {}
        people_positions = {}
        
        # Separate people and portable objects
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['class_name'] == 'person':
                people_positions[obj_id] = obj_data['centroid']
            elif obj_data['class_name'] in ['backpack', 'handbag', 'suitcase', 'bag', 'purse']:
                portable_objects[obj_id] = obj_data
        
        # Update tracked objects
        for obj_id, obj_data in portable_objects.items():
            current_pos = obj_data['centroid']
            
            if obj_id in self.tracked_objects:
                # Check if object moved significantly
                prev_pos = self.tracked_objects[obj_id]['position']
                distance_moved = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
                
                if distance_moved < 5:  # Minimal movement
                    self.tracked_objects[obj_id]['frames_stationary'] += 1
                    stationary_time = self.tracked_objects[obj_id]['frames_stationary'] / fps
                    
                    # Find current nearest person
                    nearest_person, min_distance = self._find_nearest_person(current_pos, people_positions)
                    
                    # Check if abandoned (stationary for long time AND no owner nearby)
                    if stationary_time > self.abandonment_time and min_distance > self.max_distance:
                        anomalies.append(f"Abandoned {obj_data['class_name']} detected (ID: {obj_id})")
                    
                    # Update position and owner
                    self.tracked_objects[obj_id]['position'] = current_pos
                    self.tracked_objects[obj_id]['last_owner'] = nearest_person
                else:
                    # Object moved significantly, reset tracking
                    nearest_person, _ = self._find_nearest_person(current_pos, people_positions)
                    self.tracked_objects[obj_id] = {
                        'position': current_pos,
                        'last_owner': nearest_person,
                        'frames_stationary': 0
                    }
            else:
                # New object to track
                nearest_person, _ = self._find_nearest_person(current_pos, people_positions)
                self.tracked_objects[obj_id] = {
                    'position': current_pos,
                    'last_owner': nearest_person,
                    'frames_stationary': 0
                }
        
        return anomalies
    
    def _find_nearest_person(self, object_pos, people_positions):
        if not people_positions:
            return None, float('inf')
        
        min_distance = float('inf')
        nearest_person = None
        
        for person_id, person_pos in people_positions.items():
            distance = np.linalg.norm(np.array(object_pos) - np.array(person_pos))
            if distance < min_distance:
                min_distance = distance
                nearest_person = person_id
        
        return nearest_person, min_distance

class AnomalyDetector:
    def __init__(self, fps=25):
        self.fps = fps
        self.frame_history = deque(maxlen=30)  # Store recent frames for motion analysis
        self.object_history = deque(maxlen=100)  # Store object movements
        self.abandoned_detector = AbandonedObjectDetector()
        
        # Configurable thresholds
        self.running_threshold = 8.0  # pixels/frame for running
        self.crowd_threshold = 5      # number of people for crowd
        self.motion_anomaly_threshold = 0.15  # motion level for anomaly
        self.stationary_time_threshold = 5.0  # seconds for suspicious stationary behavior
        
    def detect_running_people(self, tracked_objects):
        """Detect people moving faster than normal walking speed"""
        anomalies = []
        
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['class_name'] == 'person' and len(obj_data['path']) >= 5:
                # Calculate speed (pixels per frame)
                recent_path = obj_data['path'][-5:]
                total_distance = 0
                
                for i in range(1, len(recent_path)):
                    distance = np.linalg.norm(
                        np.array(recent_path[i]) - np.array(recent_path[i-1])
                    )
                    total_distance += distance
                
                avg_speed = total_distance / (len(recent_path) - 1)
                
                # Convert to real-world estimate (approx 1 pixel = 2-3 cm)
                speed_mps = avg_speed * 0.025 * self.fps  # meters per second
                
                if speed_mps > 3.0:  # Running speed threshold (3 m/s ≈ 10.8 km/h)
                    anomalies.append(f"Person {obj_id} running ({speed_mps:.1f} m/s)")
                
        return anomalies
    
    def detect_crowds(self, tracked_objects):
        """Detect unusually dense groups of people"""
        anomalies = []
        people = []
        
        # Collect all people positions
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['class_name'] == 'person':
                people.append((obj_id, obj_data['centroid']))
        
        if len(people) >= self.crowd_threshold:
            # Check if people are clustered together
            positions = np.array([pos for _, pos in people])
            
            if len(positions) > 1:
                # Calculate average distance between people
                distances = []
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        distance = np.linalg.norm(positions[i] - positions[j])
                        distances.append(distance)
                
                if distances:
                    avg_distance = np.mean(distances)
                    
                    # If people are too close together (crowded)
                    if avg_distance < 100:  # pixels threshold
                        anomalies.append(f"Crowd detected: {len(people)} people, avg distance {avg_distance:.1f}px")
        
        return anomalies
    
    def detect_unusual_motion_patterns(self, tracked_objects, current_frame):
        """Detect unusual movement patterns"""
        anomalies = []
        
        # Store current frame for motion analysis
        self.frame_history.append(current_frame)
        
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['class_name'] == 'person' and len(obj_data['path']) >= 10:
                path = np.array(obj_data['path'][-10:])
                
                # 1. Detect erratic movement (sudden direction changes)
                if len(path) >= 3:
                    directions = []
                    for i in range(1, len(path)):
                        vector = path[i] - path[i-1]
                        if np.linalg.norm(vector) > 0.1:  # Avoid division by zero
                            directions.append(vector / np.linalg.norm(vector))
                    
                    if len(directions) >= 2:
                        # Calculate direction changes
                        direction_changes = []
                        for i in range(1, len(directions)):
                            dot_product = np.dot(directions[i], directions[i-1])
                            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                            direction_changes.append(angle)
                        
                        # If frequent large direction changes
                        large_changes = sum(1 for angle in direction_changes if angle > np.pi/3)  # >60 degrees
                        if large_changes >= 2:
                            anomalies.append(f"Erratic movement: Person {obj_id}")
                
                # 2. Detect circling/loitering behavior
                if len(path) >= 15:
                    # Calculate net displacement vs total distance
                    net_displacement = np.linalg.norm(path[-1] - path[0])
                    total_distance = sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))
                    
                    if total_distance > 0:
                        efficiency = net_displacement / total_distance
                        if efficiency < 0.3:  # Low movement efficiency (circling)
                            anomalies.append(f"Circling behavior: Person {obj_id}")
        
        return anomalies
    
    def detect_abandoned_objects(self, tracked_objects, frame_count):
        """Detect objects left behind by people"""
        return self.abandoned_detector.detect_abandoned_objects(tracked_objects, frame_count, self.fps)
    
    def detect_sudden_motion_changes(self, current_frame):
        """Detect sudden changes in overall scene motion"""
        anomalies = []
        
        if len(self.frame_history) >= 10:
            # Calculate motion between consecutive frames
            motion_levels = []
            for i in range(1, len(self.frame_history)):
                gray_prev = cv2.cvtColor(self.frame_history[i-1], cv2.COLOR_RGB2GRAY)
                gray_curr = cv2.cvtColor(self.frame_history[i], cv2.COLOR_RGB2GRAY)
                
                frame_diff = cv2.absdiff(gray_prev, gray_curr)
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                
                motion_pixels = np.sum(thresh) / 255
                motion_level = motion_pixels / thresh.size
                motion_levels.append(motion_level)
            
            # Check for sudden changes
            if len(motion_levels) >= 5:
                recent_avg = np.mean(motion_levels[-3:])
                previous_avg = np.mean(motion_levels[-6:-3])
                
                if recent_avg > self.motion_anomaly_threshold:
                    anomalies.append(f"High motion level: {recent_avg:.3f}")
                
                elif abs(recent_avg - previous_avg) > previous_avg * 0.8:  # 80% change
                    anomalies.append(f"Sudden motion change: {previous_avg:.3f} → {recent_avg:.3f}")
        
        return anomalies
    
    def detect_all_anomalies(self, tracked_objects, current_frame, frame_count):
        """Run all anomaly detection methods with balanced sensitivity"""
        all_anomalies = []
        
        # Run all detectors
        all_anomalies.extend(self.detect_running_people(tracked_objects))
        all_anomalies.extend(self.detect_crowds(tracked_objects))
        all_anomalies.extend(self.detect_unusual_motion_patterns(tracked_objects, current_frame))
        all_anomalies.extend(self.detect_abandoned_objects(tracked_objects, frame_count))
        all_anomalies.extend(self.detect_sudden_motion_changes(current_frame))
        
        # Remove duplicates and limit to most relevant
        unique_anomalies = list(set(all_anomalies))
        
        # Prioritize serious anomalies (abandoned objects, running)
        serious_anomalies = [a for a in unique_anomalies if any(keyword in a.lower() 
                              for keyword in ['abandoned', 'running', 'crowd'])]
        
        # Return serious anomalies first, then others (max 5 total)
        return serious_anomalies + [a for a in unique_anomalies if a not in serious_anomalies][:5-len(serious_anomalies)]