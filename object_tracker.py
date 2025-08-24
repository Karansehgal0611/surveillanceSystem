import numpy as np
from collections import OrderedDict

class ObjectTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize object tracker
        
        Args:
            max_disappeared: How many frames an object can disappear before being removed
            max_distance: Maximum distance (pixels) to associate objects between frames
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # {object_id: object_data}
        self.disappeared = OrderedDict()  # {object_id: frames_missing}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox, class_name, confidence):
        """Register a new object with the tracker"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,      # (x, y) center point
            'bbox': bbox,              # (x, y, w, h) bounding box
            'class_name': class_name,  # Object class name
            'confidence': confidence,  # Detection confidence
            'path': [centroid]         # History of positions
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1  # Return the new ID
        
    def deregister(self, object_id):
        """Remove an object from tracking"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        
    def update(self, detections):
        """
        Update tracker with new detections from current frame
        
        Args:
            detections: List of detection dictionaries from object detector
        
        Returns:
            Dictionary of tracked objects with their IDs
        """
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Prepare input data from detections
        input_centroids = []
        input_bboxes = []
        input_classes = []
        input_confidences = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            centroid_x = int(x + w / 2)
            centroid_y = int(y + h / 2)
            input_centroids.append((centroid_x, centroid_y))
            input_bboxes.append(detection['bbox'])
            input_classes.append(detection['class_name'])
            input_confidences.append(detection['confidence'])
        
        # Convert to numpy array for distance calculations
        input_centroids = np.array(input_centroids)
        
        # If no objects currently tracked, register all detections as new objects
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], 
                             input_classes[i], input_confidences[i])
        else:
            # Match existing objects with new detections
            object_ids = list(self.objects.keys())
            object_centroids = []
            
            # Get current centroids of tracked objects
            for object_id, object_data in self.objects.items():
                object_centroids.append(object_data['centroid'])
            
            object_centroids = np.array(object_centroids)
            
            # Calculate Euclidean distances between all objects and detections
            distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            
            # Find best matches (minimum distance)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Update matched objects
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                # Only match if distance is within threshold
                if distances[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                # Update object with new detection
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = input_bboxes[col]
                self.objects[object_id]['path'].append(input_centroids[col])
                self.disappeared[object_id] = 0  # Reset disappearance counter
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle objects that disappeared in this frame
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects from unmatched detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col],
                             input_classes[col], input_confidences[col])
        
        return self.objects