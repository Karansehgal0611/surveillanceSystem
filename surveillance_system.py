import cv2
import os
import numpy as np
import glob
from collections import defaultdict,deque
import time
import csv
# Add this import
from object_tracker import ObjectTracker
# Add this import
from anomaly_detector import AnomalyDetector
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the dataset
DATASET_PATH = os.path.join(script_dir, "data", "Avenue Dataset")


class AvenueDatasetLoader:
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        
        # Try to load from config file first
        try:
            from dataset_config import TRAINING_VIDEOS, TESTING_VIDEOS
            self.training_videos = TRAINING_VIDEOS
            self.testing_videos = TESTING_VIDEOS
            print("Loaded video paths from dataset_config.py")
            if not self.training_videos and not self.testing_videos:
                print("Config file is empty, searching for videos...")
                self.training_videos, self.testing_videos = self._find_videos()
        except ImportError:
            # If config doesn't exist, search for videos
            print("dataset_config.py not found, searching for videos...")
            self.training_videos, self.testing_videos = self._find_videos()
        
        print(f"Found {len(self.training_videos)} training videos")
        print(f"Found {len(self.testing_videos)} testing videos")
        
        # Print video paths for debugging
        if self.training_videos:
            print("Training videos:")
            for video in self.training_videos[:3]:  # Show first 3 only
                print(f"  {video}")
            if len(self.training_videos) > 3:
                print(f"  ... and {len(self.training_videos) - 3} more")
        
        if self.testing_videos:
            print("Testing videos:")
            for video in self.testing_videos[:3]:  # Show first 3 only
                print(f"  {video}")
            if len(self.testing_videos) > 3:
                print(f"  ... and {len(self.testing_videos) - 3} more")
    
    def _find_videos(self):
        """Find video files in the dataset directory"""
        # Look for video files with common extensions
        video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv', '*.AVI', '*.MP4']
        all_videos = []
        
        for ext in video_extensions:
            videos = glob.glob(os.path.join(self.dataset_path, '**', ext), recursive=True)
            all_videos.extend(videos)
        
        # Separate into training and testing videos based on folder names
        training_videos = []
        testing_videos = []
        
        for video_path in all_videos:
            video_path_lower = video_path.lower()
            if 'train' in video_path_lower or 'training' in video_path_lower:
                training_videos.append(video_path)
            elif 'test' in video_path_lower or 'testing' in video_path_lower:
                testing_videos.append(video_path)
            else:
                # If we can't determine, check parent directory
                parent_dir = os.path.basename(os.path.dirname(video_path)).lower()
                if 'train' in parent_dir or 'training' in parent_dir:
                    training_videos.append(video_path)
                elif 'test' in parent_dir or 'testing' in parent_dir:
                    testing_videos.append(video_path)
                else:
                    # Default to training if we can't determine
                    training_videos.append(video_path)
        
        # Sort the lists
        training_videos.sort()
        testing_videos.sort()
        
        return training_videos, testing_videos
        
    def get_training_videos(self):
        return self.training_videos
        
    def get_testing_videos(self):
        return self.testing_videos


class VideoProcessor:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def initialize_video(self):
        """Initialize video capture object"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.frame_count}")
        return True
    
    def extract_frames(self, resize=None, normalize=False, max_frames=None, save_frames=False):
        """
        Extract frames from video with optional resizing and normalization
        """
        if not self.initialize_video():
            return [], []
            
        frames = []
        frame_numbers = []
        count = 0
        
        # Create frames directory if saving frames
        if save_frames:
            frames_dir = os.path.join(self.output_dir, "extracted_frames")
            os.makedirs(frames_dir, exist_ok=True)
        
        while True:
            ret, frame = self.cap.read()
            if not ret or (max_frames and count >= max_frames):
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if resize:
                frame_rgb = cv2.resize(frame_rgb, resize)
            
            # Normalize if needed
            if normalize:
                frame_rgb = frame_rgb.astype(np.float32) / 255.0
                
            frames.append(frame_rgb)
            frame_numbers.append(count)
            
            # Save frame as image if requested
            if save_frames:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if normalize:
                    frame_bgr = (frame_bgr * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{count:04d}.jpg"), frame_bgr)
            
            count += 1
            
            if count % 100 == 0:
                print(f"Processed {count} frames")
        
        self.cap.release()
        print(f"Finished extracting {len(frames)} frames")
        return frames, frame_numbers
    
    def get_video_writer(self, output_name, fps=None, size=None):
        """Create a video writer for output"""
        if fps is None:
            fps = self.fps
            
        if size is None:
            size = (self.width, self.height)
            
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.join(self.output_dir, output_name)
        return cv2.VideoWriter(output_path, fourcc, fps, size)
    
    def release(self):
        """Release video capture object"""
        if self.cap:
            self.cap.release()


class ObjectDetector:
    def __init__(self, conf_threshold=0.5):
        """
        Initialize object detector - tries YOLOv3 first, falls back to YOLOv8
        """
        self.conf_threshold = conf_threshold
        
        # Try to initialize YOLOv3 first
        if self._init_yolov3():
            print("Using YOLOv3 for object detection")
        else:
            # Fall back to YOLOv8
            if self._init_yolov8():
                print("Using YOLOv8 for object detection")
            else:
                raise Exception("Could not initialize any object detector")
    
    def _init_yolov3(self):
        """Initialize YOLOv3 model for OpenCV DNN"""
        # Try different possible paths for YOLO files
        possible_paths = [
            "data/models/",
            "models/",
            ""  # Current directory
        ]
        
        possible_files = {
            "weights": ["yolov3.weights"],
            "config": ["yolov3.cfg"],
            "classes": ["coco.names"]
        }
        
        self.weights_path = None
        self.config_path = None
        self.classes_path = None
        
        # Find the model files
        for path in possible_paths:
            for weights_file in possible_files["weights"]:
                weights_candidate = os.path.join(path, weights_file)
                if os.path.exists(weights_candidate):
                    self.weights_path = weights_candidate
                    break
            
            for config_file in possible_files["config"]:
                config_candidate = os.path.join(path, config_file)
                if os.path.exists(config_candidate):
                    self.config_path = config_candidate
                    break
                    
            for classes_file in possible_files["classes"]:
                classes_candidate = os.path.join(path, classes_file)
                if os.path.exists(classes_candidate):
                    self.classes_path = classes_candidate
                    break
                    
            if self.weights_path and self.config_path and self.classes_path:
                break
        
        # Check if all files were found
        if not self.weights_path or not self.config_path or not self.classes_path:
            print("YOLOv3 files not found, will try YOLOv8")
            return False
        
        print(f"Using weights: {self.weights_path}")
        print(f"Using config: {self.config_path}")
        print(f"Using classes: {self.classes_path}")
        
        try:
            # Load YOLO network
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(self.classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate random colors for each class
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            self.use_yolov8 = False
            print("YOLOv3 model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading YOLOv3: {e}")
            return False
    
    def _init_yolov8(self):
        """Initialize YOLOv8 model as a fallback"""
        try:
            from ultralytics import YOLO #type: ignore
            # Load YOLOv8 model (will auto-download if not present)
            print("Using YOLOv8 as fallback (will auto-download)...")
            self.model = YOLO('yolov8n.pt')  # This will auto-download
            self.classes = self.model.names
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            self.use_yolov8 = True
            print("YOLOv8 model loaded successfully!")
            return True
        except ImportError:
            print("Please install ultralytics: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Error loading YOLOv8: {e}")
            return False
    
    def detect_objects(self, frame):
        """Detect objects in a frame"""
        if self.use_yolov8:
            return self._detect_yolov8(frame)
        else:
            return self._detect_yolov3(frame)
    
    def _detect_yolov3(self, frame):
        """Detect objects using YOLOv3"""
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maxima suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.4)
        
        # Prepare results
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': (x, y, w, h)
                })
        
        return results
    
    def _detect_yolov8(self, frame):
        """Detect objects using YOLOv8"""
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'class_id': cls_id,
                        'class_name': self.classes[cls_id],
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detection bounding boxes and labels on frame"""
        # Convert back to BGR for OpenCV if needed
        if frame.dtype == np.float32:
            frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            color = self.colors[detection['class_id']]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame


class ObjectDetectionSystem:
    def __init__(self, dataset_path=DATASET_PATH, output_dir="output"):
        self.dataset_loader = AvenueDatasetLoader(dataset_path)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize detector (we'll do this lazily when needed)
        self.detector = None
        self.anomaly_detector = AnomalyDetector()  # Initialize anomaly detector
        self.previous_frame = None  # For motion detection
        self.object_counts = defaultdict(int)
        self.detection_log = []
    
    def initialize_detector(self):
        """Initialize the object detector when needed"""
        if self.detector is None:
            try:
                self.detector = ObjectDetector()
                return True
            except Exception as e:
                print(f"Error initializing detector: {e}")
                return False
        return True
    
    def process_video(self, video_path, output_video_name="output.avi", max_frames=None):
        """Process a video and save object detection results"""
        # Initialize detector if not already initialized
        if not self.initialize_detector():
            return False
            
        # Initialize object tracker
        tracker = ObjectTracker(max_disappeared=30, max_distance=100)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Processing video: {video_name}")
        
        # Initialize video processor
        video_processor = VideoProcessor(video_path, self.output_dir)
        
        # Initialize video
        if not video_processor.initialize_video():
            print("Failed to initialize video")
            return False
        
        # Prepare video writer
        out = video_processor.get_video_writer(output_video_name)
        
        frame_idx = 0
        processing_times = []
        self.previous_frame = None  # Reset for each video
        
        while True:
            ret, frame = video_processor.cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            # Start timing
            start_time = time.time()
            
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects
            detections = self.detector.detect_objects(frame_rgb)
            
            # Update tracker with new detections
            tracked_objects = tracker.update(detections)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_all_anomalies(
                tracked_objects, frame_rgb, frame_idx
            )
            
            # Update object counts and log (with anomaly info)
            for object_id, object_data in tracked_objects.items():
                # Only count once per object to avoid double-counting in same frame
                if frame_idx % 30 == 0:  # Update count every 30 frames
                    self.object_counts[object_data['class_name']] += 1
                
                # Log detection with object ID and anomaly info
                self.detection_log.append({
                    'video': video_name,
                    'frame': frame_idx,
                    'time': frame_idx / video_processor.fps,
                    'object_id': object_id,
                    'object': object_data['class_name'],
                    'confidence': object_data['confidence'],
                    'bbox': object_data['bbox'],
                    'centroid': object_data['centroid'],
                    'is_anomaly': len(anomalies) > 0,
                    'anomalies': anomalies
                })
            
            frame_with_detections = self.draw_detections_with_anomalies(
                frame.copy(), tracked_objects, anomalies, frame_idx
            )
            
            # Display information
            self._display_info(frame_with_detections, frame_idx, processing_times, 
                            tracked_objects, anomalies)
            
            # Write frame to output video
            out.write(frame_with_detections)
            
            # Display frame
            cv2.imshow('Surveillance System - Detection & Anomalies', frame_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # End timing
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames, Tracking {len(tracked_objects)} objects")
                if anomalies:
                    print(f"Anomalies detected: {anomalies}")
        
        # Clean up
        video_processor.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        self._print_summary(video_name, frame_idx, processing_times, len(tracked_objects), anomalies)
        return True
    
    def draw_detections_with_anomalies(self, frame, tracked_objects, anomalies, frame_count=None):
        """
        Draw bounding boxes and labels for tracked objects, highlighting anomalies
        
        Args:
            frame: Input frame
            tracked_objects: Dictionary of tracked objects
            anomalies: List of anomaly messages
            frame_count: Current frame number (optional)
        
        Returns:
            Frame with drawn detections and anomaly indicators
        """
        # Make a copy of the frame to draw on
        frame_with_detections = frame.copy()
        
        # Convert to BGR if needed (OpenCV uses BGR format)
        if len(frame_with_detections.shape) == 3 and frame_with_detections.shape[2] == 3:
            # Check if it's RGB and convert to BGR
            frame_with_detections = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
        
        # Pre-process anomalies to map them to object IDs
        anomaly_map = defaultdict(list)
        object_anomaly_types = {}  # Store what type of anomaly each object has
        
        for anomaly in anomalies:
            # Map anomalies to object IDs based on different message formats
            if "Object" in anomaly and "ID:" in anomaly:
                # Format: "Abandoned backpack detected (ID: 5)"
                try:
                    obj_id_str = anomaly.split("ID:")[1].split(")")[0].strip()
                    obj_id = int(obj_id_str)
                    anomaly_map[obj_id].append(anomaly)
                    object_anomaly_types[obj_id] = "abandoned"
                except (IndexError, ValueError):
                    pass
            elif "Person" in anomaly:
                # Format: "Person 3 running" or "Suspicious stationary behavior: Person 3"
                try:
                    # Extract number after "Person"
                    parts = anomaly.split("Person")
                    if len(parts) > 1:
                        obj_id_str = ''.join(filter(str.isdigit, parts[1]))
                        if obj_id_str:
                            obj_id = int(obj_id_str)
                            anomaly_map[obj_id].append(anomaly)
                            if "running" in anomaly.lower():
                                object_anomaly_types[obj_id] = "running"
                            else:
                                object_anomaly_types[obj_id] = "suspicious"
                except (IndexError, ValueError):
                    pass
            elif "ID:" in anomaly:
                # Format with explicit ID notation
                try:
                    obj_id_str = anomaly.split("ID:")[1].split()[0].strip()
                    obj_id = int(obj_id_str)
                    anomaly_map[obj_id].append(anomaly)
                    object_anomaly_types[obj_id] = "general"
                except (IndexError, ValueError):
                    pass
        
        # Draw all tracked objects
        for object_id, object_data in tracked_objects.items():
            x, y, w, h = object_data['bbox']
            centroid = object_data['centroid']
            class_name = object_data['class_name']
            confidence = object_data['confidence']
            
            # Check if this object has any anomalies and what type
            has_anomaly = object_id in anomaly_map
            anomaly_type = object_anomaly_types.get(object_id, "")
            
            # Choose color and style based on anomaly type
            if has_anomaly:
                if anomaly_type == "abandoned":
                    color = (0, 0, 255)  # Red for abandoned objects
                    thickness = 4
                elif anomaly_type == "running":
                    color = (255, 0, 0)  # Blue for running people
                    thickness = 3
                elif anomaly_type == "suspicious":
                    color = (0, 255, 255)  # Yellow for suspicious behavior
                    thickness = 3
                else:
                    color = (255, 165, 0)  # Orange for general anomalies
                    thickness = 3
            else:
                # Normal objects
                if class_name == 'person':
                    color = (0, 255, 0)  # Green for people
                elif class_name == 'car':
                    color = (255, 0, 0)  # Blue for cars
                elif class_name in ['bag', 'backpack', 'suitcase', 'handbag']:
                    color = (255, 165, 0)  # Orange for bags/luggage
                else:
                    color = (128, 128, 128)  # Gray for other objects
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label with object ID and anomaly indicator
            anomaly_indicator = " ⚠️" if has_anomaly else ""
            label = f"ID:{object_id} {class_name}{anomaly_indicator}: {confidence:.2f}"
            
            # Background for better text visibility
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_with_detections, (x, y - text_height - 10), 
                        (x + text_width, y), color, -1)
            
            # Text in contrasting color
            text_color = (255, 255, 255) if has_anomaly else (0, 0, 0)
            cv2.putText(frame_with_detections, label, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Draw centroid
            cv2.circle(frame_with_detections, centroid, 4, color, -1)
            
            # Draw path (last 10 positions) - only for moving objects or anomalies
            if len(object_data['path']) > 1 and (has_anomaly or class_name == 'person'):
                for i in range(1, min(15, len(object_data['path']))):
                    alpha = 1.0 - (i / min(15, len(object_data['path'])))  # Fade out older points
                    path_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame_with_detections, object_data['path'][i - 1], 
                            object_data['path'][i], path_color, 2)
        
        # Add anomaly alert if any anomalies detected
        if anomalies:
            # Draw colored border based on severity
            border_color = (0, 0, 255)  # Red for high severity
            border_thickness = 15
            
            cv2.rectangle(frame_with_detections, (0, 0), 
                        (frame_with_detections.shape[1], frame_with_detections.shape[0]), 
                        border_color, border_thickness)
            
            # Add anomaly header
            cv2.putText(frame_with_detections, "ANOMALY DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # List specific anomalies (max 4)
            for i, anomaly in enumerate(anomalies[:4]):
                # Shorten long messages for display
                if len(anomaly) > 50:
                    anomaly = anomaly[:47] + "..."
                
                cv2.putText(frame_with_detections, f"• {anomaly}", (50, 100 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show count if there are more anomalies
            if len(anomalies) > 4:
                cv2.putText(frame_with_detections, f"• ... and {len(anomalies) - 4} more", 
                        (50, 100 + 4 * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add frame counter if provided
        if frame_count is not None:
            # Background for frame counter
            cv2.rectangle(frame_with_detections, (5, 5), (200, 40), (0, 0, 0), -1)
            cv2.putText(frame_with_detections, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_with_detections

    def draw_tracked_objects(self, frame, tracked_objects, anomalies):
        """Draw tracked objects with IDs, paths, and anomaly highlighting"""
        # Draw all tracked objects
        for object_id, object_data in tracked_objects.items():
            x, y, w, h = object_data['bbox']
            centroid = object_data['centroid']
            class_name = object_data['class_name']
            confidence = object_data['confidence']
            
            # Choose color based on object type
            if class_name == 'person':
                color = (0, 255, 0)  # Green for people
            elif class_name == 'car':
                color = (255, 0, 0)  # Blue for cars
            else:
                color = (0, 165, 255)  # Orange for other objects
            
            # Draw bounding box (thicker if anomalies detected)
            thickness = 3 if anomalies else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label with object ID
            label = f"ID:{object_id} {class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw centroid
            cv2.circle(frame, centroid, 4, color, -1)
            
            # Draw path (last 10 positions)
            if len(object_data['path']) > 1:
                for i in range(1, min(10, len(object_data['path']))):
                    cv2.line(frame, object_data['path'][i - 1], object_data['path'][i], color, 2)
        
        # Add anomaly alert if any anomalies detected
        if anomalies:
            # Draw red border around entire frame
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            
            # Add anomaly text
            cv2.putText(frame, "ANOMALY DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # List specific anomalies
            for i, anomaly in enumerate(anomalies[:3]):  # Show first 3 anomalies max
                cv2.putText(frame, anomaly, (50, 100 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def calculate_motion_level(self, previous_frame, current_frame):
        """Calculate motion level between two frames"""
        # Convert to grayscale
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray_prev, gray_curr)
        
        # Apply threshold to highlight significant changes
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion level (percentage of changed pixels)
        motion_pixels = np.sum(thresh) / 255
        total_pixels = thresh.size
        motion_level = motion_pixels / total_pixels
        
        return motion_level
    
    def _display_info(self, frame, frame_idx, processing_times, tracked_objects, anomalies):
        """Display information on the frame"""
        # Calculate current FPS
        if processing_times:
            avg_time = np.mean(processing_times[-10:]) if len(processing_times) > 10 else np.mean(processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            fps = 0
        
        # Count objects by type
        object_counts = {}
        for object_data in tracked_objects.values():
            class_name = object_data['class_name']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Display frame info
        y_offset = 30
        info_lines = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.1f}",
            f"Objects: {len(tracked_objects)}",
            f"Anomalies: {len(anomalies)}"
        ]
        
        # Choose text color based on anomalies
        text_color = (0, 0, 255) if anomalies else (0, 255, 0)
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y_offset + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Display object counts (if space available)
        if len(tracked_objects) < 5:  # Only show if not too many objects
            for i, (obj_type, count) in enumerate(object_counts.items()):
                if y_offset + 120 + i * 30 < frame.shape[0] - 50:  # Check if space available
                    cv2.putText(frame, f"{obj_type}: {count}", (10, y_offset + 120 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _print_summary(self, video_name, total_frames, processing_times, total_tracked_objects, anomalies):
        """Print processing summary"""
        if processing_times:
            avg_time = np.mean(processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            fps = 0
        
        print("\n" + "="*50)
        print(f"PROCESSING SUMMARY - {video_name}")
        print("="*50)
        print(f"Total frames processed: {total_frames}")
        print(f"Average processing time per frame: {avg_time:.3f}s")
        print(f"Average FPS: {fps:.1f}")
        print(f"Total processing time: {sum(processing_times):.2f}s")
        print(f"Maximum objects tracked simultaneously: {total_tracked_objects}")
        
        if anomalies:
            print(f"Anomalies detected in this video: {len(anomalies)}")
            for anomaly in anomalies[:5]:  # Show first 5 anomalies
                print(f"  - {anomaly}")
            if len(anomalies) > 5:
                print(f"  - ... and {len(anomalies) - 5} more")
        else:
            print("No anomalies detected in this video")
        
        print("\nObject detection summary:")
        for obj, count in sorted(self.object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{obj}: {count}")
    
    def save_detection_log(self, filename_prefix="detection_log"):
        """Save detection results to a CSV file with anomaly information"""
        from datetime import datetime
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Updated header to include anomaly information
            writer.writerow([
                'video', 'frame', 'time', 'object_id', 'object', 'confidence',
                'x', 'y', 'width', 'height', 'centroid_x', 'centroid_y',
                'is_anomaly', 'anomalies'
            ])
            
            for log in self.detection_log:
                x, y, w, h = log['bbox']
                centroid_x, centroid_y = log.get('centroid', (0, 0))
                anomalies = log.get('anomalies', [])
                
                writer.writerow([
                    log['video'],
                    log['frame'],
                    f"{log['time']:.2f}",
                    log.get('object_id', 'N/A'),
                    log['object'],
                    f"{log['confidence']:.3f}",
                    x, y, w, h,
                    centroid_x, centroid_y,
                    'YES' if log.get('is_anomaly', False) else 'NO',
                    '; '.join(anomalies) if anomalies else 'None'
                ])
        
        print(f"Detection log with anomalies saved to {output_path}")
        return output_path
        
    def extract_all_frames(self, video_type="training", resize=(640, 360), max_frames_per_video=50, save_frames=True):
        """Extract frames from all videos of a specific type"""
        if video_type == "training":
            videos = self.dataset_loader.get_training_videos()
        else:
            videos = self.dataset_loader.get_testing_videos()
        
        if not videos:
            print(f"No {video_type} videos found!")
            return
        
        print(f"Extracting frames from {len(videos)} {video_type} videos")
        
        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Extracting frames from: {video_name}")
            
            video_processor = VideoProcessor(video_path, self.output_dir)
            frames, frame_numbers = video_processor.extract_frames(
                resize=resize, 
                max_frames=max_frames_per_video,
                save_frames=save_frames
            )
            
            print(f"Extracted {len(frames)} frames from {video_name}")
    
    def process_all_videos(self, video_type="training", max_frames_per_video=None):
        """Process all videos of a specific type"""
        if video_type == "training":
            videos = self.dataset_loader.get_training_videos()
        else:
            videos = self.dataset_loader.get_testing_videos()
        
        if not videos:
            print(f"No {video_type} videos found!")
            return
        
        print(f"Processing {len(videos)} {video_type} videos")
        
        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"{video_type}_{video_name}_output.avi"
            
            success = self.process_video(video_path, output_name, max_frames_per_video)
            if not success:
                print(f"Failed to process {video_name}, skipping to next video")
                break


def main():
    # Create output directory for tracking results
    tracking_output_dir = "output_tracking_anomaly_better_try1"
    if not os.path.exists(tracking_output_dir):
        os.makedirs(tracking_output_dir)
    
    # Create object detection system with the new output directory
    system = ObjectDetectionSystem(output_dir=tracking_output_dir)

    
    # Process videos with tracking
    print("\nProcessing training videos with object detection and tracking...")
    system.process_all_videos(video_type="training", max_frames_per_video=100)
    
    print("\nProcessing testing videos with object detection and tracking...")
    system.process_all_videos(video_type="testing", max_frames_per_video=100)
    
    # Save detection log with tracking info
    system.save_detection_log()
    
    print("Processing with tracking completed!")



if __name__ == "__main__":
    main()