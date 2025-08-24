import json
import os
from datetime import datetime

class EnhancedLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"detection_log_{timestamp}.json")
        
        self.log_data = {
            "start_time": datetime.now().isoformat(),
            "videos_processed": [],
            "anomalies_detected": [],
            "object_statistics": {}
        }
    
    def log_video_processing(self, video_path, frame_count, processing_time):
        """Log video processing details"""
        self.log_data["videos_processed"].append({
            "video_path": video_path,
            "frame_count": frame_count,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_anomaly(self, frame_number, anomalies, video_path):
        """Log detected anomalies"""
        for anomaly in anomalies:
            self.log_data["anomalies_detected"].append({
                "frame": frame_number,
                "anomaly": anomaly,
                "video_path": video_path,
                "timestamp": datetime.now().isoformat()
            })
    
    def update_object_stats(self, tracked_objects):
        """Update object statistics"""
        for object_id, object_data in tracked_objects.items():
            class_name = object_data['class_name']
            
            if class_name not in self.log_data["object_statistics"]:
                self.log_data["object_statistics"][class_name] = {
                    "count": 0,
                    "total_confidence": 0,
                    "max_confidence": 0
                }
            
            stats = self.log_data["object_statistics"][class_name]
            stats["count"] += 1
            stats["total_confidence"] += object_data['confidence']
            stats["max_confidence"] = max(stats["max_confidence"], object_data['confidence'])
    
    def save_log(self):
        """Save log to file"""
        self.log_data["end_time"] = datetime.now().isoformat()
        
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        print(f"Log saved to: {self.log_file}")
        
        # Also generate a summary report
        self.generate_summary()
    
    def generate_summary(self):
        """Generate a summary report"""
        summary_file = self.log_file.replace('.json', '_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("SURVEILLANCE SYSTEM SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing Period: {self.log_data['start_time']} to {self.log_data['end_time']}\n")
            f.write(f"Videos Processed: {len(self.log_data['videos_processed'])}\n")
            f.write(f"Anomalies Detected: {len(self.log_data['anomalies_detected'])}\n\n")
            
            f.write("OBJECT STATISTICS:\n")
            for class_name, stats in self.log_data['object_statistics'].items():
                avg_confidence = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
                f.write(f"  {class_name}: {stats['count']} detections, "
                       f"Avg confidence: {avg_confidence:.3f}, "
                       f"Max confidence: {stats['max_confidence']:.3f}\n")
            
            f.write("\nANOMALIES DETECTED:\n")
            for anomaly in self.log_data['anomalies_detected']:
                f.write(f"  Frame {anomaly['frame']}: {anomaly['anomaly']}\n")