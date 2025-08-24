# AI-Powered Surveillance System

An intelligent, multi-layered surveillance system that automates object detection, tracking, and anomaly analysis in video streams. This project uses a robust pipeline to transform raw video footage into actionable security insights, identifying behaviors like abandoned objects, running, and unusual crowd formations.

## 📋 Key Features

- **Multi-Layered Analysis**: A full pipeline for video ingestion, object detection, tracking, and behavioral analysis
- **Resilient Detection Engine**: Automatically uses YOLOv3 and falls back to YOLOv8 if local files are missing, ensuring the system always runs
- **Advanced Anomaly Detection**: Identifies a range of complex behaviors:
  - 👜 **Abandoned Objects**: Detects bags and suitcases left unattended
  - 🏃 **Suspicious Movement**: Flags individuals running or moving erratically
  - 👨‍👩‍👧‍👦 **Unusual Crowd Behavior**: Monitors for sudden or dense crowd formations
  - 💥 **Sudden Scene Changes**: Detects abrupt changes in overall scene motion
- **Persistent Object Tracking**: Assigns a unique ID to each object and tracks its path across frames
- **Rich Visualization & Logging**: Generates an output video with detailed overlays (bounding boxes, IDs, paths) and creates comprehensive JSON and CSV logs for forensic analysis
- **Modular and Extensible**: Built with a clean, object-oriented architecture that makes it easy to add new features or anomaly detection modules

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.8+
- An environment with pip to install dependencies

### Installation

1. **Clone the repository** (or download the source code):
   ```bash
   git clone https://github.com/your-username/ai-surveillance-system.git
   cd ai-surveillance-system
   ```

2. **Install the required Python libraries**:
   ```bash
   pip install opencv-python numpy ultralytics
   ```

3. **Download the YOLOv3 Model Files**:
   Run the included download script to automatically fetch the necessary model weights, configuration, and class name files.
   ```bash
   python download.py
   ```
   This will create a `models` directory and place the `yolov3.weights`, `yolov3.cfg`, and `coco.names` files inside it.

4. **Set up the Dataset**:
   - Create a `data` directory in the project's root folder
   - Inside `data`, place your video dataset (e.g., the "Avenue Dataset")
   - Ensure your video files are organized into subdirectories that contain "training" or "testing" in their names for automatic discovery

   **Example structure:**
   ```
   .
   ├── data/
   │   └── Avenue Dataset/
   │       ├── training_videos/
   │       │   ├── 01.avi
   │       │   └── 02.avi
   │       └── testing_videos/
   │           ├── 01.avi
   │           └── 02.avi
   ├── surveillance_system.py
   └── ...
   ```

## 💻 Usage

To run the full surveillance pipeline on your dataset, execute the main script from the terminal:

```bash
python surveillance_system.py
```

The script will:
- Automatically find the training and testing videos in your `data` directory
- Process each video frame by frame
- Display a real-time window showing the detections, tracking, and any anomalies
- Save the processed videos with all visualizations to the `output/` directory
- Generate detailed detection and anomaly logs in the `output/` directory

## 📂 Project Structure

```
├── data/                     # Directory for video datasets
├── models/                   # Stores YOLOv3 model files (created by download.py)
├── output/                   # Default directory for output videos and logs
├── anomaly_detector.py       # Contains the logic for detecting anomalous behaviors
├── dataset_config.py         # (Optional) Manually configure video paths
├── download.py               # Script to download YOLOv3 model files
├── enhanced_logger.py        # Handles detailed JSON and text logging
├── object_tracker.py         # Implements the centroid-based object tracking algorithm
├── surveillance_system.py    # Main script to run the entire processing pipeline
└── video_processor.py        # Helper class for video file handling
```

## 🔧 Configuration

The system uses several configurable parameters that can be adjusted in the respective modules:

- **Detection thresholds**: Confidence and NMS thresholds for object detection
- **Tracking parameters**: Maximum disappeared frames, distance thresholds
- **Anomaly detection**: Running speed thresholds, abandonment timers, crowd density limits

## 📊 Output

The system generates:
- **Processed videos**: Annotated videos with bounding boxes, object IDs, and tracking paths
- **JSON logs**: Detailed frame-by-frame detection and anomaly data
- **Text summaries**: Human-readable reports of detected anomalies

## 🔮 Future Improvements

This project provides a strong foundation. Future enhancements could include:

- **GPU Acceleration**: Modify the detector to use a CUDA backend for a massive performance boost in real-time processing
- **Advanced Tracking**: Replace the current centroid tracker with a more robust algorithm like DeepSORT to better handle object occlusions
- **Adaptive Thresholds**: Implement a calibration mode to allow the system to learn "normal" behavior in a scene and set anomaly thresholds dynamically
- **Web-Based UI**: Develop a web interface to view live streams, manage alerts, and review logged events

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The YOLO models (v3 and v8) for state-of-the-art object detection
- The creators of the Avenue Dataset for providing valuable data for anomaly detection research
- The OpenCV and NumPy teams for their essential open-source libraries

## 📞 Contact

**Karan Sehgal** - 22BCE3939  
*Vellore Institute of Technology, Vellore*

Project Link: [https://github.com/your-username/ai-surveillance-system](https://github.com/your-username/ai-surveillance-system)
