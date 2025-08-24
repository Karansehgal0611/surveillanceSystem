import os
import urllib.request
import requests

def download_yolo_files():
    """Download YOLO model files from alternative sources"""
    models_dir = "Avenue Dataset/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Alternative download sources
    files_to_download = {
        'yolov3.weights': [
            'https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true',
            'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
        ],
        'yolov3.cfg': [
            'https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true',
            'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
        ],
        'coco.names': [
            'https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true',
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        ]
    }
    
    # Special handling for weights file (large file)
    weights_urls = [
        'https://pjreddie.com/media/files/yolov3.weights',
        'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights'
    ]
    
    print("Downloading YOLO model files...")
    
    # Download weights file first (special handling)
    weights_path = os.path.join(models_dir, 'yolov3.weights')
    if not os.path.exists(weights_path):
        print("Downloading yolov3.weights (this may take a while)...")
        success = False
        
        for url in weights_urls:
            try:
                urllib.request.urlretrieve(url, weights_path)
                file_size = os.path.getsize(weights_path)
                if file_size > 100000000:  # Should be around 248MB
                    print(f"Downloaded yolov3.weights ({file_size/1000000:.1f} MB)")
                    success = True
                    break
                else:
                    os.remove(weights_path)
                    print(f"Download failed, file too small: {file_size} bytes")
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
        
        if not success:
            print("Failed to download yolov3.weights. Please download it manually:")
            print("https://pjreddie.com/media/files/yolov3.weights")
            print("And place it in Avenue Dataset/models/")
    else:
        print("yolov3.weights already exists")
    
    # Download other files
    for filename, urls in files_to_download.items():
        if filename == 'yolov3.weights':
            continue  # Already handled above
            
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            success = False
            
            for url in urls:
                try:
                    if 'raw=true' in url or 'raw.githubusercontent.com' in url:
                        response = requests.get(url)
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                    else:
                        urllib.request.urlretrieve(url, filepath)
                    
                    print(f"Downloaded {filename}")
                    success = True
                    break
                except Exception as e:
                    print(f"Error downloading from {url}: {e}")
            
            if not success:
                print(f"Failed to download {filename}")
        else:
            print(f"{filename} already exists")
    
    print("Download process completed!")

if __name__ == "__main__":
    download_yolo_files()