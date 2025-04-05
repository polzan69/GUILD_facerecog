import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "dnn_model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
AUTHORIZED_DIR = os.path.join(OUTPUT_DIR, "authorized")
UNAUTHORIZED_DIR = os.path.join(OUTPUT_DIR, "unauthorized")

# Create directories if they don't exist
for directory in [MODEL_DIR, DATASET_DIR, OUTPUT_DIR, AUTHORIZED_DIR, UNAUTHORIZED_DIR]:
    os.makedirs(directory, exist_ok=True)

# YOLO model settings
YOLO_CONFIG = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.txt")

# Face recognition settings
FACE_ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
DETECTION_METHOD = "hog"  # Options: "hog" (faster) or "cnn" (more accurate)

# Camera settings
CAMERA_ID = 0  # Default camera (usually webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for human detection
ROI_SELECTION = True  # Whether to select a region of interest