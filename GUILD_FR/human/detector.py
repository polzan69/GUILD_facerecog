import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_CONFIG, YOLO_WEIGHTS, CLASSES_FILE, MODEL_DIR, CONFIDENCE_THRESHOLD

class HumanDetector:
    def __init__(self):
        # Check if model files exist
        if not os.path.exists(YOLO_CONFIG) or not os.path.exists(YOLO_WEIGHTS):
            print(f"[ERROR] YOLO model files not found. Please download them to {MODEL_DIR}")
            print("You can download yolov4-tiny.weights from: https://github.com/AlexeyAB/darknet/releases")
            sys.exit(1)
            
        # Load YOLO network
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
        
        # Load classes
        with open(CLASSES_FILE, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.roi = None
        
    def detect_humans(self, frame):
        """
        Detect humans in the frame using YOLO
        Returns: (is_human_detected, processed_frame)
        """
        height, width, _ = frame.shape
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set the blob as input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD and self.classes[class_id] == "person":
                    # Object detected is a person with sufficient confidence
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
        
        # Apply non-maximum suppression to remove redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
        
        # Draw bounding boxes for humans
        human_detected = len(indexes) > 0
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"Person: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if human is in ROI if ROI is defined
        if human_detected and self.roi is not None:
            roi_x, roi_y, roi_w, roi_h = self.roi
            human_in_roi = False
            
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    # Check if the human box intersects with ROI
                    if (x < roi_x + roi_w and x + w > roi_x and 
                        y < roi_y + roi_h and y + h > roi_y):
                        human_in_roi = True
                        break
            
            # Draw ROI on frame
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
            
            return human_in_roi, frame
        
        return human_detected, frame
    
    def select_roi(self, frame):
        """
        Allow user to select a region of interest
        """
        print("Select a Region of Interest (ROI) and press ENTER when done")
        self.roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return self.roi