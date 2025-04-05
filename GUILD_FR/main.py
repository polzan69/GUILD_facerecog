import cv2
import os
import sys
import argparse
import time

# Import from config file
from config import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, ROI_SELECTION
)
from human.detector import HumanDetector
from face.recognizer import FaceRecognizer
from face.encoder import FaceEncoder

def parse_arguments():
    parser = argparse.ArgumentParser(description='Home Security System with Facial Recognition')
    parser.add_argument('--mode', type=str, default='run', choices=['run', 'encode'],
                        help='Mode to run: "run" for detection, "encode" for encoding faces')
    parser.add_argument('--camera', type=int, default=CAMERA_ID,
                        help='Camera ID to use')
    parser.add_argument('--no-roi', action='store_true',
                        help='Disable ROI selection')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # If in encode mode, run the face encoder and exit
    if args.mode == 'encode':
        encoder = FaceEncoder()
        success = encoder.encode_faces()
        sys.exit(0 if success else 1)
    
    # Initialize the human detector
    human_detector = HumanDetector()
    
    # Initialize the face recognizer
    face_recognizer = FaceRecognizer()
    
    # Open the video capture
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        sys.exit(1)
    
    print("[INFO] Starting video stream...")
    time.sleep(2.0)  # Allow camera to warm up
    
    # Select ROI if enabled
    roi_selected = False
    
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        
        # Select ROI if enabled and not already selected
        if ROI_SELECTION and not args.no_roi and not roi_selected:
            human_detector.select_roi(frame)
            roi_selected = True
        
        # Detect humans in the frame
        human_detected, frame = human_detector.detect_humans(frame)
        
        # If a human is detected, recognize faces
        if human_detected:
            frame, names = face_recognizer.recognize_faces(frame)
            
            # Display alert if unknown person detected
            if "Unknown" in names:
                cv2.putText(frame, "ALERT: Unknown Person Detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Home Security System", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()