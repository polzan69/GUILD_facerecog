import face_recognition
import pickle
import cv2
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR, FACE_ENCODINGS_FILE, DETECTION_METHOD

from imutils import paths

class FaceEncoder:
    def __init__(self):
        self.dataset_path = DATASET_DIR
        self.encodings_file = FACE_ENCODINGS_FILE
        self.detection_method = DETECTION_METHOD
        
    def encode_faces(self):
        """
        Process the dataset directory and create face encodings
        """
        print("[INFO] Quantifying faces...")
        
        # Check if dataset directory exists and has subdirectories
        if not os.path.exists(self.dataset_path):
            print(f"[ERROR] Dataset directory {self.dataset_path} does not exist")
            return False
            
        # Get list of subdirectories (each should be a person's name)
        person_dirs = [d for d in os.listdir(self.dataset_path) 
                      if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if not person_dirs:
            print(f"[ERROR] No person directories found in {self.dataset_path}")
            print("Create a directory for each person with their images inside")
            return False
            
        # Initialize lists for encodings and names
        known_encodings = []
        known_names = []
        
        # Process each person directory
        for person_dir in person_dirs:
            person_path = os.path.join(self.dataset_path, person_dir)
            person_name = person_dir  # Directory name is the person's name
            
            # Get all image files in the person's directory
            image_files = [f for f in os.listdir(person_path) 
                          if os.path.isfile(os.path.join(person_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"[WARNING] No images found for {person_name}")
                continue
                
            print(f"[INFO] Processing {len(image_files)} images for {person_name}")
            
            # Process each image
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                
                # Load and convert image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"[WARNING] Could not read {image_path}")
                    continue
                    
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces in the image
                boxes = face_recognition.face_locations(rgb, model=self.detection_method)
                
                if not boxes:
                    print(f"[WARNING] No face found in {image_path}")
                    continue
                    
                # Compute face encodings
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                # Add encodings and names to lists
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)
        
        # Save encodings to file
        if known_encodings:
            print(f"[INFO] Serializing {len(known_encodings)} encodings...")
            data = {"encodings": known_encodings, "names": known_names}
            with open(self.encodings_file, "wb") as f:
                f.write(pickle.dumps(data))
            print(f"[INFO] Encodings saved to {self.encodings_file}")
            return True
        else:
            print("[ERROR] No faces were encoded")
            return False

# If run directly, encode faces
if __name__ == "__main__":
    encoder = FaceEncoder()
    encoder.encode_faces()