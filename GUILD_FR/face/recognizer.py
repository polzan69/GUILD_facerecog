import face_recognition
import pickle
import cv2
import os
import sys
import datetime

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FACE_ENCODINGS_FILE, DETECTION_METHOD, AUTHORIZED_DIR, UNAUTHORIZED_DIR

class FaceRecognizer:
    def __init__(self):
        self.encodings_file = FACE_ENCODINGS_FILE
        self.detection_method = DETECTION_METHOD
        self.authorized_dir = AUTHORIZED_DIR
        self.unauthorized_dir = UNAUTHORIZED_DIR
        
        # Load the known face encodings
        self.load_encodings()
        
    def load_encodings(self):
        """
        Load the known face encodings from the pickle file
        """
        if not os.path.exists(self.encodings_file):
            print(f"[ERROR] Encodings file {self.encodings_file} not found")
            print("Please run the face encoder first to create the encodings file")
            self.data = {"encodings": [], "names": []}
            return False
            
        print(f"[INFO] Loading encodings from {self.encodings_file}")
        with open(self.encodings_file, "rb") as f:
            self.data = pickle.loads(f.read())
        
        print(f"[INFO] Loaded {len(self.data['encodings'])} face encodings")
        return True
        
    def recognize_faces(self, frame):
        """
        Recognize faces in the frame
        Returns: (processed_frame, recognized_names)
        """
        # Convert the frame to RGB (face_recognition uses RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the frame
        boxes = face_recognition.face_locations(rgb, model=self.detection_method)
        
        # If no faces are detected, return the original frame
        if not boxes:
            return frame, []
            
        # Compute face encodings
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        names = []
        
        # Loop over the face encodings
        for (box, encoding) in zip(boxes, encodings):
            # Compare face encoding with known encodings
            matches = face_recognition.compare_faces(self.data["encodings"], encoding)
            name = "Unknown"
            
            # If there's a match
            if True in matches:
                # Find all matched indexes
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                # Count occurrences of each name
                for i in matched_idxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                # Get the name with the most votes
                name = max(counts, key=counts.get)
            
            # Add name to the list
            names.append(name)
            
            # Draw the name and box on the frame
            (top, right, bottom, left) = box
            
            # Determine color based on authorization
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            # Save face image if it's unknown
            if name == "Unknown":
                self.save_face(frame, box, "unauthorized")
            else:
                self.save_face(frame, box, "authorized", name)
        
        return frame, names
    
    def save_face(self, frame, box, status, name=None):
        """
        Save the detected face to the appropriate directory
        """
        (top, right, bottom, left) = box
        
        # Extract the face from the frame
        face = frame[top:bottom, left:right]
        
        # Create a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if status == "authorized" and name:
            # Save to authorized directory with person's name
            filename = f"{name}_{timestamp}.jpg"
            save_path = os.path.join(self.authorized_dir, filename)
        else:
            # Save to unauthorized directory
            filename = f"unknown_{timestamp}.jpg"
            save_path = os.path.join(self.unauthorized_dir, filename)
        
        # Save the face image
        cv2.imwrite(save_path, face)