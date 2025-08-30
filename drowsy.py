import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import time

class DrowsinessDetector:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        # Download shape_predictor_68_face_landmarks.dat from dlib's website
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Eye aspect ratio (EAR) threshold and frame counters
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.COUNTER = 0
        self.ALARM_ON = False
        
        # Initialize pygame for alarm sound
        pygame.mixer.init()
        
        # Eye landmark indices (based on dlib's 68-point model)
        self.LEFT_EYE_START = 42
        self.LEFT_EYE_END = 48
        self.RIGHT_EYE_START = 36
        self.RIGHT_EYE_END = 42
    
    def eye_aspect_ratio(self, eye):
        """Calculate the eye aspect ratio (EAR)"""
        # Compute euclidean distances between vertical eye landmarks
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        C = distance.euclidean(eye[0], eye[3])
        
        # Compute eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, landmarks, eye_start, eye_end):
        """Extract eye landmarks from facial landmarks"""
        eye_points = []
        for i in range(eye_start, eye_end):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            eye_points.append((x, y))
        return np.array(eye_points)
    
    def play_alarm(self):
        """Play alarm sound (you can replace this with actual sound file)"""
        if not self.ALARM_ON:
            self.ALARM_ON = True
            # Create a simple beep sound
            frequency = 800
            duration = 1000
            sample_rate = 22050
            frames = int(duration * sample_rate / 1000)
            arr = np.zeros(frames)
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            arr = (arr * 32767).astype(np.int16)
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            sound = pygame.sndarray.make_sound(stereo_arr)
            sound.play()
    
    def stop_alarm(self):
        """Stop the alarm"""
        if self.ALARM_ON:
            pygame.mixer.stop()
            self.ALARM_ON = False
    
    def detect_drowsiness(self):
        """Main function to detect drowsiness"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting drowsiness detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            for face in faces:
                # Get facial landmarks
                landmarks = self.predictor(gray, face)
                
                # Extract left and right eye coordinates
                left_eye = self.extract_eye_landmarks(landmarks, self.LEFT_EYE_START, self.LEFT_EYE_END)
                right_eye = self.extract_eye_landmarks(landmarks, self.RIGHT_EYE_START, self.RIGHT_EYE_END)
                
                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                
                # Average EAR for both eyes
                ear = (left_ear + right_ear) / 2.0
                
                # Draw eye contours
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                
                # Check if EAR is below threshold
                if ear < self.EAR_THRESHOLD:
                    self.COUNTER += 1
                    
                    # If eyes closed for sufficient frames, trigger alarm
                    if self.COUNTER >= self.EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.play_alarm()
                else:
                    self.COUNTER = 0
                    self.stop_alarm()
                
                # Display EAR value
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Display counter
                cv2.putText(frame, f"Blink Counter: {self.COUNTER}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw face rectangle
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display status
            status = "AWAKE" if not self.ALARM_ON else "DROWSY"
            color = (0, 255, 0) if not self.ALARM_ON else (0, 0, 255)
            cv2.putText(frame, f"Status: {status}", (10, frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Drowsiness Detection", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def main():
    """Main function"""
    detector = DrowsinessDetector()
    
    try:
        detector.detect_drowsiness()
    except FileNotFoundError:
        print("Error: shape_predictor_68_face_landmarks.dat not found!")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place it in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()