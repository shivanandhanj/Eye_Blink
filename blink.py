# ====================================
# Improved Eye Blink Detection System
# Features:
# - Blink Duration Analysis
# - Blink Frequency (Blinks per Minute)
# - Multi-face Detection
# ====================================

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)

# Eye landmark indices (from Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# Parameters
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 2
DROWSY_TIME = 2.0   # seconds eyes must remain closed to trigger drowsy alert

# Blink & time tracking per face
face_data = {}  # Dictionary: {face_id: {blink_count, frame_counter, blink_times, closed_start}}

# Start video capture
cap = cv2.VideoCapture(0)
start_time = time.time()

print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_id, landmarks in enumerate(results.multi_face_landmarks):
            if face_id not in face_data:
                face_data[face_id] = {
                    "blink_count": 0,
                    "frame_counter": 0,
                    "blink_times": [],
                    "closed_start": None
                }

            left_ear = eye_aspect_ratio(landmarks.landmark, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                face_data[face_id]["frame_counter"] += 1
                if face_data[face_id]["closed_start"] is None:
                    face_data[face_id]["closed_start"] = time.time()
                else:
                    closed_duration = time.time() - face_data[face_id]["closed_start"]
                    if closed_duration >= DROWSY_TIME:
                        cv2.putText(frame, "DROWSY!", (30, 150 + face_id * 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                if face_data[face_id]["frame_counter"] >= CLOSED_FRAMES:
                    face_data[face_id]["blink_count"] += 1
                    face_data[face_id]["blink_times"].append(time.time())
                face_data[face_id]["frame_counter"] = 0
                face_data[face_id]["closed_start"] = None

            # Blink frequency (blinks/min)
            elapsed_time = (time.time() - start_time) / 60.0
            if elapsed_time > 0:
                blink_freq = face_data[face_id]["blink_count"] / elapsed_time
            else:
                blink_freq = 0

            # Display info on frame
            y_offset = 50 + face_id * 100
            cv2.putText(frame, f"Face {face_id+1}", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {face_data[face_id]['blink_count']}", (30, y_offset + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Blink/min: {blink_freq:.1f}", (30, y_offset + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Improved Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
