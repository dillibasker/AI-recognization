import cv2
import mediapipe as mp
import numpy as np

def detect_movement(hip_y, knee_y, ankle_y):
    if abs(hip_y - knee_y) < 50 and abs(knee_y - ankle_y) < 50:
        return "Standing"
    elif hip_y > knee_y > ankle_y:
        return "Walking"
    elif ankle_y > knee_y > hip_y:
        return "Slipped/Fallen!"
    else:
        return "Unknown Movement"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        height, width, _ = frame.shape
        
        # Extract key body points
        landmarks = results.pose_landmarks.landmark
        hip_y = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height)
        knee_y = int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * height)
        ankle_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * height)
        
        # Detect movement
        movement = detect_movement(hip_y, knee_y, ankle_y)
        
        # Display movement status
        cv2.putText(frame, movement, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display frame
    cv2.imshow("Movement Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
