import cv2
import mediapipe as mp
import numpy as np

def detect_emotion(mouth_ratio, eyebrow_ratio):
    if mouth_ratio > 0.04 and eyebrow_ratio > 0.03:
        return "Smiling 😊"
    elif mouth_ratio < 0.02 and eyebrow_ratio < 0.02:
        return "Sad 😔"
    else:
        return "Neutral 😐"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            
            # Get mouth landmark points
            upper_lip = np.array([
                (int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height))
                for i in [61, 40, 37, 267, 0]
            ])
            lower_lip = np.array([
                (int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height))
                for i in [146, 91, 181, 405, 17]
            ])
            
            # Get eyebrow landmark points
            left_eyebrow = np.array([
                (int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height))
                for i in [70, 63, 105]
            ])
            right_eyebrow = np.array([
                (int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height))
                for i in [336, 296, 334]
            ])
            
            # Calculate mouth and eyebrow movement ratios
            mouth_height = np.linalg.norm(upper_lip[2] - lower_lip[2])
            mouth_width = np.linalg.norm(upper_lip[0] - upper_lip[-1])
            mouth_ratio = mouth_height / mouth_width
            
            eyebrow_height = np.linalg.norm(left_eyebrow[1] - right_eyebrow[1])
            eyebrow_width = np.linalg.norm(left_eyebrow[0] - left_eyebrow[2])
            eyebrow_ratio = eyebrow_height / eyebrow_width
            
            # Detect emotion
            emotion = detect_emotion(mouth_ratio, eyebrow_ratio)
            
            # Draw text on frame
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    
    # Display frame
    cv2.imshow("Facial Expression Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
