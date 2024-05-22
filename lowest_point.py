import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, enable_segmentation=True, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Start webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
min_angle = 180  # Impossible angle in practice, used to find minimum
best_frame = None
best_landmarks = None

while time.time() - start_time < 10:  # Record for 5 seconds
    success, frame = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the image
    results = pose.process(image)
    image.flags.writeable = True

    # Recolor back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get necessary landmarks for knee angle calculation (hip, knee, ankle)
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate the angle
        angle = calculate_angle(hip, knee, ankle)
        
        # Check if this is the smallest angle so far
        if angle < min_angle:
            min_angle = angle
            best_frame = frame  # Save the frame with the lowest knee angle
            best_landmarks = results.pose_landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
            )  

    cv2.imshow('Mediapipe Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



# Save the image of the lowest point in the squat
if best_frame is not None:
    # Convert the best frame to RGB
    best_frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    
    # Draw the landmarks on the best frame
    mp_drawing.draw_landmarks(
        best_frame_rgb, 
        best_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    # Convert back to BGR for saving
    best_frame_bgr = cv2.cvtColor(best_frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('deepest_squat.jpg', best_frame_bgr)

# Release resources
cap.release()
cv2.destroyAllWindows()
