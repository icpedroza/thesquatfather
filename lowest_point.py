import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, smooth_landmarks=True)

# Function to calculate the vertical position of the hips.
def get_hip_y_position(landmarks, height):
    # Averaging the y-coordinates of the left and right hips for a central hip position
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_y = (left_hip.y + right_hip.y) / 2 * height
    return hip_y

# Load the video.
video_path = 'squat_video.mp4'
cap = cv2.VideoCapture(video_path)

frame_number = 0
lowest_point_frame_number = 0
max_hip_height = 0
frame_of_interest = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Get the y-coordinate of the hips.
        hip_y = get_hip_y_position(results.pose_landmarks, frame_height)
        
        # Update the lowest point tracking.
        if hip_y > max_hip_height:
            max_hip_height = hip_y
            lowest_point_frame_number = frame_number
            frame_of_interest = frame.copy()

    frame_number += 1

# Release the video capture.
cap.release()

# Display or save the frame at the lowest point.
if frame_of_interest is not None:
    # Display the frame.
    cv2.imshow('Lowest Point in Squat', frame_of_interest)
    cv2.waitKey(0)  # Wait for a key press to close.
    cv2.destroyAllWindows()

    # Optionally, save the frame to an image file.
    cv2.imwrite('lowest_point_squat_frame.jpg', frame_of_interest)

    print("The lowest point in the squat is at frame:", lowest_point_frame_number)
else:
    print("No valid frames found.")


