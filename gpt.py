import cv2
import mediapipe as mp
import numpy as np
import openai
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import math

def calculate_vector(p1, p2):
    return [p2[i] - p1[i] for i in range(3)]

def dot_product(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(vi ** 2 for vi in v))

def angle_between_vectors(v1, v2):
    dot_prod = dot_product(v1, v2)
    mag_v1 = magnitude(v1)
    mag_v2 = magnitude(v2)
    if mag_v1 == 0 or mag_v2 == 0:
        return None  # Handling zero magnitude vectors
    cos_angle = dot_prod / (mag_v1 * mag_v2)
    angle_radians = math.acos(cos_angle)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def calculate_angle(p1, p2, p3):
    vector1 = calculate_vector(p1, p2)
    vector2 = calculate_vector(p2, p3)
    angle = 180 - angle_between_vectors(vector1, vector2)
    return angle

'''
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
'''
#cap = cv2.VideoCapture("C:/Users/zhang/Documents/WeChat Files/wxid_i3ekuzo7351q22/FileStorage/Video/2024-05/1447e7c2f5fb9b90fc2f317bd0675282.mp4")
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].z]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].z]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]

            
            # Calculate angle
            left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            left_ankle_angle = 180 - calculate_angle(left_foot_index, left_ankle, left_knee)
            right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            right_ankle_angle = 180 - calculate_angle(right_foot_index, right_ankle, right_knee)
            #print(landmarks)
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print(left_ankle, left_hip, left_knee, left_shoulder, right_ankle, right_hip, right_knee, right_shoulder)
print(left_ankle_angle, right_ankle_angle, left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle)

openai.api_key = "KEY_HERE"
completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a fitness expert. You should talk to me as if you are my in-person instructor. Respond with only three sentences."},
        {"role": "user", "content": "Next I'll input some angles of joints when i'm at the lowest position doing squats."},
        {"role": "assistant", "content": "Sure, please feel free to do that. I can tell you how you should adjust your actions."},
        {"role": "user", "content": f"the angle of the left hip is {left_hip_angle}, the angle of the right hip is {right_hip_angle}, the angle of the left knee is {left_knee_angle}, the angle of the right knee is {right_knee_angle}, the angle of the left ankle is {left_ankle_angle}, the angle of the right ankle is {right_ankle_angle}"}
    ]
)

print(completion.choices[0].message.content)
