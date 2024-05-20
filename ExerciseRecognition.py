import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def count_pull_ups(image, landmarks, counter, stage):
    mp_pose = mp.solutions.pose
    # Get coordinates
    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    
    # Calculate angles
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_index)
    right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)

    # Visualize angle
    cv2.putText(image, str(left_elbow_angle), 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
    cv2.putText(image, str(left_wrist_angle), tuple(np.multiply(left_wrist, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(left_shoulder_angle), tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(left_hip_angle), tuple(np.multiply(left_hip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(left_knee_angle), tuple(np.multiply(left_knee, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(left_ankle_angle), tuple(np.multiply(left_ankle, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(image, str(right_elbow_angle), 
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
    cv2.putText(image, str(right_wrist_angle), tuple(np.multiply(right_wrist, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(right_shoulder_angle), tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(right_hip_angle), tuple(np.multiply(right_hip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(right_knee_angle), tuple(np.multiply(right_knee, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(right_ankle_angle), tuple(np.multiply(right_ankle, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Curl counter logic
    if left_shoulder_angle > 150 and left_elbow_angle > 150 and right_shoulder_angle > 150 and right_elbow_angle > 150:
        stage = "down"
    if left_shoulder_angle < 30 and left_elbow_angle < 40 and right_shoulder_angle < 30 and right_elbow_angle < 40 and stage == 'down':
        stage = "up"
        counter += 1
        print(counter)
        
    return counter, stage

def count_curls(image, landmarks, counter, stage):
    mp_pose = mp.solutions.pose
    # Get coordinates
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    # Calculate angle
    angle = calculate_angle(shoulder, elbow, wrist)

    # Visualize angle
    cv2.putText(image, str(angle), 
                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
    
    # Curl counter logic
    if angle > 160:
        stage = "down"
    if angle < 30 and stage == 'down':
        stage = "up"
        counter += 1
        print(counter)
        
    return counter, stage