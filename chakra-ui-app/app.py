import os

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import base64
import logging
import pose_evaluation_stuff.pose_evaluation
from Sound.SoundTracksGeneration import textVoice

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debug mode
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.DEBUG,  # Set the logging level to DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
                    handlers=[
                        logging.StreamHandler(),  # Log to console
                        logging.FileHandler('app.log')  # Log to a file named app.log
                    ])
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, enable_segmentation=True, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Global variables to store the frame and recording status
frame = None
recording = False
cv2_webcam_image = None


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def gen_frames():
    global frame, recording, cv2_webcam_image
    logger.debug("Generating frames started")

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        cv2_webcam_image = frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        encoded_frame = base64.b64encode(frame).decode('utf-8')
        socketio.emit('frame', {'frame': encoded_frame})
        time.sleep(0.05)  # Adjust the frame rate as needed

    cap.release()


def start_recording():
    global frame, recording, cv2_webcam_image

    logger.debug("Recording started")

    min_angle = 180
    best_frame = None
    best_landmarks = None
    start_time = time.time()

    while time.time() - start_time < 10:
        if frame is None:
            continue

        # print("ahahahah")

        image = cv2_webcam_image
        # image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angle = calculate_angle(hip, knee, ankle)

            if angle < min_angle:
                min_angle = angle
                best_frame = image
                best_landmarks = results.pose_landmarks

    if best_frame is not None:
        print("Best frame found")
        best_frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(best_frame_rgb, best_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        best_frame_bgr = cv2.cvtColor(best_frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('deepest_squat.jpg', best_frame_bgr)

        # AI pose evaluation
        pose_evaluation_stuff.pose_evaluation.evaluate_pose(results)
    else:
        print("Best frame not found")




    #Assume the generated text by ChatGPT is 'output'(String)
    global output
    textVoice(output)


@socketio.on('connect')
def handle_connect():
    logger.debug("Client connected")
    emit('response', {'data': 'Connected'})
    thread = threading.Thread(target=gen_frames)
    thread.start()


@socketio.on('start_recording')
def handle_start_recording():
    logger.debug("Recording emit received")
    app.logger.warning("recording emit received")
    thread = threading.Thread(target=start_recording)
    thread.start()


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
