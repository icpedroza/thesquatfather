from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import time
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, enable_segmentation=True, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

@socketio.on('connect')
def handle_connect():
    emit('response', {'data': 'Connected'})

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        frame = base64.b64encode(frame).decode('utf-8')
        socketio.emit('frame', {'frame': frame})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(target=gen_frames)
    socketio.run(app, host='0.0.0.0', port=5000)
