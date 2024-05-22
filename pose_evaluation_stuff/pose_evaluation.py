import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle
import numpy

# Load model
# Doing this since import is from app, in another folder
MODEL_PATH = r'pose_evaluation_stuff\pose_lsvc_acc84.model'
SCALER_PATH = r'pose_evaluation_stuff\pose_lsvc_acc84.scaler'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

categories = {
    0: "back bowled",
    1: "back too low",
    2: "correct",
    3: "knees inward",
    4: "knees too far forward",
    5: "legs not far",
    6: "legs too far",
    7: "not deep enough",
    8: "too deep enough"
}


def preprocess_pose(df):
    # Normalize data
    pose_mean = df.stack().mean()
    pose_std = df.stack().std()
    df = (df - pose_mean) / pose_std
    return df


# Model predict function, includes reshape
def predict(input_pose):
    # Reshape
    input_pose = np.array([input_pose])

    # reshape into 2d
    nsamples, nx, ny = input_pose.shape
    input_pose_reshaped = input_pose.reshape((nsamples, nx * ny))

    # input_pose_reshaped = input_pose

    # scaling
    # scaler = StandardScaler()
    # s = np.sqrt(scaler.var_)
    # m = scaler.mean_

    print(input_pose_reshaped)
    # print(m)
    # input_pose_scaled = (np.array([input_pose_reshaped] - m)) / s
    input_pose_scaled = scaler.transform(input_pose_reshaped.astype(np.float32))

    prediction = model.predict(input_pose_scaled)

    return prediction


def convert_to_df(input_pose):
    new_pose = []
    landmarks = input_pose.pose_world_landmarks.landmark
    for point in landmarks:
        new_pose.append(numpy.array([point.x, point.y, point.z]))

    return pd.DataFrame(new_pose)


def convert_and_predict(input_pose):
    return predict(preprocess_pose(convert_to_df(input_pose)))


def evaluate_pose(results):
    # AI now, eepy!!!1!!11!
    # but first data restructure
    detected = False
    try:
        _ = results.pose_world_landmarks.landmark
        prediction_output = convert_and_predict(results)
        print(prediction_output)
        print(f"{categories[prediction_output[0]]}")
        return True, prediction_output[0]
    except KeyError:
        return False, ""
