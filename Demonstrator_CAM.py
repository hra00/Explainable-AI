import numpy as np
import tensorflow as tf
import cv2
from collections import deque

import ModelGenerator

import Visualizer



""" Settings """
IMG_SHAPE = (100, 100, 3)
MODEL_NAME = 'FERplus-impr-std_0124-1040_weights.h5'
LAST_CONV_NAME = 'block3_conv3'

EMOTION_LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# offsets for face detection
USE_OFFSET = False
OFFSET_H_W = (10, 10)

N_SMOOTING_FRAMES = 10



""" Preparation """
# load face detection
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# font for text annotations
font = cv2.FONT_HERSHEY_SIMPLEX

# load emotion recognition model
model = ModelGenerator.get_base_model2(IMG_SHAPE)
model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))
model.load_weights('./models/' + MODEL_NAME)

# helper functions
def preprocess(x):
    mean = [129.4432, 129.4432, 129.4432]
    std = [64.87448751, 64.87448751, 64.87448751]
    # ensure image format
    x = np.array(x, dtype='float32')
    # normalize
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2] 
    return x

def vec2label(onehot_vec):
    major_vote = np.argmax(onehot_vec)
    return EMOTION_LABELS[major_vote]

# load visualizer
vis = Visualizer.Visualizer(model, IMG_SHAPE, LAST_CONV_NAME, preprocess)



""" Main run """
# run video for classification
path = './data/vid.mp4'
cap = cv2.VideoCapture(path)

""" TODO: 
- smoothing over face bboxes and predictions over k number of frames
- apply heatmap with values below zero cut out
    - np.max(0, heatmap)
    - apply to red channel np.uint8(*255)
"""
preds_queue = deque()
prev_bbox = None

while cap.isOpened():
    
    # read frame
    ret, frame = cap.read()

    # check for input
    if not ret:
        break

    # convert to gray for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(200, 200))
    
    # get coordinates and width, height (TODO: Check for face)
    if faces.size == 0:
        cv2.imshow("Live Output", frame)
    else:
        # TODO: smoothing over bounding boxes
        (x, y, w, h) = faces[0]

        # get crop of face
        if USE_OFFSET:        
            face = frame[
                y + int(OFFSET_H_W[0]/2.)   :   y + h - int(OFFSET_H_W[0]/2.),
                x + int(OFFSET_H_W[1]/2.)   :   x + w - int(OFFSET_H_W[1]/2.) ]
        else:
            face = frame[y:y + h, x:x + w]
        
        # get prediction and heatmaps
        heatmap, preds = vis.get_CAM(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        
        # compute smoothed predictions
        if len(preds_queue) >= N_SMOOTING_FRAMES: preds_queue.popleft()
        preds_queue.append(preds[0])
        avg_pred = np.mean(preds_queue, axis=0)
        
        # create output frame
        output_frame = frame.copy()

        # overlay bounding box of face detector
        output_frame = cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # add emotion label
        txt = f'{np.max(avg_pred)*100:.2f}% {vec2label(avg_pred)}'
        cv2.putText(output_frame, txt, (x, y-20), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # add heatmap (resize it first to match face frame)
        h_face, w_face = face.shape[:2]
        heatmap = cv2.resize(heatmap, (w_face, h_face))
        
        # prepare heatmap
        heatmap = (heatmap - np.min(heatmap) ) / ( np.max(heatmap) - np.min(heatmap) )
        heatmap = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET)
        output_frame[y:y+h_face, x:x+w_face, :] = cv2.addWeighted(
            output_frame[y:y+h_face, x:x+w_face, :], 0.6,
            heatmap, 0.4, 0)
        
        # show output frame
        output_frame = cv2.resize(output_frame, (output_frame.shape[1]//2, output_frame.shape[0]//2))
        cv2.imshow("Live Output", output_frame)
    
    # check for quitting
    key = cv2.waitKey(1)
    if key == 27:   # ESC
        break


# END
cv2.destroyAllWindows()
cap.release()