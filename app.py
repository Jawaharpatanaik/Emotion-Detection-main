import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# UI title and description
st.title("😎 Emotion Recognition from Live Webcam")
st.write("This app identifies your emotions in real-time using a webcam and deep learning model.")

# Load model and labels
emotion_model = load_model("emotion_model.h5")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Custom video frame processor
class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            face_img = gray_frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)

            preds = emotion_model.predict(face_img, verbose=0)
            emotion_label = emotions[np.argmax(preds)]

            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame_bgr, emotion_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame_bgr

# Start the webcam stream
webrtc_streamer(key="emotion-detector-v2", video_processor_factory=EmotionProcessor)
