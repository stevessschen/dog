import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="DogTalk AI MVP", layout="centered")
st.title("🐶 DogTalk AI — Real-Time Dog Communication System")

st.markdown("Live camera + AI emotion + gesture recognition + voice interpretation")

# -----------------------
# Load AI Model
# -----------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------
# AI Logic (MVP Heuristics)
# -----------------------

def classify_emotion(box_area, posture):
    if posture == "lying":
        return "Relaxed"
    if posture == "sitting":
        return "Attentive"
    if posture == "standing" and box_area > 120000:
        return "Excited"
    return "Curious"

def classify_gesture(w, h):
    ratio = h / w

    if ratio > 1.2:
        return "standing"
    elif ratio < 0.7:
        return "lying"
    else:
        return "sitting"

def generate_interpretation(emotion, gesture):
    return f"Your dog is {gesture} and feels {emotion}"

# -----------------------
# Voice Output
# -----------------------

def speak(text):
    tts = gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# -----------------------
# WebRTC Video Processor
# -----------------------

class DogVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_message = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, conf=0.4, verbose=False)

        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                label = model.names[int(cls)]

                if label == "dog":
                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h

                    gesture = classify_gesture(w, h)
                    emotion = classify_emotion(area, gesture)
                    interpretation = generate_interpretation(emotion, gesture)

                    self.last_message = interpretation

                    # Draw box
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

                    # Label
                    cv2.putText(img, interpretation, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return img

# -----------------------
# UI
# -----------------------

ctx = webrtc_streamer(
    key="dogtalk",
    video_transformer_factory=DogVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.subheader("🧠 AI Interpretation")

if ctx.video_transformer:
    message = ctx.video_transformer.last_message

    if message:
        st.success(message)

        if st.button("🔊 Let DogTalk AI Speak"):
            audio_file = speak(message)
            st.audio(audio_file)
            os.remove(audio_file)

else:
    st.info("Point your camera at your dog to start analysis.")
