import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

st.set_page_config(page_title="DogTalk AI MVP", layout="centered")
st.title("🐶 DogTalk AI — Real-Time Dog Communication")

st.markdown("AI analyzes your dog's posture and behavior in real time.")

# Load YOLO model (small + fast)
model = YOLO("yolov8n.pt")

# Simple dog emotion logic (MVP)
def analyze_dog_state(box_area):
    if box_area > 120000:
        return "🐕 Dog is very close — seeking attention"
    elif box_area > 50000:
        return "🐶 Dog is calm and observing"
    else:
        return "🐾 Dog is far — maybe curious or cautious"


class DogVideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(img, conf=0.4, verbose=False)

        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                label = model.names[int(cls)]

                if label == "dog":
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)

                    emotion = analyze_dog_state(area)

                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, emotion, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return img


webrtc_streamer(
    key="dogtalk",
    video_transformer_factory=DogVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.markdown("### 🧠 DogTalk AI Interpretation")
st.info("Point the camera at your dog and observe its behavior in real-time.")
