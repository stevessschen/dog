import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import streamlit.components.v1 as components

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="DogTalk AI MVP", layout="wide")
st.title("🐶 DogTalk AI — Real-Time Dog Communication System")

st.markdown("Point your camera at your dog. AI will analyze posture, emotion and speak.")

# --------------------------------------------------
# Load AI Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --------------------------------------------------
# Shared State
# --------------------------------------------------
if "last_message" not in st.session_state:
    st.session_state.last_message = "No dog detected yet."

# --------------------------------------------------
# AI Logic
# --------------------------------------------------
def classify_gesture(w, h):
    ratio = h / w
    if ratio > 1.2:
        return "standing"
    elif ratio < 0.7:
        return "lying"
    else:
        return "sitting"

def classify_emotion(area, gesture):
    if gesture == "lying":
        return "relaxed"
    if gesture == "sitting":
        return "attentive"
    if area > 120000:
        return "excited"
    return "curious"

# --------------------------------------------------
# Browser Voice Engine (Persistent)
# --------------------------------------------------
def init_speech_engine():
    components.html("""
    <script>
    window.dogtalkSpeak = function(text) {
        const msg = new SpeechSynthesisUtterance(text);
        msg.lang = "en-US";
        msg.rate = 1;
        msg.pitch = 1;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
    }
    </script>
    """, height=0)

def speak(text):
    components.html(f"""
    <script>
        if (window.dogtalkSpeak) {{
            window.dogtalkSpeak("{text}");
        }}
    </script>
    """, height=0)

# Initialize voice engine ONCE
init_speech_engine()

# --------------------------------------------------
# Sidebar UI (Control Panel)
# --------------------------------------------------
st.sidebar.title("🎛 DogTalk AI Control Panel")

st.sidebar.subheader("AI Interpretation")
st.sidebar.success(st.session_state.last_message)

if st.sidebar.button("🔊 Speak Dog Emotion"):
    speak(st.session_state.last_message)

st.sidebar.markdown("---")
st.sidebar.info("Allow camera access and point at your dog.")

# --------------------------------------------------
# Video Processor
# --------------------------------------------------
class DogVideoProcessor(VideoTransformerBase):
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

                    message = f"Your dog is {gesture} and feels {emotion}"
                    st.session_state.last_message = message

                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

                    # Label
                    cv2.putText(
                        img,
                        message,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2
                    )

        return img

# --------------------------------------------------
# Webcam (Main Panel)
# --------------------------------------------------
webrtc_streamer(
    key="dogtalk",
    video_transformer_factory=DogVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
