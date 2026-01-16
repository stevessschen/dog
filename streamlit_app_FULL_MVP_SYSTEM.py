import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import streamlit.components.v1 as components

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Dog Communicator 2.0",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body { background-color: #0f172a; color: white; }
.big-title { font-size: 38px; font-weight: bold; text-align: center; }
.subtitle { text-align: center; font-size: 18px; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🐶 Dog Communicator 2.0</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time Canine Emotion & Behavior AI</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Browser TTS
# --------------------------------------------------
components.html("""
<script>
window.speakDog = function(text) {
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  u.rate = 1;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}
</script>
""", height=0)

def speak(text):
    components.html(f"<script>speakDog(`{text}`)</script>", height=0)

# --------------------------------------------------
# Load Models (Auto Download YOLO)
# --------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    models["yolo"] = YOLO("yolov8n.pt")

    # Placeholder torch models (future replacement)
    models["pose"] = None
    models["tail"] = None
    models["emotion"] = None
    models["behavior"] = None
    return models

models = load_models()

# --------------------------------------------------
# AI Inference Pipeline (REAL STRUCTURE)
# --------------------------------------------------
def infer_pose(crop):
    # Placeholder PoseNet
    return np.random.choice(["standing", "sitting", "lying", "running"])

def infer_tail(crop):
    return np.random.choice(["high", "low", "tucked", "wagging"])

def infer_emotion(pose, tail):
    if pose == "lying":
        return "relaxed"
    if tail == "tucked":
        return "nervous"
    if tail == "wagging":
        return "excited"
    return "alert"

def infer_behavior(pose, emotion):
    if emotion == "excited":
        return "playful"
    if emotion == "nervous":
        return "avoidance"
    return "neutral"

def gpt_explain(pose, tail, emotion, behavior):
    return (
        f"The dog is {pose} with a {tail} tail. "
        f"This suggests the dog feels {emotion}. "
        f"Current behavior is classified as {behavior}. "
        f"Recommendation: stay calm and reinforce positive interaction."
    )

# --------------------------------------------------
# Webcam Processor
# --------------------------------------------------
class DogVision(VideoTransformerBase):
    def __init__(self):
        self.last_result = "No dog detected."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = models["yolo"](img, conf=0.4, verbose=False)

        detected = False

        for r in results:
            if r.boxes is None:
                continue

            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if models["yolo"].names[int(cls)] != "dog":
                    continue

                detected = True
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]

                pose = infer_pose(crop)
                tail = infer_tail(crop)
                emotion = infer_emotion(pose, tail)
                behavior = infer_behavior(pose, emotion)

                explanation = gpt_explain(pose, tail, emotion, behavior)
                self.last_result = explanation

                cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(
                    img,
                    f"{pose} | {emotion}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

        if not detected:
            self.last_result = "No dog detected."

        return img

# --------------------------------------------------
# Webcam Stream
# --------------------------------------------------
ctx = webrtc_streamer(
    key="dog-ai",
    video_processor_factory=DogVision,
    media_stream_constraints={
        "video": {"facingMode": "environment"},
        "audio": False
    },
    async_processing=True
)

# --------------------------------------------------
# UI
# --------------------------------------------------
col1, col2 = st.columns([3,1])

with col2:
    st.subheader("🧠 AI Interpretation")
    output_box = st.empty()
    speak_btn = st.button("🔊 Speak")

if ctx.video_transformer:
    output_box.info(ctx.video_transformer.last_result)

if speak_btn and ctx.video_transformer:
    speak(ctx.video_transformer.last_result)

st.markdown("<div style='opacity:0.5;text-align:center'>Dog Communicator 2.0 © 2026</div>", unsafe_allow_html=True)
