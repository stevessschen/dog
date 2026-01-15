import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np
import openai
import streamlit.components.v1 as components
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="DogTalk AI Pro",
    page_icon="🐕",
    layout="wide"
)

st.title("🐕 DogTalk AI Pro")
st.caption("Real-time Dog Behavior & Emotion AI Coach")

# -----------------------------
# Browser Voice Engine
# -----------------------------
components.html("""
<script>
window.dogtalkSpeak = function(text) {
    const msg = new SpeechSynthesisUtterance(text);
    msg.lang = "en-US";
    msg.rate = 1;
    msg.pitch = 1.1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
}
</script>
""", height=0)

def speak(text):
    components.html(f"""
    <script>window.dogtalkSpeak("{text}");</script>
    """, height=0)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo = load_yolo()

# -----------------------------
# AI MODELS (可換成真模型)
# -----------------------------
def posenet_estimate(dog_crop):
    return "sitting"   # standing / sitting / lying / running

def tailnet_predict(dog_crop):
    return "wagging"   # high / low / tucked / wagging

def emotionnet_predict(pose, tail):
    if tail == "wagging":
        return "excited 🤩"
    if pose == "lying":
        return "relaxed 😌"
    return "alert 👀"

def behaviordnet_predict(pose, emotion):
    if emotion == "excited 🤩":
        return "playing"
    if pose == "lying":
        return "resting"
    return "observing"

# -----------------------------
# GPT Coach
# -----------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

def gpt_coach(emotion, behavior):
    if not openai.api_key:
        return "Your dog seems happy. Keep engaging with calm play."

    prompt = f"""
    My dog is feeling {emotion} and currently {behavior}.
    Give me a short coaching suggestion.
    """

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return resp.choices[0].message.content

# -----------------------------
# Vision Pipeline
# -----------------------------
class DogVision(VideoTransformerBase):
    def __init__(self):
        self.status = "Waiting for dog..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = yolo.predict(img, conf=0.45, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                label = yolo.names[int(cls)]
                if label == "dog":
                    x1, y1, x2, y2 = map(int, box)
                    dog_crop = img[y1:y2, x1:x2]

                    pose = posenet_estimate(dog_crop)
                    tail = tailnet_predict(dog_crop)
                    emotion = emotionnet_predict(pose, tail)
                    behavior = behaviordnet_predict(pose, emotion)
                    coach = gpt_coach(emotion, behavior)

                    self.status = f"""
                    🐶 Pose: {pose}
                    🐕 Tail: {tail}
                    😃 Emotion: {emotion}
                    🎯 Behavior: {behavior}
                    💡 Coach: {coach}
                    """

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
                    cv2.putText(img, f"{emotion} | {behavior}",
                        (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,0),2)

        return img

# -----------------------------
# UI Layout
# -----------------------------
col1, col2 = st.columns([3,2])

with col2:
    st.subheader("📊 Dog Interpretation")
    status_box = st.empty()
    speak_btn = st.button("🔊 AI Coach Speak")

# -----------------------------
# Webcam
# -----------------------------
webrtc_ctx = webrtc_streamer(
    key="dogtalk-pro",
    video_transformer_factory=DogVision,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# -----------------------------
# Live Status
# -----------------------------
if webrtc_ctx.video_transformer:
    status = webrtc_ctx.video_transformer.status
else:
    status = "Starting camera..."

status_box.info(status)

if speak_btn:
    speak(status)

# -----------------------------
# Auto Refresh
# -----------------------------
time.sleep(0.5)
st.rerun()
