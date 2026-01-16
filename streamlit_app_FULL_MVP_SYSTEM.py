import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np
import streamlit.components.v1 as components

# -------------------------------
# Page Config (Mobile friendly)
# -------------------------------
st.set_page_config(
    page_title="DogTalk AI",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body { background-color: #0f172a; color: white; }
.big-title { font-size: 40px; font-weight: bold; text-align: center; }
.subtitle { text-align: center; font-size: 18px; }
.footer { text-align: center; opacity: 0.6; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🐶 DogTalk AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time Dog Emotion & Gesture Translator</div>", unsafe_allow_html=True)

# -------------------------------
# Browser TTS Engine
# -------------------------------
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
    <script>
        window.dogtalkSpeak("{text}");
    </script>
    """, height=0)

# -------------------------------
# Load YOLO (auto-downloads)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------
# AI Logic
# -------------------------------
def classify_gesture(w, h):
    ratio = h / w
    if ratio > 1.25:
        return "standing"
    elif ratio < 0.7:
        return "lying"
    else:
        return "sitting"

def classify_emotion(area, gesture):
    if gesture == "lying":
        return "relaxed 😌"
    if gesture == "sitting":
        return "attentive 👀"
    if area > 130000:
        return "excited 🤩"
    return "curious 🐾"

# -------------------------------
# Webcam Processor (Thread Safe)
# -------------------------------
class DogVision(VideoTransformerBase):
    def __init__(self):
        self.last_message = "No dog detected yet."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.45, verbose=False)

        detected = False

        for r in results:
            if r.boxes is None:
                continue

            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                label = model.names[int(cls)]
                if label == "dog":
                    detected = True
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    area = w * h

                    gesture = classify_gesture(w, h)
                    emotion = classify_emotion(area, gesture)

                    msg = f"The dog is {gesture} and feeling {emotion}"
                    self.last_message = msg

                    # Draw overlay
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
                    cv2.putText(
                        img, msg, (x1, y1-12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2
                    )

        if not detected:
            self.last_message = "No dog detected yet."

        return img

# -------------------------------
# Webcam
# -------------------------------
webrtc_ctx = webrtc_streamer(
    key="dogtalk-ai",
    video_transformer_factory=DogVision,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# -------------------------------
# UI Layout
# -------------------------------
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("🎯 Dog Interpretation")

    placeholder = st.empty()

    #st.markdown("---")

    speak_btn = st.button("🔊 Speak Dog Emotion", use_container_width=True)

    #st.markdown("---")
    st.markdown("### 📱 Mobile Tips")
    st.markdown("• Allow camera access\n• Tap Speak button\n• Use landscape mode")

# -------------------------------
# Live Sidebar Sync
# -------------------------------
if webrtc_ctx.video_transformer:
    current_message = webrtc_ctx.video_transformer.last_message
else:
    current_message = "Starting camera..."

placeholder.info(current_message)

# -------------------------------
# Speak Button
# -------------------------------
if speak_btn:
    speak(current_message)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<div class='footer'>DogTalk AI © 2026 — MVP Prototype</div>", unsafe_allow_html=True)
