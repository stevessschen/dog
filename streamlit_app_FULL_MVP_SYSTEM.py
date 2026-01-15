import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import pyttsx3
import tempfile
import torch
import time
import openai

# ============ CONFIG ============
st.set_page_config(page_title="DogTalk AI", layout="wide")
st.title("🐕 DogTalk AI — 即時狗狗情緒翻譯系統")

# ============ Load Models ============
from ultralytics import YOLO
import torch
import streamlit as st
import os

@st.cache_resource
def load_models():
    # 1️⃣ Dog Detection (COCO has dog class)
    dog_detector = YOLO("yolov8n.pt")   # auto download

    # 2️⃣ Dog Pose Model (YOLOv8 Pose)
    # 使用官方 pose 模型，之後你可以換成你自己訓練的狗狗專用 pose
    pose_model = YOLO("yolov8n-pose.pt")  # auto download

    # 3️⃣ Emotion model
    emotion_model_path = "models/emotionnet.pt"
    if not os.path.exists(emotion_model_path):
        st.warning("EmotionNet not found. Using demo model.")
        emotion_model = None
    else:
        emotion_model = torch.jit.load(emotion_model_path)

    # 4️⃣ Behavior model
    behavior_model_path = "models/behaviornet.pt"
    if not os.path.exists(behavior_model_path):
        st.warning("BehaviorNet not found. Using demo model.")
        behavior_model = None
    else:
        behavior_model = torch.jit.load(behavior_model_path)

    return dog_detector, pose_model, emotion_model, behavior_model


dog_detector, pose_model, emotion_model, behavior_model = load_models()

# ============ TTS ============
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.say(text)
    engine.runAndWait()

# ============ GPT Coach ============
def gpt_coach(emotion, behavior):
    prompt = f"""
A dog is showing {emotion} emotion and behavior is {behavior}.
Give short coaching advice for the owner.
"""

    return f"Your dog feels {emotion}. Suggested action: Give calm interaction and positive reinforcement."

# ============ Pose Analysis ============
def analyze_pose(kpts):
    xs = kpts[:, 0]
    ys = kpts[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    ratio = height / (width + 1e-5)

    if ratio > 1.3:
        return "standing"
    elif ratio < 0.7:
        return "lying"
    else:
        return "sitting"

# ============ Vision Pipeline ============
class DogVision(VideoTransformerBase):
    def __init__(self):
        self.last_message = "No dog detected yet."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = dog_detector(img, conf=0.4, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dog_crop = img[y1:y2, x1:x2]

                # Pose
                pose_results = pose_model.predict(dog_crop, conf=0.4, verbose=False)

                pose = "unknown"
                for pr in pose_results:
                    if pr.keypoints is None:
                        continue
                    kpts = pr.keypoints.xy.cpu().numpy()[0]
                    pose = analyze_pose(kpts)

                    # Draw skeleton
                    for x, y in kpts:
                        cv2.circle(img, (int(x + x1), int(y + y1)), 3, (0, 0, 255), -1)

                # Fake emotion + behavior (replace with real inference)
                emotion = np.random.choice(["relaxed", "excited", "alert", "nervous"])
                behavior = np.random.choice(["resting", "playing", "guarding", "running"])

                self.last_message = f"The dog is {pose} and feeling {emotion}"

                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, self.last_message, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                st.session_state["dog_message"] = self.last_message
                st.session_state["emotion"] = emotion
                st.session_state["behavior"] = behavior

        return img

# ============ UI ============
if "dog_message" not in st.session_state:
    st.session_state["dog_message"] = "No dog detected yet."

if "emotion" not in st.session_state:
    st.session_state["emotion"] = "unknown"

if "behavior" not in st.session_state:
    st.session_state["behavior"] = "unknown"

# ============ Layout ============
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📷 Live Camera")
    webrtc_streamer(key="dogcam", video_transformer_factory=DogVision, media_stream_constraints={"video": True, "audio": False})

with col2:
    st.subheader("🧠 Dog Interpretation")

    st.metric("Pose", st.session_state["dog_message"])
    st.metric("Emotion", st.session_state["emotion"])
    st.metric("Behavior", st.session_state["behavior"])

    if st.button("🔊 Speak Dog Emotion"):
        speak(st.session_state["dog_message"])

    if st.button("🤖 AI Coach Advice"):
        advice = gpt_coach(st.session_state["emotion"], st.session_state["behavior"])
        st.success(advice)
        speak(advice)

st.markdown("---")
st.caption("DogTalk AI © 2026 - Real-time Dog Emotion Translator")
