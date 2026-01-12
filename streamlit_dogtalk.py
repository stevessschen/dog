import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import platform

st.set_page_config(page_title="DogTalk AI MVP", layout="wide")
st.title("🐶 DogTalk AI MVP")

# 判斷是否在 Cloud
IS_CLOUD = platform.system() == "Linux"

# 載入模型
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 語音輸出（Cloud 用播放器）
def speak(text):
    tts = gTTS(text=text, lang="zh-tw")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name)

# 分析圖片
def analyze(img):
    results = model.predict(img, classes=[16], verbose=False)

    if len(results[0].boxes) == 0:
        return img, "找不到狗狗", "請重新拍攝"

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    emotion = "放鬆"
    suggestion = "牠現在很放鬆，可以輕鬆互動"

    cv2.putText(img, f"情緒: {emotion}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(img, f"建議: {suggestion}", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    return img, emotion, suggestion


# UI
st.subheader("📷 上傳狗狗照片")
uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始圖片", use_column_width=True)

    img_np = np.array(image)
    result_img, emotion, suggestion = analyze(img_np)

    st.image(result_img, caption="AI 分析結果", use_column_width=True)
    st.success(f"情緒判斷：{emotion}")
    st.info(f"行為建議：{suggestion}")

    speak(suggestion)
