# streamlit_dogtalk_gtts.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import tempfile
from playsound import playsound
import threading

st.set_page_config(page_title="DogTalk AI MVP", layout="wide")
st.title("🐶 DogTalk AI 即時互動 MVP (Cloud 兼容版)")

# 載入 YOLO 模型
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 語音播放函數（使用 gTTS）
def speak(text):
    def _play():
        tts = gTTS(text=text, lang='zh-tw')
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            playsound(fp.name)
    threading.Thread(target=_play).start()  # 非阻塞

# 本地 webcam 功能（Streamlit Cloud 無法直接 webcam）
use_webcam = st.checkbox("使用 webcam (本地測試)", value=False)

if use_webcam:
    st.warning("請在本地執行 Streamlit 以使用 webcam")
    cap = cv2.VideoCapture(0)
    placeholder = st.empty()
    run = st.checkbox("啟動即時分析", value=True)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("無法讀取 webcam")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, classes=[16], verbose=False)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        pose = "sit"  # 簡化示範
        emotion_map = {"sit": "放鬆", "stand": "警戒", "lay": "休息"}
        emotion = emotion_map.get(pose, "未知")

        suggestions = {
            "放鬆": "牠現在很放鬆，可以輕鬆互動",
            "警戒": "牠有點警戒，建議保持距離",
            "休息": "牠在休息，請不要打擾"
        }
        suggestion = suggestions.get(emotion, "觀察牠的動作")

        cv2.putText(frame, f"情緒: {emotion}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, f"建議: {suggestion}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        speak(suggestion)

    cap.release()
else:
    # 上傳圖片模式
    uploaded_file = st.file_uploader("上傳狗狗圖片", type=["jpg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="原始圖片", use_column_width=True)

        img_np = np.array(image)
        results = model.predict(img_np, classes=[16], verbose=False)

        if len(results[0].boxes) == 0:
            st.warning("找不到狗狗")
        else:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            pose = "sit"
            emotion_map = {"sit": "放鬆", "stand": "警戒", "lay": "休息"}
            emotion = emotion_map.get(pose, "未知")

            suggestions = {
                "放鬆": "牠現在很放鬆，可以輕鬆互動",
                "警戒": "牠有點警戒，建議保持距離",
                "休息": "牠在休息，請不要打擾"
            }
            suggestion = suggestions.get(emotion, "觀察牠的動作")

            st.image(img_np, caption="偵測結果", use_column_width=True)
            st.success(f"情緒: {emotion}")
            st.info(f"建議: {suggestion}")
            speak(suggestion)
