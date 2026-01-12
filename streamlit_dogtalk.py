# streamlit_dogtalk.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pyttsx3

st.set_page_config(page_title="DogTalk AI Interactive MVP", layout="wide")
st.title("🐶 DogTalk AI 即時互動 MVP")

# 初始化 TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 語速
engine.setProperty('volume', 1.0)

# 載入 YOLO 模型
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

# 建立 webcam 捕捉
st.info("本地測試用，請允許 webcam 權限")
cap = cv2.VideoCapture(0)

run = st.checkbox("啟動即時分析", value=True)
placeholder = st.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("無法讀取 webcam")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # YOLO 偵測狗狗
    results = model.predict(frame_rgb, classes=[16], verbose=False)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 假設姿態/情緒
    pose = "sit"
    emotion_map = {"sit": "放鬆", "stand": "警戒", "lay": "休息"}
    emotion = emotion_map.get(pose, "未知")
    
    # GPT 建議
    suggestions = {
        "放鬆": "牠現在很放鬆，可以輕鬆互動",
        "警戒": "牠有點警戒，建議保持距離",
        "休息": "牠在休息，請不要打擾"
    }
    suggestion = suggestions.get(emotion, "觀察牠的動作")

    # 框線+文字疊加
    cv2.putText(frame, f"情緒: {emotion}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, f"建議: {suggestion}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 顯示畫面
    placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 播放語音建議
    engine.say(suggestion)
    engine.runAndWait()

cap.release()
