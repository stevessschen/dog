import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import platform

st.set_page_config(page_title="DogTalk AI MVP", layout="wide")
st.title("🐶 DogTalk AI MVP (Cloud + 本地 webcam)")

# 判斷是否 Cloud 環境
IS_CLOUD = platform.system() == "Linux" and "KERNEL" in platform.uname().version

# 載入 YOLO 模型
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

# 語音播放 (Cloud 用 st.audio)
def speak(text):
    tts = gTTS(text=text, lang="zh-tw")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        if IS_CLOUD:
            st.audio(fp.name)
        else:
            # 本地測試可用 pyttsx3 或 gTTS 播放
            import os
            os.system(f"mpg123 {fp.name} >/dev/null 2>&1")  # Linux 本地播放
            

# 狗狗偵測 + 框線 + 情緒 + 建議
def analyze_image(image_np):
    results = model.predict(image_np, classes=[16], verbose=False)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    pose = "sit"  # 簡化
    emotion_map = {"sit": "放鬆", "stand": "警戒", "lay": "休息"}
    emotion = emotion_map.get(pose, "未知")

    suggestions = {
        "放鬆": "牠現在很放鬆，可以輕鬆互動",
        "警戒": "牠有點警戒，建議保持距離",
        "休息": "牠在休息，請不要打擾"
    }
    suggestion = suggestions.get(emotion, "觀察牠的動作")

    cv2.putText(image_np, f"情緒: {emotion}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(image_np, f"建議: {suggestion}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return image_np, emotion, suggestion

# 操作選擇
mode = st.radio("操作模式", ["上傳圖片", "本地 webcam"])

if mode == "上傳圖片":
    uploaded_file = st.file_uploader("上傳狗狗圖片", type=["jpg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="原始圖片", use_column_width=True)
        img_np = np.array(image)
        result_img, emotion, suggestion = analyze_image(img_np)
        st.image(result_img, caption="偵測結果", use_column_width=True)
        st.success(f"情緒: {emotion}")
        st.info(f"建議: {suggestion}")
        speak(suggestion)

else:
    if IS_CLOUD:
        st.warning("Cloud 無法直接使用 webcam，本地測試可用")
    else:
        cap = cv2.VideoCapture(0)
        placeholder = st.empty()
        run = st.checkbox("啟動即時 webcam 分析", value=True)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("無法讀取 webcam")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img, emotion, suggestion = analyze_image(frame_rgb)
            placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            speak(suggestion)

        cap.release()
