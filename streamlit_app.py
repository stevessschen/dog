# streamlit_app.py
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

st.title("DogTalk AI MVP")

# 使用者上傳圖片
uploaded_file = st.file_uploader("上傳狗狗照片或截圖", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的圖片", use_column_width=True)

    # 1️⃣ 狗狗偵測
    model = YOLO("yolov8n.pt")  # 輕量 YOLOv8
    results = model.predict(np.array(image), classes=[16])  # COCO class 16 = dog

    if len(results[0].boxes) == 0:
        st.write("找不到狗狗，請換張圖片或調整角度")
    else:
        st.write("偵測到狗狗！")
        # 框線標示
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="偵測結果")

        # 2️⃣ 簡單姿態/情緒判斷 (範例)
        pose = "sit"  # 假設
        emotion_map = {"sit": "放鬆", "stand": "警戒", "lay": "休息"}
        emotion = emotion_map.get(pose, "未知")
        st.write(f"姿態: {pose}, 情緒: {emotion}")

        # 3️⃣ GPT 行為建議 (範例)
        suggestions = {
            "放鬆": "牠現在很放鬆，可以輕鬆互動",
            "警戒": "牠有點警戒，建議保持距離",
            "休息": "牠在休息，請不要打擾"
        }
        suggestion = suggestions.get(emotion, "觀察牠的動作")
        st.write(f"建議: {suggestion}")
