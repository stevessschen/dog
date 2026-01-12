import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="DogTalk AI MVP", layout="centered")
st.title("🐶 DogTalk AI MVP")

# 上傳或使用攝像頭
st.sidebar.header("操作方式")
use_webcam = st.sidebar.checkbox("使用攝像頭", value=False)

if use_webcam:
    st.warning("目前 Streamlit Cloud 不支援直接 webcam，需要本地測試")
else:
    uploaded_file = st.file_uploader("上傳狗狗圖片", type=["jpg", "png"])

# 載入 YOLOv8 模型
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def analyze_image(image: Image.Image):
    img_np = np.array(image)
    results = model.predict(img_np, classes=[16])  # COCO 16 = dog
    if len(results[0].boxes) == 0:
        return None, None

    # 標記框線
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 假設姿態 / 情緒
    pose = "sit"  # 簡化示範
    emotion_map = {"sit": "放鬆", "stand": "警戒", "lay": "休息"}
    emotion = emotion_map.get(pose, "未知")

    # GPT 建議
    suggestions = {
        "放鬆": "牠現在很放鬆，可以輕鬆互動",
        "警戒": "牠有點警戒，建議保持距離",
        "休息": "牠在休息，請不要打擾"
    }
    suggestion = suggestions.get(emotion, "觀察牠的動作")

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), (emotion, suggestion)

# 主流程
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始圖片", use_column_width=True)
    result_img, info = analyze_image(image)
    if result_img is None:
        st.warning("找不到狗狗，請換張圖片或調整角度")
    else:
        st.image(result_img, caption="偵測結果", use_column_width=True)
        st.success(f"情緒: {info[0]}")
        st.info(f"建議: {info[1]}")
