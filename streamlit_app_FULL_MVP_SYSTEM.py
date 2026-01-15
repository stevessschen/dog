import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

# -------------------------------
# Load Models (Auto Download)
# -------------------------------
@st.cache_resource
def load_models():
    dog_model = YOLO("yolov8n.pt")        # auto-download
    pose_model = YOLO("yolov8n-pose.pt")  # auto-download
    return dog_model, pose_model

dog_model, pose_model = load_models()

# -------------------------------
# Session State
# -------------------------------
if "dog_status" not in st.session_state:
    st.session_state.dog_status = "No dog detected yet."

# -------------------------------
# Video Processor
# -------------------------------
class DogAIProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Dog Detection
        results = dog_model(img, conf=0.4, verbose=False)[0]

        dog_found = False

        for box in results.boxes:
            cls = int(box.cls[0])
            name = dog_model.names[cls]

            if name == "dog":
                dog_found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

                # Crop dog for pose
                dog_crop = img[y1:y2, x1:x2]

                if dog_crop.size > 0:
                    pose_results = pose_model(dog_crop, verbose=False)[0]

                    if pose_results.keypoints is not None:
                        for kp in pose_results.keypoints.xy[0]:
                            cv2.circle(
                                img,
                                (int(kp[0]+x1), int(kp[1]+y1)),
                                3, (0,0,255), -1
                            )

                status = "The dog is detected and being analyzed."
                st.session_state.dog_status = status

                cv2.putText(
                    img,
                    status,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        if not dog_found:
            st.session_state.dog_status = "No dog detected yet."

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🐶 DogTalk AI — Real-time Dog Understanding System")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📷 Live Camera")
    webrtc_streamer(
        key="dog-ai",
        video_processor_factory=DogAIProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("🧠 Dog Interpretation")
    status_box = st.empty()

    while True:
        status_box.markdown(
            f"""
            ### 🐕 Current Dog Status
            **{st.session_state.dog_status}**
            """
        )
        st.sleep(0.3)
