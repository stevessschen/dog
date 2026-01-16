import streamlit as st
import cv2
import numpy as np
import av
import queue
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

# ----------------------------------------
# Global message queue (thread-safe bridge)
# ----------------------------------------
status_queue = queue.Queue(maxsize=1)

# ----------------------------------------
# Load Models (auto download)
# ----------------------------------------
@st.cache_resource
def load_models():
    dog_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")
    return dog_model, pose_model

dog_model, pose_model = load_models()

# ----------------------------------------
# Video Processor
# ----------------------------------------
class DogAIProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = dog_model(img, conf=0.4, verbose=False)[0]

        dog_found = False
        status_text = "No dog detected yet."

        for box in results.boxes:
            cls = int(box.cls[0])
            name = dog_model.names[cls]

            if name == "dog":
                dog_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

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

                status_text = "🐶 Dog detected — analysing pose"

                cv2.putText(
                    img,
                    status_text,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                break

        # push status into queue (non-blocking)
        if not status_queue.full():
            status_queue.put(status_text)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.set_page_config(layout="wide")
st.title("🐶 DogTalk AI — Real-Time Dog Understanding System")

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

    if "dog_status" not in st.session_state:
        st.session_state.dog_status = "No dog detected yet."

    status_box = st.empty()

    # read latest status from queue
    try:
        while True:
            st.session_state.dog_status = status_queue.get_nowait()
    except queue.Empty:
        pass

    status_box.markdown(
        f"""
        ## 🐕 Dog Status
        **{st.session_state.dog_status}**
        """
    )
