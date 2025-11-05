import os
import streamlit as st
import numpy as np
import cv2
import fitz  # PyMuPDF
# Force TensorFlow to use CPU and quiet logs to avoid CUDA errors on systems without GPU drivers
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

st.set_page_config(page_title="Digit-Cam", page_icon="🔢", layout="wide")

@st.cache_resource
def load_model(path: str = 'digit_model.h5'):
    return tf.keras.models.load_model(path)

model = load_model()

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class DigitRecognitionTransformer(VideoTransformerBase):
    """Real-time video transformer - processes exactly like cam_infer.py"""
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        roi_size = 150
        x1, y1, x2, y2 = w//2-roi_size, h//2-roi_size, w//2+roi_size, h//2+roi_size
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[y1:y2, x1:x2].copy()
        roi = cv2.medianBlur(roi, 5)
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contours = []
        for c in contours:
            x, y, w_c, h_c = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area > 100 and w_c > 10 and h_c > 20 and w_c < roi.shape[1]//3:
                digit_contours.append((x, c))
        
        digit_contours.sort(key=lambda item: item[0])
        digit_contours = digit_contours[:5]
        
        batch_inputs = []
        for x_pos, c in digit_contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            digit_crop = roi[y:y+h_box, x:x+w_box]
            size = int(max(w_box, h_box) * 1.2)
            square = np.zeros((size, size), dtype=np.uint8)
            x_offset, y_offset = (size - w_box) // 2, (size - h_box) // 2
            square[y_offset:y_offset+h_box, x_offset:x_offset+w_box] = digit_crop
            small = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            batch_inputs.append(small.astype('float32')/255.0)
        
        digits_result = []
        if batch_inputs:
            x_batch = np.array(batch_inputs).reshape(-1, 28, 28, 1)
            probs = model.predict(x_batch, verbose=0)
            for prob in probs:
                digit, conf = int(np.argmax(prob)), float(np.max(prob))
                if conf > 0.5:
                    digits_result.append((digit, conf))
        
        if digits_result:
            number_str = ''.join([str(d[0]) for d in digits_result])
            avg_conf = sum([d[1] for d in digits_result]) / len(digits_result)
        else:
            number_str, avg_conf = "?", 0.0
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{number_str} ({avg_conf:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Digit-Cam Frontend 🔢")
st.write("Real-time digit recognition - same preprocessing as cam_infer.py (grayscale → blur → adaptive threshold)")

tab_camera, tab_upload = st.tabs(["Real-time Camera", "Upload file (image/PDF)"])

with tab_camera:
    st.subheader("Real-time webcam recognition")
    st.info("📷 Click 'START' to begin real-time digit recognition (same as cam_infer.py)")
    
    webrtc_streamer(
        key="digit-recognition",
        video_transformer_factory=DigitRecognitionTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    st.markdown("""
    **Instructions:**
    - Click START to begin
    - Hold digits in the green box (center of frame)
    - Predictions shown in real-time with confidence scores
    - Uses exact preprocessing: grayscale → median blur → adaptive threshold → batch prediction
    """)

with tab_upload:
    st.subheader("Upload image or PDF")
    from inference_utils import recognize_lines
    
    uploaded = st.file_uploader("Upload image/PDF", type=["png", "jpg", "jpeg", "pdf"]) 
    if uploaded is not None:
        if uploaded.type == 'application/pdf' or uploaded.name.lower().endswith('.pdf'):
            # Process first page for preview; allow selection
            data = uploaded.read()
            doc = fitz.open(stream=data, filetype='pdf')
            page_index = st.number_input("Page", min_value=1, max_value=len(doc), value=1)
            page = doc[page_index-1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results = recognize_lines(img, model)
            st.subheader("Results")
            if not results:
                st.info("No lines detected.")
            else:
                for r in results:
                    st.write(f"Line {r['index']}: {r['number_str']} (avg_conf={r['avg_conf']:.2f})")
            # Show image
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"PDF Page {page_index}", use_column_width=True)
            doc.close()
        else:
            # Handle image - use line-by-line recognition like PDFs
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Use the same line-by-line recognition as PDFs
            results = recognize_lines(img, model)
            
            # Draw bounding boxes for each line on the image
            img_display = img.copy()
            for r in results:
                x, y, w, h = r['bbox']
                cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Label the line
                cv2.putText(img_display, f"Line {r['index']}: {r['number_str']}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            st.subheader("Results - Line by Line")
            if not results:
                st.info("No lines detected.")
            else:
                for r in results:
                    if r['number_str']:
                        st.write(f"**Line {r['index']}:** {r['number_str']} (avg_conf={r['avg_conf']:.2f})")
                    else:
                        st.write(f"**Line {r['index']}:** _No digits detected_")
            
            st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption=uploaded.name, use_column_width=True)

st.markdown("---")
st.caption("Tip: For best results, ensure good lighting and hold digits steady in the green box.")
