#!/usr/bin/env python3
"""
Real-time digit recognition web app using Flask
Works exactly like cam_infer.py but in a browser
"""

import os
from flask import Flask, render_template, Response, request
import cv2
import numpy as np

# Force TensorFlow to use CPU and quiet logs to avoid CUDA errors on systems without GPU drivers
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import base64
import io
import fitz  # PyMuPDF
from inference_utils import recognize_lines

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model('digit_model.h5')

def generate_frames():
    """Generate frames exactly like cam_infer.py processes them"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        h, w = frame.shape[:2]
        roi_size = 150  # half of 300
        x1, y1, x2, y2 = w//2-roi_size, h//2-roi_size, w//2+roi_size, h//2+roi_size
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y1:y2, x1:x2].copy()
        
        # Enhanced preprocessing for better accuracy
        roi = cv2.medianBlur(roi, 5)
        
        # Adaptive thresholding for better binarization
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up noise and connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours to detect multiple digits
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours with improved criteria
        digit_contours = []
        for c in contours:
            x, y, w_c, h_c = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect_ratio = w_c / float(h_c) if h_c > 0 else 0
            # Better filtering: area, size, aspect ratio (digits typically 0.2-1.0 aspect ratio)
            if area > 100 and w_c > 10 and h_c > 20 and w_c < roi.shape[1]//3 and 0.2 < aspect_ratio < 2.0:
                digit_contours.append((x, c))
        
        digit_contours.sort(key=lambda item: item[0])
        digit_contours = digit_contours[:5]
        
        # Prepare batch for prediction with better centering
        batch_inputs = []
        for x_pos, c in digit_contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            digit_crop = roi[y:y+h_box, x:x+w_box]
            
            # Better padding (20% margin)
            size = int(max(w_box, h_box) * 1.4)
            square = np.zeros((size, size), dtype=np.uint8)
            x_offset, y_offset = (size - w_box) // 2, (size - h_box) // 2
            square[y_offset:y_offset+h_box, x_offset:x_offset+w_box] = digit_crop
            
            # Use INTER_AREA for downsampling (better quality)
            small = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            batch_inputs.append(small.astype('float32') / 255.0)
        
        # Batch prediction
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
        
        # Draw on frame (GREEN like cam_infer.py)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{number_str} ({avg_conf:.2f})", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def bgr_to_base64(img_bgr: np.ndarray) -> str:
    """Encode BGR image to base64 PNG string."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode('.png', rgb)
    if not success:
        return ''
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload an image or PDF and recognize digits line-by-line."""
    ctx = {
        'results': None,
        'annotated_image_b64': None,
        'filename': None,
        'error': None,
    }
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            ctx['error'] = 'No file selected.'
            return render_template('upload.html', **ctx)
        ctx['filename'] = file.filename
        name_lower = file.filename.lower()

        try:
            if name_lower.endswith('.pdf') or file.mimetype == 'application/pdf':
                # Read PDF first page to image
                data = file.read()
                doc = fitz.open(stream=data, filetype='pdf')
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                doc.close()
            else:
                # Read as image
                file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    ctx['error'] = 'Unsupported image format or corrupted file.'
                    return render_template('upload.html', **ctx)

            # Run line-by-line recognition
            results = recognize_lines(img, model)
            ctx['results'] = results

            # Annotate lines on a copy
            annotated = img.copy()
            for r in results:
                x, y, w, h = r['bbox']
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"Line {r['index']}: {r['number_str']}"
                cv2.putText(annotated, label, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            ctx['annotated_image_b64'] = bgr_to_base64(annotated)
        except Exception as e:
            ctx['error'] = f"Failed to process file: {e}"

    return render_template('upload.html', **ctx)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔢 Digit-Cam Real-time Web App")
    print("="*60)
    print("📹 Starting webcam recognition server...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
