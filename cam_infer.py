import cv2, numpy as np, tensorflow as tf
import time

model = tf.keras.models.load_model('digit_model.h5')
cap = cv2.VideoCapture(0)

THRESH = True   # Always apply thresholding
INVERT = True   # Always invert to match MNIST

# Performance optimization variables
last_prediction_time = 0
prediction_interval = 0.2  # Predict every 200ms instead of every frame
cached_result = ("", 0.0)
cached_processed_roi = None

while True:
    ok, frame = cap.read()
    if not ok: break

    h, w = frame.shape[:2]
    roi_size = 150  # half of 300
    x1, y1, x2, y2 = w//2-roi_size, h//2-roi_size, w//2+roi_size, h//2+roi_size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]

    # Only run prediction every prediction_interval seconds
    current_time = time.time()
    should_predict = current_time - last_prediction_time > prediction_interval
    
    if should_predict:
        last_prediction_time = current_time
        
        # Median blur to reduce salt-and-pepper noise
        roi = cv2.medianBlur(roi, 5)

        # Adaptive thresholding for better binarization
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours to detect multiple digits
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and sort by x-coordinate (left to right)
        digit_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Filter out noise (too small) and too large contours
            if area > 100 and w > 10 and h > 20 and w < roi.shape[1]//3:
                digit_contours.append((x, c))
        
        # Sort by x-coordinate to read digits left to right
        digit_contours.sort(key=lambda item: item[0])
        
        digits_result = []
        processed_digits = []
        
        # Limit to max 5 digits to prevent excessive processing
        digit_contours = digit_contours[:5]
        
        for x_pos, c in digit_contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            digit_crop = roi[y:y+h_box, x:x+w_box]
            
            # Center the digit in a square mask
            size = max(w_box, h_box)
            # Add padding around the digit
            size = int(size * 1.2)
            square = np.zeros((size, size), dtype=np.uint8)
            x_offset = (size - w_box) // 2
            y_offset = (size - h_box) // 2
            square[y_offset:y_offset+h_box, x_offset:x_offset+w_box] = digit_crop
            
            # Resize for display and model input
            processed_roi_single = cv2.resize(square, (56, 56), interpolation=cv2.INTER_AREA)
            processed_digits.append(processed_roi_single)
            
            # Downscale to 28x28 for model input
            small = cv2.resize(processed_roi_single, (28, 28), interpolation=cv2.INTER_AREA)
            x_input = small.astype('float32')/255.0
            x_input = x_input.reshape(1, 28, 28, 1)
            prob = model.predict(x_input, verbose=0)[0]
            digit = int(np.argmax(prob))
            conf = float(np.max(prob))
            
            # Only include confident predictions
            if conf > 0.5:
                digits_result.append((digit, conf))
        
        # Combine results and cache them
        if digits_result:
            # Create combined display
            if processed_digits:
                # Concatenate all processed digits horizontally for display
                cached_processed_roi = np.hstack(processed_digits)
            else:
                cached_processed_roi = cv2.resize(roi, (56, 56), interpolation=cv2.INTER_AREA)
            
            # Format the result as a number string
            number_str = ''.join([str(d[0]) for d in digits_result])
            avg_conf = sum([d[1] for d in digits_result]) / len(digits_result)
            cached_result = (number_str, avg_conf)
        else:
            cached_processed_roi = cv2.resize(roi, (56, 56), interpolation=cv2.INTER_AREA)
            cached_result = ("?", 0.0)
    
    # Use cached results for display
    number_str, avg_conf = cached_result

    # UI
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"{number_str} ({avg_conf:.2f})  [q:quit]",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Digit Cam", frame)
    # Enlarge processed ROI for display
    if cached_processed_roi is not None:
        display_roi = cv2.resize(cached_processed_roi, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Processed ROI", display_roi)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('t'): THRESH = not THRESH
    if k == ord('i'): INVERT = not INVERT

cap.release()
cv2.destroyAllWindows()
