import cv2, numpy as np, tensorflow as tf

model = tf.keras.models.load_model('digit_model.h5')
cap = cv2.VideoCapture(0)

THRESH = True   # Always apply thresholding
INVERT = True   # Always invert to match MNIST

while True:
    ok, frame = cap.read()
    if not ok: break

    h, w = frame.shape[:2]
    roi_size = 150  # half of 300
    x1, y1, x2, y2 = w//2-roi_size, h//2-roi_size, w//2+roi_size, h//2+roi_size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]

    # Apply Gaussian blur to reduce noise
    roi = cv2.GaussianBlur(roi, (5,5), 0)

    # Always apply thresholding and inversion
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(roi)

    # Optional: dilate to strengthen strokes
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)

    # Use 56x56 for display and preprocessing
    processed_roi = cv2.resize(roi, (56,56), interpolation=cv2.INTER_AREA)
    # Downscale to 28x28 for model input
    small = cv2.resize(processed_roi, (28,28), interpolation=cv2.INTER_AREA)
    x = small.astype('float32')/255.0
    x = x.reshape(1,28,28,1)
    prob = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(prob))
    conf = float(np.max(prob))

    # UI
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"{digit} ({conf:.2f})  [q:quit]",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Digit Cam", frame)
    # Enlarge processed ROI for display
    display_roi = cv2.resize(processed_roi, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Processed ROI", display_roi)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('t'): THRESH = not THRESH
    if k == ord('i'): INVERT = not INVERT

cap.release()
cv2.destroyAllWindows()
