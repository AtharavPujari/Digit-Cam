import cv2
import numpy as np
from typing import List, Tuple, Dict

# Types
DigitPred = Tuple[int, float]
LineResult = Dict[str, object]


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold to binary image with white digits on black background."""
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    # Morphological cleanup for better digit shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return bin_img


def segment_digits(binary_roi: np.ndarray) -> List[np.ndarray]:
    """From a binary ROI (white digits on black), segment individual digits as 28x28-ready crops.

    Returns list of square grayscale digit images (uint8), centered and padded.
    """
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter and sort left-to-right with improved criteria
    items = []
    H, W = binary_roi.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect_ratio = w / float(h) if h > 0 else 0
        # Enhanced filtering: area, size, aspect ratio
        if area > 100 and w > 10 and h > 20 and w < W // 2 and 0.2 < aspect_ratio < 2.0:
            items.append((x, c))
    items.sort(key=lambda t: t[0])

    digit_imgs: List[np.ndarray] = []
    for _, c in items:
        x, y, w, h = cv2.boundingRect(c)
        crop = binary_roi[y:y+h, x:x+w]
        size = int(max(w, h) * 1.4)  # Better padding (20% margin)
        square = np.zeros((size, size), dtype=np.uint8)
        x_off = (size - w) // 2
        y_off = (size - h) // 2
        square[y_off:y_off+h, x_off:x_off+w] = crop
        digit_imgs.append(square)
    return digit_imgs


def predict_digits(digit_imgs: List[np.ndarray], model, conf_threshold: float = 0.5) -> List[DigitPred]:
    """Predict each digit image (square uint8) returning (digit, confidence).
    
    Only returns predictions with confidence >= conf_threshold.
    """
    preds: List[DigitPred] = []
    if not digit_imgs:
        return preds
    # Batch process for speed
    batch = []
    for img in digit_imgs:
        small = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        x = small.astype('float32') / 255.0
        x = x.reshape(28, 28, 1)
        batch.append(x)
    x_batch = np.stack(batch, axis=0)
    prob = model.predict(x_batch, verbose=0)
    for p in prob:
        d = int(np.argmax(p))
        c = float(np.max(p))
        # Only include confident predictions
        if c >= conf_threshold:
            preds.append((d, c))
    return preds


def image_to_lines(gray_or_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    """Split a page image into line ROIs using morphology; returns list of (binary_line_roi, bbox)."""
    bin_img = _binarize(gray_or_bgr)
    H, W = bin_img.shape[:2]
    # Improved morphological operation for better line merging
    kernel_width = max(W // 15, 30)  # Adaptive kernel based on image width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    merged = cv2.dilate(bin_img, kernel, iterations=2)
    merged = cv2.erode(merged, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Stricter filtering for line-sized boxes
        if h > 15 and w > 40 and w > h:  # Lines should be wider than tall
            roi = bin_img[y:y+h, x:x+w]
            lines.append((roi, (x, y, w, h)))
    # Sort top-to-bottom by y
    lines.sort(key=lambda item: item[1][1])
    return lines


def recognize_lines(gray_or_bgr: np.ndarray, model) -> List[LineResult]:
    """Recognize digits per line; returns list of dicts: {index, number_str, avg_conf, bbox, digits} """
    results: List[LineResult] = []
    lines = image_to_lines(gray_or_bgr)
    for idx, (line_roi, bbox) in enumerate(lines, start=1):
        digit_imgs = segment_digits(line_roi)
        preds = predict_digits(digit_imgs, model)
        if preds:
            number_str = ''.join(str(d) for d, _ in preds)
            avg_conf = float(np.mean([c for _, c in preds]))
        else:
            number_str = ""
            avg_conf = 0.0
        results.append({
            'index': idx,
            'number_str': number_str,
            'avg_conf': avg_conf,
            'bbox': bbox,
            'digits': preds,
            'preview_row': digit_imgs,
        })
    return results
