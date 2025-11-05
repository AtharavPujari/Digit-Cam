import sys
import cv2
import numpy as np
import fitz  # PyMuPDF
import tensorflow as tf
from inference_utils import recognize_lines


def pdf_to_images(pdf_path: str, zoom: float = 2.0):
    """Render PDF pages to images (BGR) using PyMuPDF."""
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            # PyMuPDF returns RGB, convert to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield img
    finally:
        doc.close()


def main(pdf_path: str, model_path: str = 'digit_model.h5'):
    model = tf.keras.models.load_model(model_path)
    page_num = 0
    for bgr in pdf_to_images(pdf_path):
        page_num += 1
        results = recognize_lines(bgr, model)
        print(f"\n=== Page {page_num} ===")
        if not results:
            print("No lines detected.")
            continue
        for r in results:
            print(f"Line {r['index']:02d}: {r['number_str']} (avg_conf={r['avg_conf']:.2f})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pdf_infer.py <path-to-pdf> [model_path]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'digit_model.h5'
    main(pdf_path, model_path)
