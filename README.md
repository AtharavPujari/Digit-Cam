# 🔢 Digit-Cam: Real-Time Handwritten Digit Recognition

A powerful, real-time handwritten digit recognition system powered by deep learning. Features multiple interfaces including live webcam recognition, web app with dark mode UI, PDF/image processing, and more!

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- 📹 **Live Webcam Recognition** - Real-time multi-digit detection with visual feedback
- 🌐 **Modern Web Interface** - Beautiful Flask web app with dark mode UI
- 📄 **PDF Processing** - Extract and recognize digits from scanned documents
- 🖼️ **Image Upload** - Process PNG, JPG, JPEG images with line-by-line detection
- 🎯 **High Accuracy** - Trained on MNIST dataset with confidence scoring
- 🚀 **Optimized Performance** - Adaptive preprocessing and morphological operations
- 💻 **Multiple Interfaces** - Choose between OpenCV, Flask, or Streamlit

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera (for live recognition)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AtharavPujari/Digit-Cam.git
   cd Digit-Cam
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### 1️⃣ Live Camera Recognition

Real-time digit recognition using your webcam:

```bash
python cam_infer.py
```

- Write digits in the **green rectangle** on screen
- Recognition updates automatically
- Press **'q'** to quit
- Results printed to console with confidence scores

**Example output:**
```
Recognized: 12345 (avg_conf=0.87)
```

---

### 2️⃣ Flask Web Application (Recommended)

Modern web interface with two modes:

```bash
python webapp_realtime.py
```

Visit `http://localhost:5000` in your browser

#### Features:
- 🎥 **Live Camera Mode** - Real-time streaming and recognition
- 📤 **Upload Mode** - Process images and PDFs
- 🎨 **Dark Theme** - Beautiful, modern UI with glass-morphism effects
- 📊 **Confidence Scores** - See prediction confidence for each digit
- 🖼️ **Visual Feedback** - Annotated results with bounding boxes

---

### 3️⃣ PDF Recognition

Process scanned PDFs with line-by-line digit detection:

```bash
python pdf_infer.py /path/to/scanned.pdf
```

**Example output:**
```
=== Page 1 ===
Line 01: 12345 (avg_conf=0.85)
Line 02: 9087 (avg_conf=0.79)
Line 03: No digits detected
```

---

### 4️⃣ Streamlit Interface

Alternative web UI with simple interface:

```bash
streamlit run streamlit_app.py
```

Upload images or PDFs and see results instantly!

---

## 🏗️ Project Structure

```
Digit-Cam/
├── webapp_realtime.py      # Flask web application (main app)
├── cam_infer.py            # OpenCV live camera recognition
├── pdf_infer.py            # PDF processing script
├── streamlit_app.py        # Streamlit web interface
├── inference_utils.py      # Line detection utilities
├── train_mnist.py          # Model training script
├── digit_model.h5          # Pre-trained CNN model (MNIST)
├── requirements.txt        # Python dependencies
└── templates/              # Flask HTML templates
    ├── index.html          # Live camera page
    └── upload.html         # File upload page
```

---

## 🧠 How It Works

1. **Image Acquisition** - Capture from webcam or load from file
2. **Preprocessing** - Grayscale conversion, median blur, adaptive thresholding
3. **Morphological Operations** - Noise reduction and stroke connection
4. **Line Segmentation** - Detect horizontal lines of text
5. **Digit Segmentation** - Extract individual digits using contours
6. **Recognition** - CNN model predicts each digit (0-9)
7. **Post-processing** - Confidence filtering and result aggregation

---

## 🎯 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Training Dataset**: MNIST (70,000 handwritten digits)
- **Input**: 28×28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Accuracy**: High accuracy with confidence scoring
- **Framework**: TensorFlow/Keras

### Retrain the Model

```bash
python train_mnist.py
```

This will train a new model and save it as `digit_model.h5`.

---

## 🛠️ Technologies Used

- **Python 3.7+** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision and image processing
- **Flask** - Web application framework
- **Streamlit** - Alternative web UI framework
- **PyMuPDF (fitz)** - PDF processing
- **NumPy** - Numerical computations

---

## 📸 Screenshots

### Live Camera Recognition
Real-time digit detection with visual ROI

### Web Application
Modern dark mode interface with dual modes

### Upload Recognition
Line-by-line processing with annotated results

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Atharav Pujari**

- GitHub: [@AtharavPujari](https://github.com/AtharavPujari)
- Repository: [Digit-Cam](https://github.com/AtharavPujari/Digit-Cam)

---

## 🙏 Acknowledgments

- MNIST dataset creators
- TensorFlow and OpenCV communities
- Flask framework developers

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful! ⭐</strong>
</div>
