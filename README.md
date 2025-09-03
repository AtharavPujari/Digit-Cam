# ✍️ Handwritten Digit Recognition using Python, CNN, and OpenCV  

## 📌 Project Description  
This project recognizes handwritten digits (0–9) in real time using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
The system takes input directly from a **webcam feed**, processes the handwritten digit inside a **Region of Interest (ROI)**, and predicts the number live on the video stream.  

---

## ⚙️ How It Works  

### 🔹 Training Phase  
- A CNN model is trained on the **MNIST dataset** (70,000 handwritten digits).  
- The model learns to classify digits **0–9**.  
- After training, the model is saved as **`digit_model.h5`**.  

### 🔹 Real-Time Recognition  
1. Webcam is opened using **OpenCV**.  
2. A box (ROI) is displayed where the user writes a digit.  
3. The digit inside ROI is preprocessed → *(grayscale → resized to 28×28 → normalized)*.  
4. The trained CNN model predicts the digit.  
5. Prediction is displayed **live on the video feed**.  

---

## 🛠️ Tech Stack  
- 🐍 **Python 3**  
- 🧠 **TensorFlow / Keras** (Deep Learning, CNN)  
- 🎥 **OpenCV** (Webcam, Image Preprocessing, Visualization)  
- 🔢 **NumPy & Matplotlib** (Numerical operations, plotting)  

---

## 🚀 How to Run  

### 1️⃣ Clone the Repository  

git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Train the Model

python3 train_model.py
This will train the CNN on MNIST and save the model as digit_model.h5.

### 4️⃣ Run Real-Time Digit Recognition

python3 realtime_digit_recognition.py
A webcam window will open.

Write a digit inside the ROI box.

The prediction will appear live on the screen.

📈 Future Improvements
📷 Collect a custom dataset from the camera to improve real-world accuracy.

🔄 Apply data augmentation (rotation, noise, scaling).

🏗️ Experiment with deeper CNN architectures.

🔤 Extend to recognize letters (EMNIST dataset).

🎯 Output Demo
👉 Example: When you write a digit 3 inside the ROI box:
Prediction: 3
(Displayed directly on the webcam feed)
