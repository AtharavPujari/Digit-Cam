# Digit Camera Recognition

A real-time handwritten digit recognition system using webcam input and a trained neural network model.

## Requirements

- Python 3.7 or higher
- Webcam/Camera
- The required Python packages (see installation below)

## Installation

1. Clone or download this project
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure your webcam is connected and working
2. Run the program:
   ```bash
   python cam_infer.py
   ```
3. Write digits in the green rectangle on the screen
4. The program will recognize multiple digits and display the result
5. Press 'q' to quit

## Features

- Real-time multi-digit recognition
- Optimized for smooth performance
- Adaptive preprocessing for better accuracy
- Confidence scoring for predictions

## Files

- `cam_infer.py` - Main application
- `digit_model.h5` - Pre-trained neural network model
- `train_mnist.py` - Script to train the model
- `requirements.txt` - Python dependencies
