# 🚨 Real-Time Accident Detection and Emergency Response System

A Deep Learning-based solution for detecting accidents in real-time video feeds and triggering emergency alerts, developed using Python, OpenCV, and TensorFlow.

## 📌 Overview

This project aims to automatically detect road accidents from dashcam or CCTV footage using a trained deep learning model. Once an accident is detected, an emergency alert can be triggered to notify authorities or emergency contacts.

## 🎯 Features

- 🚘 Real-time accident detection from video frames  
- 🧠 Deep learning model trained on accident/non-accident datasets  
- ⚙️ Frame extraction and preprocessing  
- 🧪 Model inference with high accuracy  
- 📤 Emergency alert placeholder for integration (e.g., Twilio, Email, API)

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib (for visualizations)
- [Optional] Twilio / Email API for alerting

## 🧪 Project Structure

```bash
.
├── dataset/                   # Training and validation images
│   ├── train/
│   └── val/
├── videos/                   # Raw input videos
│   ├── Crash-1500/           # Accident videos
│   └── normal/               # Normal driving videos
├── models/                   # Saved model weights
├── utils/                    # Helper scripts
├── prepare_dataset.py        # Frame extraction and dataset creation
├── train_model.py            # Model training script
├── detection.py              # Model loading and prediction
├── main.py                   # Main app to run real-time detection
├── requirements.txt          # Python dependencies
└── README.md                 # This file

⚠️ Disclaimer
This project is for research and academic purposes only. It is not production-ready and should not be solely relied upon for life-critical decisions.

🧠 How It Works
Frame Extraction: Extracts frames from videos at a specified frame rate.

Data Preparation: Labels frames as "Accident" or "Normal".

Model Training: Trains a CNN model using TensorFlow or any pre-trained architecture.

Inference: Loads a video and detects whether frames indicate an accident.

Emergency Response (Planned): Sends alert if an accident is detected.

🚀 Getting Started
1. Clone the repo:
git clone https://github.com/yourusername/accident-detection-system.git
cd accident-detection-system
2. Install dependencies:
pip install -r requirements.txt
3. Prepare Dataset:
python prepare_dataset.py
4. Train the Model:
python train_model.py
5. Run Real-Time Detection:
python main.py
