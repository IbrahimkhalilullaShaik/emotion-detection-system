# 🎭 Real-Time Emotion Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Overview

This project implements a **real-time human emotion detection system** using deep learning and computer vision.  
It captures live video from a webcam, detects faces, and predicts **facial emotions in real time**.

The system is designed to be:
- **Efficient** (runs on Apple Silicon / CPU)
- **Modular** (clean project structure)
- **Reproducible** (training pipeline included)
- **Interview-ready** (professional ML practices)

---

## 🎯 Emotions Detected

The model predicts the following **7 basic human emotions**:

- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😊 Happy  
- 😢 Sad  
- 😲 Surprise  
- 😐 Neutral  

---

## 🧠 Model Architecture

- **Model:** MobileNet-V2 (CNN)
- **Training strategy:** Transfer Learning
- **Pretrained on:** ImageNet
- **Classifier head:** Fully connected layer (7 classes)

### Why MobileNet-V2?
- Lightweight and fast
- Optimized for edge devices
- Ideal for real-time inference
- Performs well on limited hardware (MacBook Air M2)

---

## 📊 Dataset

- **Dataset:** FER-2013 (Facial Expression Recognition)
- **Source:** Kaggle
- **Image format:** Grayscale facial images
- **Original size:** 48×48 pixels
- **Classes:** 7 emotions

### Preprocessing Steps
- CSV → image folder conversion
- Resized to 224×224
- Normalization (ImageNet mean & std)
- Data augmentation (horizontal flip)

---

## ⚙️ Training Details

- **Framework:** PyTorch
- **Loss function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning rate:** 1e-4
- **Batch size:** 16
- **Epochs:** 10
- **Device:** Apple MPS (Metal) / CPU fallback

The trained model is saved as a `.pth` checkpoint and loaded during real-time inference.

---

## 🎥 Real-Time Inference Pipeline

1. Capture frame from webcam (OpenCV)
2. Detect face (MediaPipe)
3. Crop and preprocess face
4. Emotion prediction using trained CNN
5. Temporal smoothing to stabilize predictions
6. Display bounding box + emotion label

---

## 🕒 Temporal Smoothing

To reduce flickering predictions:
- Uses a sliding window majority vote
- Produces smoother and more stable emotion outputs in live video

---

## 🧩 Project Structure

emotion-detection-system/
│
├── src/
│   ├── face/              # Face detection
│   ├── models/            # CNN architectures
│   ├── preprocessing/     # Dataset & normalization
│   ├── training/          # Training pipeline
│   ├── realtime/          # Live inference
│
├── notebooks/             # Experiments & analysis
├── tests/                 # Unit tests
├── config.yaml            # Central configuration
├── requirements.txt       # Dependencies
├── .gitignore
└── README.md

---

## 🧪 Performance

- **Accuracy:** ~65–75% (FER-2013 benchmark)
- **FPS:** ~20–30 FPS on MacBook Air M2
- **Latency:** Low (real-time capable)
- **Memory usage:** Optimized for 8 GB RAM

---

## ▶️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/IbrahimkhalilullaShaik/emotion-detection-system.git
cd emotion-detection-system

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run real-time emotion detection
python -m src.realtime.infer_live

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

⸻

👤 Author

Ibrahim Khalilullah Shaik
Integrated M.Tech – Computer Science
Interests: Deep Learning, Computer Vision, Quantum Computing
