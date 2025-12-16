emotion-detection
=================
# Real-Time Emotion Detection System

## Overview
This project detects human emotions in real time using a webcam.
It uses a deep learning model trained on the FER-2013 dataset.

## Emotions Detected
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Tech Stack
- Python
- PyTorch
- OpenCV
- MediaPipe
- MobileNet-V2

## Model
- Architecture: MobileNet-V2
- Dataset: FER-2013
- Training: Transfer Learning
- Optimized for Apple Silicon (M2)

## How to Run
```bash
python -m src.realtime.infer_live
# emotion-detection-system
