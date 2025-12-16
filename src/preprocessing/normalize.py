import cv2
import torch
import numpy as np

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    face = np.transpose(face, (2, 0, 1))  # CHW
    tensor = torch.tensor(face)
    return tensor.unsqueeze(0)