import cv2
import torch
import os

from src.face.face_detector import FaceDetector
from src.models.cnn.mobilenet_emotion import MobileNetEmotion
from src.preprocessing.normalize import preprocess_face
from src.realtime.temporal_smoother import EmotionSmoother

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def main():
    detector = FaceDetector()

    # ✅ FIX 1: Correct model class name
    model = MobileNetEmotion()

    # ✅ FIX 2: Correct checkpoint path (MobileNet, not ResNet)
    checkpoint_path = "checkpoints/mobilenet_emotion.pth"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu")
        )
        print("✅ Trained model loaded.")
    else:
        print("⚠️ No trained model found. Running with untrained model.")

    model.eval()

    smoother = EmotionSmoother()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            input_tensor = preprocess_face(face)

            with torch.no_grad():
                logits = model(input_tensor)
                emotion_id = logits.argmax().item()
                emotion_id = smoother.smooth(emotion_id)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                EMOTIONS[emotion_id],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("Advanced Emotion Detection", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()