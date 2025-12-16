from collections import deque
import numpy as np

class EmotionSmoother:
    def __init__(self, window=7):
        self.buffer = deque(maxlen=window)

    def smooth(self, emotion_id):
        self.buffer.append(emotion_id)
        return max(set(self.buffer), key=self.buffer.count)