from collections import deque
import numpy as np

class FrameBuffer:
    def __init__(self, size=16):
        self.buffer = deque(maxlen=size)
        self.size = size

    def add_frame(self, frame):
        self.buffer.append(frame)

    def is_full(self):
        return len(self.buffer) == self.size

    def get_sequence(self):
        return np.array(self.buffer)
