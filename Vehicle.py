from collections import deque
class Vehicle_Detect():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_boxes = deque(maxlen=12)

