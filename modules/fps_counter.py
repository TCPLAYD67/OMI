import time


class FPSCounter:

    def __init__(self):

        self.last_time = time.time()
        self.frames = 0
        self.fps = 0

    def update(self):

        self.frames += 1

        current_time = time.time()

        elapsed = current_time - self.last_time

        if elapsed >= 1.0:

            self.fps = round(self.frames / elapsed)

            self.frames = 0
            self.last_time = current_time

        return self.fps