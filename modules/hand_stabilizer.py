import time


class HandStabilizer:

    def __init__(self, hold_time=0.25):

        # How long to keep the last detected hand
        self.hold_time = hold_time

        self.last_hands = []
        self.last_detection_time = 0

    def update(self, hands):

        current_time = time.time()

        # --------------------------------------
        # Hands detected
        # --------------------------------------

        if len(hands) > 0:

            # Save latest hands
            self.last_hands = hands
            self.last_detection_time = current_time

            return hands

        # --------------------------------------
        # No hands detected
        # --------------------------------------

        if current_time - self.last_detection_time < self.hold_time:

            # Continue using previous hands
            return self.last_hands

        # Lost for too long
        self.last_hands = []

        return []