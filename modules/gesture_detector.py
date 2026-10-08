from modules.finger_state import FingerState


class GestureDetector:

    def __init__(self):
        self.finger_state = FingerState()

    def detect(self, hand):

        fingers = self.finger_state.get_states(hand)

        if all(fingers.values()):
            return "Open Palm"

        if not any(fingers.values()):
            return "Fist"

        if (
            fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        ):
            return "Point"

        if (
            fingers["index"]
            and fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        ):
            return "Peace"

        return "Unknown"