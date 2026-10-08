import math


class FingerTracker:

    def __init__(self, history_length=20):

        self.history_length = history_length

        # Separate history for each tracked hand
        self.histories = {}

    def update(self, hand):

        # -----------------------------
        # Determine unique hand key
        # -----------------------------

        if "id" in hand:
            hand_key = hand["id"]
        else:
            # Fallback until hand IDs exist
            hand_key = hand.get("handedness", "Unknown")

        # -----------------------------
        # Create history if necessary
        # -----------------------------

        if hand_key not in self.histories:
            self.histories[hand_key] = []

        history = self.histories[hand_key]

        # -----------------------------
        # Index fingertip
        # -----------------------------

        index = hand["landmarks"][8]

        history.append(index)

        # Limit history length
        if len(history) > self.history_length:
            history.pop(0)

        # -----------------------------
        # Velocity & Speed
        # -----------------------------

        velocity = (0, 0)
        speed = 0.0

        if len(history) >= 2:

            x1, y1 = history[-2]
            x2, y2 = history[-1]

            vx = x2 - x1
            vy = y2 - y1

            velocity = (vx, vy)
            speed = math.hypot(vx, vy)

        # -----------------------------
        # Save results
        # -----------------------------

        hand["finger_history"] = history.copy()
        hand["velocity"] = velocity
        hand["speed"] = round(speed, 2)

        return hand