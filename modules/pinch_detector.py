import math


class PinchDetector:

    def __init__(self):

        # Maximum distance to count as a pinch
        self.threshold = 35

        # Minimum difference between the two pinch distances
        # to decide a clear winner
        self.margin = 8

    # ==========================================
    # Detect Pinches
    # ==========================================

    def detect(self, hand):

        landmarks = hand["landmarks"]

        thumb = landmarks[4]
        index = landmarks[8]
        middle = landmarks[12]

        # --------------------------------------
        # Distances
        # --------------------------------------

        index_distance = math.hypot(
            thumb[0] - index[0],
            thumb[1] - index[1]
        )

        middle_distance = math.hypot(
            thumb[0] - middle[0],
            thumb[1] - middle[1]
        )

        hand["index_pinch_distance"] = round(index_distance, 1)
        hand["middle_pinch_distance"] = round(middle_distance, 1)

        # Keep compatibility with renderer/effects
        hand["pinch_distance"] = round(
            min(index_distance, middle_distance),
            1
        )

        # --------------------------------------
        # Decide Winner
        # --------------------------------------

        hand["pinching_index"] = False
        hand["pinching_middle"] = False
        hand["pinching"] = False

        # Clear Index Pinch
        if (
            index_distance < self.threshold
            and index_distance + self.margin < middle_distance
        ):

            hand["pinching_index"] = True
            hand["pinching"] = True

        # Clear Middle Pinch
        elif (
            middle_distance < self.threshold
            and middle_distance + self.margin < index_distance
        ):

            hand["pinching_middle"] = True
            hand["pinching"] = True

        return hand