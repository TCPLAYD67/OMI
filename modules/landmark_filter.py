class LandmarkFilter:

    def __init__(self, alpha=0.35):

        # EMA smoothing factor
        self.alpha = alpha

        # Previous landmarks for every tracked hand
        self.previous = {}

    def update(self, hands):

        for hand in hands:

            hand_id = hand["id"]

            landmarks = hand["landmarks"]

            # First time seeing this hand
            if hand_id not in self.previous:

                self.previous[hand_id] = list(landmarks)

                continue

            previous = self.previous[hand_id]

            filtered = []

            for (old_x, old_y), (new_x, new_y) in zip(
                    previous,
                    landmarks
            ):

                x = int(
                    old_x +
                    self.alpha * (new_x - old_x)
                )

                y = int(
                    old_y +
                    self.alpha * (new_y - old_y)
                )

                filtered.append((x, y))

            hand["landmarks"] = filtered

            self.previous[hand_id] = filtered

        # Remove IDs that disappeared
        active_ids = {hand["id"] for hand in hands}

        remove = []

        for hand_id in self.previous:

            if hand_id not in active_ids:
                remove.append(hand_id)

        for hand_id in remove:
            del self.previous[hand_id]

        return hands