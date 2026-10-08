import cv2


class Effects:

    def __init__(self):

        self.click_effects = []
        self.scroll_effects = []
        self.notifications = []

    # ==================================================
    # CLICK RIPPLE
    # ==================================================

    def click(self, position, color=(0, 255, 0)):

        x, y = position

        self.click_effects.append({
            "x": x,
            "y": y,
            "radius": 8,
            "life": 20,
            "max_life": 20,
            "color": color
        })

    # ==================================================
    # SCROLL EFFECT
    # ==================================================

    def scroll(self, position, direction):

        x, y = position

        self.scroll_effects.append({
            "x": x,
            "y": y,
            "direction": direction,
            "life": 18
        })

    # ==================================================
    # NOTIFICATION
    # ==================================================

    def notify(self, text, color=(255, 255, 255)):

        # Keep only the newest notification.
        self.notifications.clear()

        self.notifications.append({
            "text": text,
            "color": color,
            "life": 35
        })

    # ==================================================
    # DRAW
    # ==================================================

    def draw(self, frame, hands=None):

        # ==========================================
        # CLICK RIPPLE
        # ==========================================

        alive = []

        for ripple in self.click_effects:

            progress = 1 - (
                ripple["life"] / ripple["max_life"]
            )

            radius = int(10 + progress * 35)

            thickness = max(
                1,
                int(5 * (1 - progress))
            )

            base = ripple["color"]

            color = (
                int(base[0] * (1 - progress)),
                int(base[1] * (1 - progress)),
                int(base[2] * (1 - progress))
            )

            cv2.circle(
                frame,
                (ripple["x"], ripple["y"]),
                radius,
                color,
                thickness
            )

            ripple["life"] -= 1

            if ripple["life"] > 0:
                alive.append(ripple)

        self.click_effects = alive

        # ==========================================
        # CURSOR GLOW
        # ==========================================

        if hands:

            for hand in hands:

                index = hand["landmarks"][8]

                cv2.circle(
                    frame,
                    index,
                    12,
                    (255, 255, 0),
                    2
                )

        # ==========================================
        # PINCH GLOW
        # ==========================================

        if hands:

            for hand in hands:

                if hand.get("pinching", False):

                    thumb = hand["landmarks"][4]
                    index = hand["landmarks"][8]

                    cx = (thumb[0] + index[0]) // 2
                    cy = (thumb[1] + index[1]) // 2

                    distance = hand.get(
                        "pinch_distance",
                        35
                    )

                    radius = max(
                        8,
                        int(35 - distance)
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        radius,
                        (0, 255, 0),
                        2
                    )

        # ==========================================
        # SCROLL ARROWS
        # ==========================================

        alive = []

        for effect in self.scroll_effects:

            offset = 18 - effect["life"]

            if effect["direction"] == "up":

                text = "↑"
                y = effect["y"] - offset * 3

            else:

                text = "↓"
                y = effect["y"] + offset * 3

            cv2.putText(
                frame,
                text,
                (effect["x"] - 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            effect["life"] -= 1

            if effect["life"] > 0:
                alive.append(effect)

        self.scroll_effects = alive

        # ==========================================
        # GESTURE LABEL
        # ==========================================

        if hands:

            for hand in hands:

                gesture = hand.get(
                    "gesture",
                    "Unknown"
                )

                x, y = hand["center"]

                cv2.putText(
                    frame,
                    gesture,
                    (x - 30, y - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # ==========================================
        # NOTIFICATION
        # ==========================================

        alive = []

        h, w = frame.shape[:2]

        for note in self.notifications:

            text = note["text"]

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2
            )

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                ((w - tw) // 2 - 15, 25),
                ((w + tw) // 2 + 15, 65),
                (30, 30, 30),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.4,
                frame,
                0.6,
                0
            )

            cv2.putText(
                frame,
                text,
                ((w - tw) // 2, 53),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                note["color"],
                2
            )

            note["life"] -= 1

            if note["life"] > 0:
                alive.append(note)

        self.notifications = alive

        return frame